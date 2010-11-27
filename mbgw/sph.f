C       Attribution for LPMN and LGAMA:
C
C       COMPUTATION OF SPECIAL FUNCTIONS
C
C          Shanjie Zhang and Jianming Jin
C
C       Copyrighted but permission granted to use code in programs.
C       Buy their book "Computation of Special Functions", 1996, John Wiley & Sons, Inc.

C       **********************************

        SUBROUTINE LGAMA(KF,X,GL)
C
C       ==================================================
C       Purpose: Compute gamma function Г(x) or ln[Г(x)]
C       Input:   x  --- Argument of Г(x) ( x > 0 )
C                KF --- Function code
C                       KF=1 for Г(x); KF=0 for ln[Г(x)]
C       Output:  GL --- Г(x) or ln[Г(x)]
C       ==================================================
C
        IMPLICIT DOUBLE PRECISION (A-H,O-Z)
        DIMENSION A(10)
        DATA A/8.333333333333333D-02,-2.777777777777778D-03,
     &         7.936507936507937D-04,-5.952380952380952D-04,
     &         8.417508417508418D-04,-1.917526917526918D-03,
     &         6.410256410256410D-03,-2.955065359477124D-02,
     &         1.796443723688307D-01,-1.39243221690590D+00/
        X0=X
        N=0
        IF (X.EQ.1.0.OR.X.EQ.2.0) THEN
           GL=0.0D0
           GO TO 20
        ELSE IF (X.LE.7.0) THEN
           N=INT(7-X)
           X0=X+N
        ENDIF
        X2=1.0D0/(X0*X0)
        XP=6.283185307179586477D0
        GL0=A(10)
        DO 10 K=9,1,-1
10         GL0=GL0*X2+A(K)
        GL=GL0/X0+0.5D0*DLOG(XP)+(X0-.5D0)*DLOG(X0)-X0
        IF (X.LE.7.0) THEN
           DO 15 K=1,N
              GL=GL-DLOG(X0-1.0D0)
15            X0=X0-1.0D0
        ENDIF
20      IF (KF.EQ.1) GL=DEXP(GL)
        RETURN
        END


C       **********************************

        SUBROUTINE LPMN(MM,M,N,X,PM,PD)
C
C       =====================================================
C       Purpose: Compute the associated Legendre functions
C                Pmn(x) and their derivatives Pmn'(x)
C       Input :  x  --- Argument of Pmn(x)
C                m  --- Order of Pmn(x),  m = 0,1,2,...,n
C                n  --- Degree of Pmn(x), n = 0,1,2,...,N
C                mm --- Physical dimension of PM and PD
C       Output:  PM(m,n) --- Pmn(x)
C                PD(m,n) --- Pmn'(x)
C       =====================================================
C
        IMPLICIT DOUBLE PRECISION (P,X)
        DIMENSION PM(0:MM,0:N),PD(0:MM,0:N)
        INTRINSIC MIN
        DO 10 I=0,N
        DO 10 J=0,M
           PM(J,I)=0.0D0
10         PD(J,I)=0.0D0
        PM(0,0)=1.0D0
        IF (N.EQ.0) RETURN
        IF (DABS(X).EQ.1.0D0) THEN
           DO 15 I=1,N
              PM(0,I)=X**I
15            PD(0,I)=0.5D0*I*(I+1.0D0)*X**(I+1)
           DO 20 J=1,N
           DO 20 I=1,M
              IF (I.EQ.1) THEN
                 PD(I,J)=1.0D+300
              ELSE IF (I.EQ.2) THEN
                 PD(I,J)=-0.25D0*(J+2)*(J+1)*J*(J-1)*X**(J+1)
              ENDIF
20         CONTINUE
           RETURN
        ENDIF
        LS=1
        IF (DABS(X).GT.1.0D0) LS=-1
        XQ=DSQRT(LS*(1.0D0-X*X))
        XS=LS*(1.0D0-X*X)
        DO 30 I=1,M
30         PM(I,I)=-LS*(2.0D0*I-1.0D0)*XQ*PM(I-1,I-1)
        DO 35 I=0,MIN(M,N-1)
35         PM(I,I+1)=(2.0D0*I+1.0D0)*X*PM(I,I)
        DO 40 I=0,M
        DO 40 J=I+2,N
           PM(I,J)=((2.0D0*J-1.0D0)*X*PM(I,J-1)-
     &             (I+J-1.0D0)*PM(I,J-2))/(J-I)
40      CONTINUE
        PD(0,0)=0.0D0
        DO 45 J=1,N
45         PD(0,J)=LS*J*(PM(0,J-1)-X*PM(0,J))/XS
        DO 50 I=1,M
        DO 50 J=I,N
           PD(I,J)=LS*I*X*PM(I,J)/XS+(J+I)
     &             *(J-I+1.0D0)/XQ*PM(I-1,J)
50      CONTINUE
        RETURN
        END


      SUBROUTINE sph(x, coefs, m, n, nx, output)
cf2py intent(out) output
cf2py intent(hide) nx, m, n
      DOUBLE PRECISION x(nx,2), output(nx), coefs(n+1,m+1)
      DOUBLE PRECISION theta, phi, pmn(m+1,n+1), pmnd(m+1,n+1)
      DOUBLE PRECISION shm, g1, g2
      INTEGER i,j,k
      DOUBLE PRECISION pi
      PARAMETER (pi=3.141592653589793238462643d0)               
      
      do k=1,nx
          theta = x(k,1)
          phi = x(k,2)
          CALL lpmn(m,m,n,dcos(phi),pmn,pmnd)
          output(k)=0.0D0
          do i=0,n
              do j=0,m
                 if (j.LE.i) then
                    shm = pmn(j+1,i+1)
!                     CALL LGAMA(0,i-j+1,g1)
!                     CALL LGAMA(0,i+j+1,g2)
                    shm = shm * dsqrt((2.0D0*i+1)/4.0D0/pi)
!                     shm = shm * dexp(0.5D0*(g1-g2))
                    shm = shm * dcos(j*theta)
                    output(k) = output(k)+coefs(i+1,j+1)*shm
                 end if
              end do
          end do
      end do
      RETURN
      END


      SUBROUTINE sin(x, coefs, lx, ly m, n, nx, output)
cf2py intent(out) output
cf2py intent(hide) nx, m, n
      DOUBLE PRECISION x(nx,2), output(nx), coefs(n+1,m+1)
      DOUBLE PRECISION inci, incj, lx, ly
      INTEGER i,j,k
      DOUBLE PRECISION pi
      PARAMETER (pi=3.141592653589793238462643d0)               
      
      do k=1,nx
          output(k)=0.0D0
          do j=0,m
              if (mod(j,2).EQ.0) then
                  incj = dcos(j/2*pi*x(2)/ly)
              else
                  incj = dsin((j+1)/2*pi*x(2)/ly)
              end if
              do i=0,n
                  if (mod(j,2).EQ.0) then
                      inci = dcos(i/2*pi*x(1)/lx)
                  else
                      inci = dsin((i+1)/2*pi*x(1)/lx)
                  end if
                  output(k) = output(k)+coefs(i+1,j+1)*inci*incj
              end do
          end do
      end do
      RETURN
      END
