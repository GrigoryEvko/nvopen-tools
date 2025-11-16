// Function: sub_733CF0
// Address: 0x733cf0
//
__int64 __fastcall sub_733CF0(__int64 a1)
{
  _QWORD *i; // rdi
  _QWORD *v2; // rdi
  _QWORD *v3; // rdi
  _QWORD *m; // r14
  _QWORD *v5; // rdi
  _QWORD *n; // r15
  _QWORD *v7; // rdi
  _QWORD *ii; // rbx
  _QWORD *v9; // rdi
  _QWORD *jj; // r12
  _QWORD *v11; // rdi
  _QWORD *kk; // r13
  _QWORD *v13; // rdi
  _QWORD *mm; // rdx
  _QWORD *v15; // rdi
  __int64 nn; // rsi
  _QWORD *v18; // [rsp+8h] [rbp-58h]
  _QWORD *v20; // [rsp+18h] [rbp-48h]
  _QWORD *k; // [rsp+20h] [rbp-40h]
  _QWORD *j; // [rsp+28h] [rbp-38h]

  for ( i = *(_QWORD **)(a1 + 24); i; i = *(_QWORD **)(a1 + 24) )
    sub_733B20(i);
  for ( j = *(_QWORD **)(a1 + 48); j; j = (_QWORD *)j[7] )
  {
    while ( 1 )
    {
      v2 = (_QWORD *)j[3];
      if ( !v2 )
        break;
      sub_733B20(v2);
    }
    for ( k = (_QWORD *)j[6]; k; k = (_QWORD *)k[7] )
    {
      while ( 1 )
      {
        v3 = (_QWORD *)k[3];
        if ( !v3 )
          break;
        sub_733B20(v3);
      }
      for ( m = (_QWORD *)k[6]; m; m = (_QWORD *)m[7] )
      {
        while ( 1 )
        {
          v5 = (_QWORD *)m[3];
          if ( !v5 )
            break;
          sub_733B20(v5);
        }
        for ( n = (_QWORD *)m[6]; n; n = (_QWORD *)n[7] )
        {
          while ( 1 )
          {
            v7 = (_QWORD *)n[3];
            if ( !v7 )
              break;
            sub_733B20(v7);
          }
          for ( ii = (_QWORD *)n[6]; ii; ii = (_QWORD *)ii[7] )
          {
            while ( 1 )
            {
              v9 = (_QWORD *)ii[3];
              if ( !v9 )
                break;
              sub_733B20(v9);
            }
            for ( jj = (_QWORD *)ii[6]; jj; jj = (_QWORD *)jj[7] )
            {
              while ( 1 )
              {
                v11 = (_QWORD *)jj[3];
                if ( !v11 )
                  break;
                sub_733B20(v11);
              }
              for ( kk = (_QWORD *)jj[6]; kk; kk = (_QWORD *)kk[7] )
              {
                while ( 1 )
                {
                  v13 = (_QWORD *)kk[3];
                  if ( !v13 )
                    break;
                  sub_733B20(v13);
                }
                for ( mm = (_QWORD *)kk[6]; mm; mm = (_QWORD *)mm[7] )
                {
                  while ( 1 )
                  {
                    v15 = (_QWORD *)mm[3];
                    if ( !v15 )
                      break;
                    v20 = mm;
                    sub_733B20(v15);
                    mm = v20;
                  }
                  for ( nn = mm[6]; nn; nn = *(_QWORD *)(nn + 56) )
                  {
                    v18 = mm;
                    sub_733CF0(nn, nn);
                    mm = v18;
                  }
                  mm[6] = 0;
                }
                kk[6] = 0;
              }
              jj[6] = 0;
            }
            ii[6] = 0;
          }
          n[6] = 0;
        }
        m[6] = 0;
      }
      k[6] = 0;
    }
    j[6] = 0;
  }
  *(_QWORD *)(a1 + 48) = 0;
  return a1;
}
