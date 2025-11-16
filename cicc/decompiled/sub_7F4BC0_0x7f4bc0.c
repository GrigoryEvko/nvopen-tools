// Function: sub_7F4BC0
// Address: 0x7f4bc0
//
_QWORD *__fastcall sub_7F4BC0(_QWORD *a1)
{
  _QWORD *k; // r15
  _QWORD *m; // rbx
  _QWORD *n; // r13
  _QWORD *ii; // r12
  _QWORD *jj; // r14
  _QWORD *kk; // rsi
  _QWORD *result; // rax
  _QWORD *v8; // [rsp+8h] [rbp-48h]
  _QWORD *j; // [rsp+10h] [rbp-40h]
  _QWORD *i; // [rsp+18h] [rbp-38h]

  v8 = a1;
  if ( a1 )
  {
    do
    {
      sub_7E9AF0((__int64)v8);
      for ( i = (_QWORD *)v8[20]; i; i = (_QWORD *)*i )
      {
        sub_7E9AF0((__int64)i);
        for ( j = (_QWORD *)i[20]; j; j = (_QWORD *)*j )
        {
          sub_7E9AF0((__int64)j);
          for ( k = (_QWORD *)j[20]; k; k = (_QWORD *)*k )
          {
            sub_7E9AF0((__int64)k);
            for ( m = (_QWORD *)k[20]; m; m = (_QWORD *)*m )
            {
              sub_7E9AF0((__int64)m);
              for ( n = (_QWORD *)m[20]; n; n = (_QWORD *)*n )
              {
                sub_7E9AF0((__int64)n);
                for ( ii = (_QWORD *)n[20]; ii; ii = (_QWORD *)*ii )
                {
                  sub_7E9AF0((__int64)ii);
                  for ( jj = (_QWORD *)ii[20]; jj; jj = (_QWORD *)*jj )
                  {
                    sub_7E9AF0((__int64)jj);
                    for ( kk = (_QWORD *)jj[20]; kk; kk = (_QWORD *)*kk )
                    {
                      sub_7E9AF0((__int64)kk);
                      sub_7F4BC0(kk[20]);
                    }
                  }
                }
              }
            }
          }
        }
      }
      result = (_QWORD *)*v8;
      v8 = result;
    }
    while ( result );
  }
  return result;
}
