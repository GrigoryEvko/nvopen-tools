// Function: sub_1686480
// Address: 0x1686480
//
int __fastcall sub_1686480(_QWORD *a1)
{
  __int64 k; // rbx
  _QWORD *v2; // r12
  __int64 m; // r15
  _QWORD *v4; // r14
  __int64 n; // r13
  __int64 i; // [rsp+10h] [rbp-50h]
  _QWORD *v8; // [rsp+18h] [rbp-48h]
  __int64 j; // [rsp+20h] [rbp-40h]
  _QWORD *v10; // [rsp+28h] [rbp-38h]

  for ( i = 0; i != 16; ++i )
  {
    while ( 1 )
    {
      if ( !*((_BYTE *)a1 + i + 12) )
      {
        v8 = (_QWORD *)a1[i + 4];
        if ( v8 )
          break;
      }
      if ( ++i == 16 )
        return sub_16856A0(a1);
    }
    for ( j = 0; j != 16; ++j )
    {
      while ( 1 )
      {
        if ( !*((_BYTE *)v8 + j + 12) )
        {
          v10 = (_QWORD *)v8[j + 4];
          if ( v10 )
            break;
        }
        if ( ++j == 16 )
          goto LABEL_23;
      }
      for ( k = 0; k != 16; ++k )
      {
        while ( 1 )
        {
          if ( !*((_BYTE *)v10 + k + 12) )
          {
            v2 = (_QWORD *)v10[k + 4];
            if ( v2 )
              break;
          }
          if ( ++k == 16 )
            goto LABEL_22;
        }
        for ( m = 0; m != 16; ++m )
        {
          while ( 1 )
          {
            if ( !*((_BYTE *)v2 + m + 12) )
            {
              v4 = (_QWORD *)v2[m + 4];
              if ( v4 )
                break;
            }
            if ( ++m == 16 )
              goto LABEL_21;
          }
          for ( n = 0; n != 16; ++n )
          {
            if ( !*((_BYTE *)v4 + n + 12) && v4[n + 4] )
              sub_1686480();
          }
          sub_16856A0(v4);
        }
LABEL_21:
        sub_16856A0(v2);
      }
LABEL_22:
      sub_16856A0(v10);
    }
LABEL_23:
    sub_16856A0(v8);
  }
  return sub_16856A0(a1);
}
