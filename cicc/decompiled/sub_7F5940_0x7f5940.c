// Function: sub_7F5940
// Address: 0x7f5940
//
_BYTE *__fastcall sub_7F5940(__int64 a1, __int64 *a2, __int64 a3, int *a4)
{
  __int64 j; // rax
  __int64 i; // r12
  __int64 v6; // rbx
  const __m128i *v7; // r14
  _QWORD *v8; // rsi
  __int64 v9; // rax
  __int64 k; // r15
  _BYTE *result; // rax
  _BYTE *v12; // rdx
  __int64 v13; // r13
  __int64 v14; // rax
  __m128i *v15; // rbx
  _BYTE *v16; // rax
  __int64 v17; // r8
  __int64 v18; // r9
  int v20[13]; // [rsp+1Ch] [rbp-34h] BYREF

  if ( a3 )
  {
    j = *(_QWORD *)(a3 + 128);
    for ( i = a3; *(_BYTE *)(j + 140) == 12; j = *(_QWORD *)(j + 160) )
      ;
    v6 = *(_QWORD *)(j + 168);
    v7 = *(const __m128i **)(a1 + 56);
    *(_BYTE *)(a3 + 171) |= 0x10u;
    v8 = *(_QWORD **)(v6 + 80);
    if ( v8 )
    {
      v9 = sub_7F5750(a3, v8);
      *(_BYTE *)(v9 + 171) |= 0x10u;
      i = v9;
      for ( j = *(_QWORD *)(v9 + 128); *(_BYTE *)(j + 140) == 12; j = *(_QWORD *)(j + 160) )
        ;
    }
    for ( k = *(_QWORD *)(i + 176); k; k = *(_QWORD *)(k + 120) )
    {
      if ( (*(_BYTE *)(k + 171) & 0x20) == 0 )
        break;
    }
    result = (_BYTE *)sub_72FD90(*(_QWORD *)(j + 160), 3);
    v12 = result;
    if ( result )
    {
      v13 = 0;
      while ( *((_QWORD *)v12 + 16) != *(_QWORD *)(v6 + 72) )
      {
        if ( k )
        {
          v14 = *(_QWORD *)(k + 120);
          if ( v14 )
          {
            while ( (*(_BYTE *)(v14 + 171) & 0x20) != 0 )
            {
              v14 = *(_QWORD *)(v14 + 120);
              if ( !v14 )
                goto LABEL_21;
            }
            v13 = k;
            k = v14;
          }
          else
          {
LABEL_21:
            v13 = k;
            k = 0;
          }
        }
        result = (_BYTE *)sub_72FD90(*((_QWORD *)v12 + 14), 3);
        v12 = result;
        if ( !result )
          return result;
      }
      v20[0] = 0;
      if ( (*(_BYTE *)(i - 8) & 1) != 0 )
        sub_7296C0(v20);
      v15 = sub_740630(v7);
      sub_760760((__int64)v15, 2, i, 0);
      sub_729730(v20[0]);
      if ( v13 )
      {
        result = *(_BYTE **)(v13 + 120);
        v15[7].m128i_i64[1] = (__int64)result;
        *(_QWORD *)(v13 + 120) = v15;
      }
      else
      {
        result = *(_BYTE **)(i + 176);
        v15[7].m128i_i64[1] = (__int64)result;
        *(_QWORD *)(i + 176) = v15;
      }
      if ( *(_QWORD *)(i + 184) == v13 )
        *(_QWORD *)(i + 184) = v15;
    }
  }
  else
  {
    v16 = sub_7E45A0(a2);
    return sub_7E6A80(v16, 0x49u, a1, a4, v17, v18);
  }
  return result;
}
