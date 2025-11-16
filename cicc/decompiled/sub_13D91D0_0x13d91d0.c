// Function: sub_13D91D0
// Address: 0x13d91d0
//
_QWORD *__fastcall sub_13D91D0(__int64 a1, __int64 a2, __int64 a3, _QWORD *a4, int a5)
{
  unsigned int v5; // r15d
  __int64 v6; // r14
  __int64 v7; // r12
  __int64 v9; // rax
  __int64 *v10; // rbx
  __int64 *v11; // r12
  _QWORD *v12; // rbx
  _QWORD *v13; // rax
  __int64 v14; // rsi
  unsigned int v18; // [rsp+Ch] [rbp-44h]
  __int64 v19; // [rsp+10h] [rbp-40h]
  __int64 *v20; // [rsp+18h] [rbp-38h]

  if ( !a5 )
    return 0;
  v5 = a1;
  v6 = a2;
  v7 = a3;
  if ( *(_BYTE *)(a2 + 16) == 77 )
  {
    v7 = a2;
    v6 = a3;
  }
  else
  {
    v5 = sub_15FF5D0(a1);
  }
  if ( !sub_13CB700(v6, v7, a4[2]) )
    return 0;
  v9 = 3LL * (*(_DWORD *)(v7 + 20) & 0xFFFFFFF);
  if ( (*(_BYTE *)(v7 + 23) & 0x40) != 0 )
  {
    v10 = *(__int64 **)(v7 - 8);
    v20 = &v10[v9];
  }
  else
  {
    v20 = (__int64 *)v7;
    v10 = (__int64 *)(v7 - v9 * 8);
  }
  if ( v20 == v10 )
    return 0;
  v19 = v7;
  v11 = v10;
  v18 = a5 - 1;
  v12 = 0;
  do
  {
    v14 = *v11;
    if ( *v11 != v19 )
    {
      if ( v5 - 32 <= 9 )
      {
        v13 = (_QWORD *)sub_13D9330(v5, v14, v6, a4, v18);
        if ( !v13 )
          return 0;
      }
      else
      {
        v13 = sub_13D8D60(v5, v14, v6, 0, a4, v18);
        if ( !v13 )
          return 0;
      }
      if ( v12 && v12 != v13 )
        return 0;
      v12 = v13;
    }
    v11 += 3;
  }
  while ( v20 != v11 );
  return v12;
}
