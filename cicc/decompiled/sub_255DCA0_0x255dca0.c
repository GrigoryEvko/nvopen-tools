// Function: sub_255DCA0
// Address: 0x255dca0
//
__int64 __fastcall sub_255DCA0(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        unsigned __int8 (__fastcall *a4)(__int64, __int64, bool),
        __int64 a5)
{
  __int64 v5; // rax
  __int64 *v6; // rdx
  __int64 v8; // rcx
  __int64 *v9; // rbx
  __int64 v10; // rax
  __int64 v11; // rdx
  bool v13; // r14
  unsigned int *v14; // r12
  unsigned __int8 v15; // r15
  unsigned int *v16; // rbx
  __int64 v17; // rsi
  __int64 v18; // [rsp+0h] [rbp-80h]
  unsigned __int8 v21; // [rsp+1Fh] [rbp-61h]
  _QWORD v23[2]; // [rsp+30h] [rbp-50h] BYREF
  __int64 *v24; // [rsp+40h] [rbp-40h]
  __int64 *v25; // [rsp+48h] [rbp-38h]

  v21 = *(_BYTE *)(a1 + 393);
  if ( !v21 )
    return v21;
  if ( *(_QWORD *)(a1 + 376) || *(_DWORD *)(a1 + 296) )
    return 0;
  if ( !*(_DWORD *)(a1 + 240) )
    return v21;
  v5 = *(unsigned int *)(a1 + 248);
  v6 = *(__int64 **)(a1 + 232);
  v23[0] = a1 + 224;
  v8 = *(_QWORD *)(a1 + 224);
  v24 = v6;
  v23[1] = v8;
  v25 = &v6[12 * v5];
  sub_255DC40((__int64)v23);
  v9 = v24;
  v18 = *(_QWORD *)(a1 + 232) + 96LL * *(unsigned int *)(a1 + 248);
  if ( v24 == (__int64 *)v18 )
    return v21;
  do
  {
    v10 = *v9;
    v11 = v9[1];
    if ( a2 == 0x7FFFFFFF || a3 == 0x7FFFFFFF )
    {
LABEL_20:
      v13 = 0;
      if ( v10 == a2 )
        v13 = a3 == v11 && a3 != 0x7FFFFFFF && a2 != 0x7FFFFFFF;
LABEL_22:
      if ( v9[11] )
      {
        v14 = (unsigned int *)v9[9];
        v15 = 0;
        v16 = (unsigned int *)(v9 + 7);
      }
      else
      {
        v14 = (unsigned int *)v9[2];
        v15 = v21;
        v16 = &v14[*((unsigned int *)v9 + 6)];
      }
      while ( 1 )
      {
        if ( v16 == v14 )
          goto LABEL_12;
        v17 = *(_QWORD *)(a1 + 96);
        if ( v15 )
          break;
        if ( !a4(a5, 112LL * v14[8] + v17, v13) )
          return 0;
        v14 = (unsigned int *)sub_220EF30((__int64)v14);
      }
      while ( a4(a5, 112LL * *v14 + v17, v13) )
      {
        if ( v16 == ++v14 )
          goto LABEL_12;
        v17 = *(_QWORD *)(a1 + 96);
      }
      return 0;
    }
    if ( v10 == 0x7FFFFFFF )
    {
      v13 = 0;
      goto LABEL_22;
    }
    if ( v11 == 0x7FFFFFFF || a2 < v10 + v11 && v10 < a2 + a3 )
      goto LABEL_20;
LABEL_12:
    v9 = v24 + 12;
    v24 = v9;
    if ( v25 == v9 )
      continue;
    while ( *v9 == 0x7FFFFFFFFFFFFFFFLL )
    {
      if ( v9[1] != 0x7FFFFFFFFFFFFFFFLL )
        goto LABEL_15;
LABEL_32:
      v9 += 12;
      v24 = v9;
      if ( v25 == v9 )
        goto LABEL_16;
    }
    if ( *v9 == 0x7FFFFFFFFFFFFFFELL && v9[1] == 0x7FFFFFFFFFFFFFFELL )
      goto LABEL_32;
LABEL_15:
    v9 = v24;
LABEL_16:
    ;
  }
  while ( (__int64 *)v18 != v9 );
  return v21;
}
