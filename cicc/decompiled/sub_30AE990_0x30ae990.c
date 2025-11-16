// Function: sub_30AE990
// Address: 0x30ae990
//
__int64 __fastcall sub_30AE990(__int64 a1, int a2, __int64 **a3, __int64 a4)
{
  __int64 v5; // r12
  __int64 v6; // r8
  __int64 v7; // r12
  __int64 v8; // r12
  __int64 v9; // r10
  __int64 v10; // rbx
  unsigned __int8 **v11; // rdx
  int v12; // ecx
  unsigned __int8 **v13; // r11
  unsigned __int64 v14; // rax
  __int64 v15; // rax
  unsigned __int64 v16; // rdi
  __int64 v17; // r12
  __int64 v19; // rdx
  __int64 v20; // [rsp+8h] [rbp-E8h]
  unsigned __int8 **v21; // [rsp+20h] [rbp-D0h] BYREF
  __int64 v22; // [rsp+28h] [rbp-C8h]
  char v23[8]; // [rsp+30h] [rbp-C0h] BYREF
  char *v24; // [rsp+38h] [rbp-B8h]
  char v25; // [rsp+48h] [rbp-A8h] BYREF
  char *v26; // [rsp+68h] [rbp-88h]
  char v27; // [rsp+78h] [rbp-78h] BYREF

  if ( *(_BYTE *)a1 == 85 )
  {
    v19 = *(_QWORD *)(a1 - 32);
    if ( v19 )
    {
      if ( !*(_BYTE *)v19
        && *(_QWORD *)(v19 + 24) == *(_QWORD *)(a1 + 80)
        && (*(_BYTE *)(v19 + 33) & 0x20) != 0
        && dword_502E5A8 )
      {
        sub_DF86E0((__int64)&v21, *(_DWORD *)(v19 + 36), (unsigned __int8 *)a1, 0, 1, dword_502E5A8 == 2, a4);
        v17 = sub_DFD690((__int64)a3, (__int64)&v21);
        if ( v26 != &v27 )
          _libc_free((unsigned __int64)v26);
        v16 = (unsigned __int64)v24;
        if ( v24 == &v25 )
          return v17;
LABEL_11:
        _libc_free(v16);
        return v17;
      }
    }
  }
  v5 = 32LL * (*(_DWORD *)(a1 + 4) & 0x7FFFFFF);
  if ( (*(_BYTE *)(a1 + 7) & 0x40) != 0 )
  {
    v6 = *(_QWORD *)(a1 - 8);
    v7 = v6 + v5;
  }
  else
  {
    v6 = a1 - v5;
    v7 = a1;
  }
  v8 = v7 - v6;
  v21 = (unsigned __int8 **)v23;
  v9 = v8 >> 5;
  v22 = 0x400000000LL;
  v10 = v8 >> 5;
  if ( (unsigned __int64)v8 > 0x80 )
  {
    v20 = v6;
    sub_C8D5F0((__int64)&v21, v23, v8 >> 5, 8u, v6, (__int64)v23);
    v13 = v21;
    v12 = v22;
    v9 = v8 >> 5;
    v6 = v20;
    v11 = &v21[(unsigned int)v22];
  }
  else
  {
    v11 = (unsigned __int8 **)v23;
    v12 = 0;
    v13 = (unsigned __int8 **)v23;
  }
  if ( v8 > 0 )
  {
    v14 = 0;
    do
    {
      v11[v14 / 8] = *(unsigned __int8 **)(v6 + 4 * v14);
      v14 += 8LL;
      --v10;
    }
    while ( v10 );
    v13 = v21;
    v12 = v22;
  }
  LODWORD(v22) = v9 + v12;
  v15 = sub_DFCEF0(a3, (unsigned __int8 *)a1, v13, (unsigned int)(v9 + v12), a2);
  v16 = (unsigned __int64)v21;
  v17 = v15;
  if ( v21 != (unsigned __int8 **)v23 )
    goto LABEL_11;
  return v17;
}
