// Function: sub_350B7E0
// Address: 0x350b7e0
//
__int64 __fastcall sub_350B7E0(__int64 a1, __int64 a2, __int64 a3)
{
  unsigned int v3; // r11d
  __int64 v6; // rax
  __int64 v7; // rdx
  __int64 v8; // rax
  unsigned __int64 v9; // r12
  __int64 v10; // r15
  __int64 v11; // rdx
  unsigned int v12; // r14d
  unsigned __int64 v13; // rcx
  __int64 v14; // rsi
  unsigned __int64 i; // rax
  __int64 j; // rdi
  __int16 v17; // dx
  unsigned int v18; // ecx
  __int64 v19; // rdi
  unsigned int v20; // esi
  __int64 *v21; // rdx
  __int64 v22; // r8
  __int64 v23; // r8
  __int64 v24; // rsi
  unsigned __int64 v25; // rax
  __int64 k; // r9
  __int16 v27; // dx
  unsigned int v28; // esi
  __int64 *v29; // rdx
  __int64 v30; // r9
  __int64 v31; // rdx
  __int64 v32; // rcx
  _QWORD *v33; // r8
  __int64 v35; // rdx
  __int64 v36; // r8
  __int64 v37; // r9
  __int64 v38; // rax
  int v39; // edx
  int v40; // r10d
  int v41; // edx
  int v42; // r10d
  __int64 v43; // rax
  __int64 v44; // [rsp+0h] [rbp-90h]
  unsigned __int8 v45; // [rsp+17h] [rbp-79h]
  char v47; // [rsp+2Fh] [rbp-61h] BYREF
  char *v48; // [rsp+30h] [rbp-60h] BYREF
  __int64 v49; // [rsp+38h] [rbp-58h]
  _BYTE v50[80]; // [rsp+40h] [rbp-50h] BYREF

  v6 = *(unsigned int *)(a2 + 112);
  v7 = *(_QWORD *)(a1 + 24);
  if ( (int)v6 < 0 )
    v8 = *(_QWORD *)(*(_QWORD *)(v7 + 56) + 16 * (v6 & 0x7FFFFFFF) + 8);
  else
    v8 = *(_QWORD *)(*(_QWORD *)(v7 + 304) + 8 * v6);
  if ( !v8 )
    return 0;
  while ( (*(_BYTE *)(v8 + 4) & 8) != 0 )
  {
    v8 = *(_QWORD *)(v8 + 32);
    if ( !v8 )
      return 0;
  }
  v9 = 0;
  v10 = 0;
LABEL_6:
  v11 = *(_QWORD *)(v8 + 16);
  if ( (*(_BYTE *)(v8 + 3) & 0x10) != 0 )
  {
    if ( (!v10 || v11 == v10) && (*(_BYTE *)(*(_QWORD *)(v11 + 16) + 26LL) & 4) != 0 )
    {
      v10 = *(_QWORD *)(v8 + 16);
      goto LABEL_12;
    }
    return 0;
  }
  if ( (*(_BYTE *)(v8 + 4) & 1) != 0 )
    goto LABEL_12;
  if ( v9 && v11 != v9 || (*(_DWORD *)v8 & 0xFFF00) != 0 )
    return 0;
  v9 = *(_QWORD *)(v8 + 16);
LABEL_12:
  while ( 1 )
  {
    v8 = *(_QWORD *)(v8 + 32);
    if ( !v8 )
      break;
    if ( (*(_BYTE *)(v8 + 4) & 8) == 0 )
      goto LABEL_6;
  }
  LOBYTE(v3) = v9 == 0 || v10 == 0;
  v12 = v3;
  if ( (_BYTE)v3 )
    return 0;
  v13 = v9;
  v14 = *(_QWORD *)(*(_QWORD *)(a1 + 32) + 32LL);
  for ( i = v9; (*(_BYTE *)(i + 44) & 4) != 0; i = *(_QWORD *)i & 0xFFFFFFFFFFFFFFF8LL )
    ;
  if ( (*(_DWORD *)(v9 + 44) & 8) != 0 )
  {
    do
      v13 = *(_QWORD *)(v13 + 8);
    while ( (*(_BYTE *)(v13 + 44) & 8) != 0 );
  }
  for ( j = *(_QWORD *)(v13 + 8); j != i; i = *(_QWORD *)(i + 8) )
  {
    v17 = *(_WORD *)(i + 68);
    if ( (unsigned __int16)(v17 - 14) > 4u && v17 != 24 )
      break;
  }
  v18 = *(_DWORD *)(v14 + 144);
  v19 = *(_QWORD *)(v14 + 128);
  if ( v18 )
  {
    v20 = (v18 - 1) & (((unsigned int)i >> 9) ^ ((unsigned int)i >> 4));
    v21 = (__int64 *)(v19 + 16LL * v20);
    v22 = *v21;
    if ( *v21 == i )
      goto LABEL_24;
    v41 = 1;
    while ( v22 != -4096 )
    {
      v42 = v41 + 1;
      v20 = (v18 - 1) & (v41 + v20);
      v21 = (__int64 *)(v19 + 16LL * v20);
      v22 = *v21;
      if ( *v21 == i )
        goto LABEL_24;
      v41 = v42;
    }
  }
  v21 = (__int64 *)(v19 + 16LL * v18);
LABEL_24:
  v23 = v21[1];
  v24 = v10;
  v25 = v10;
  if ( (*(_DWORD *)(v10 + 44) & 4) != 0 )
  {
    do
      v25 = *(_QWORD *)v25 & 0xFFFFFFFFFFFFFFF8LL;
    while ( (*(_BYTE *)(v25 + 44) & 4) != 0 );
  }
  if ( (*(_DWORD *)(v10 + 44) & 8) != 0 )
  {
    do
      v24 = *(_QWORD *)(v24 + 8);
    while ( (*(_BYTE *)(v24 + 44) & 8) != 0 );
  }
  for ( k = *(_QWORD *)(v24 + 8); k != v25; v25 = *(_QWORD *)(v25 + 8) )
  {
    v27 = *(_WORD *)(v25 + 68);
    if ( (unsigned __int16)(v27 - 14) > 4u && v27 != 24 )
      break;
  }
  if ( !v18 )
  {
LABEL_48:
    v29 = (__int64 *)(v19 + 16LL * v18);
    goto LABEL_34;
  }
  v28 = (v18 - 1) & (((unsigned int)v25 >> 9) ^ ((unsigned int)v25 >> 4));
  v29 = (__int64 *)(v19 + 16LL * v28);
  v30 = *v29;
  if ( v25 != *v29 )
  {
    v39 = 1;
    while ( v30 != -4096 )
    {
      v40 = v39 + 1;
      v28 = (v18 - 1) & (v39 + v28);
      v29 = (__int64 *)(v19 + 16LL * v28);
      v30 = *v29;
      if ( v25 == *v29 )
        goto LABEL_34;
      v39 = v40;
    }
    goto LABEL_48;
  }
LABEL_34:
  if ( !(unsigned __int8)sub_3509FC0((__int64 *)a1, v10, v29[1], v23) )
    return 0;
  v47 = 1;
  v45 = sub_2E8B400(v10, (__int64)&v47, v31, v32, v33);
  if ( !v45 )
    return 0;
  v48 = v50;
  v49 = 0x800000000LL;
  if ( !((unsigned __int16)sub_2E89D80(v9, *(_DWORD *)(a2 + 112), (__int64)&v48) >> 8) )
  {
    v35 = sub_2FDFD70(*(__int64 **)(a1 + 48), v9, v48, (unsigned int)v49, v10);
    if ( v35 )
    {
      v44 = v35;
      sub_2F6A220(*(_QWORD *)(*(_QWORD *)(a1 + 32) + 32LL), v9, v35);
      if ( sub_2E88F60(v9) )
      {
        v43 = sub_2E88D60(v9);
        sub_2E7E910(v43, v9, v44);
      }
      sub_2E88E20(v9);
      sub_2E8F690(v10, *(_DWORD *)(a2 + 112), 0, 0);
      v38 = *(unsigned int *)(a3 + 8);
      if ( v38 + 1 > (unsigned __int64)*(unsigned int *)(a3 + 12) )
      {
        sub_C8D5F0(a3, (const void *)(a3 + 16), v38 + 1, 8u, v36, v37);
        v38 = *(unsigned int *)(a3 + 8);
      }
      v12 = v45;
      *(_QWORD *)(*(_QWORD *)a3 + 8 * v38) = v10;
      ++*(_DWORD *)(a3 + 8);
    }
  }
  if ( v48 != v50 )
    _libc_free((unsigned __int64)v48);
  return v12;
}
