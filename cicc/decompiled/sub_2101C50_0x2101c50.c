// Function: sub_2101C50
// Address: 0x2101c50
//
__int64 __fastcall sub_2101C50(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v8; // rax
  __int64 v9; // rdx
  __int64 v10; // rax
  __int64 v11; // rbx
  __int64 v12; // r12
  __int64 v13; // rdx
  unsigned __int64 v14; // rdx
  __int64 v15; // rcx
  __int64 v16; // rdi
  unsigned int v17; // ecx
  unsigned int v18; // esi
  __int64 *v19; // rax
  __int64 v20; // r11
  __int64 v21; // r9
  unsigned __int64 v22; // rax
  unsigned int v23; // esi
  __int64 *v24; // rdx
  int v25; // esi
  unsigned __int16 v26; // ax
  unsigned int v27; // r10d
  __int64 v29; // rax
  __int64 v30; // rdi
  __int64 v31; // rdx
  unsigned int v32; // esi
  __int64 v33; // r8
  unsigned int v34; // ecx
  __int64 *v35; // rax
  __int64 v36; // r9
  __int64 v37; // rcx
  __int64 v38; // r8
  unsigned __int64 v39; // r9
  int v40; // r8d
  int v41; // r9d
  __int64 v42; // rax
  int v43; // edx
  int v44; // eax
  int v45; // r9d
  int v46; // eax
  int v47; // r11d
  bool v48; // [rsp+Bh] [rbp-B5h]
  unsigned __int8 v49; // [rsp+Ch] [rbp-B4h]
  unsigned __int8 v50; // [rsp+Ch] [rbp-B4h]
  int v51; // [rsp+Ch] [rbp-B4h]
  char v52; // [rsp+1Fh] [rbp-A1h] BYREF
  __int64 v53; // [rsp+20h] [rbp-A0h] BYREF
  __int64 v54; // [rsp+28h] [rbp-98h] BYREF
  _BYTE v55[48]; // [rsp+30h] [rbp-90h] BYREF
  char *v56; // [rsp+60h] [rbp-60h] BYREF
  __int64 v57; // [rsp+68h] [rbp-58h]
  _BYTE v58[80]; // [rsp+70h] [rbp-50h] BYREF

  v8 = *(unsigned int *)(a2 + 112);
  v9 = *(_QWORD *)(a1 + 24);
  if ( (int)v8 < 0 )
    v10 = *(_QWORD *)(*(_QWORD *)(v9 + 24) + 16 * (v8 & 0x7FFFFFFF) + 8);
  else
    v10 = *(_QWORD *)(*(_QWORD *)(v9 + 272) + 8 * v8);
  if ( !v10 )
    return 0;
  while ( (*(_BYTE *)(v10 + 4) & 8) != 0 )
  {
    v10 = *(_QWORD *)(v10 + 32);
    if ( !v10 )
      return 0;
  }
  v11 = 0;
  v12 = 0;
LABEL_6:
  v13 = *(_QWORD *)(v10 + 16);
  if ( (*(_BYTE *)(v10 + 3) & 0x10) != 0 )
  {
    if ( (v13 == v11 || !v11) && *(char *)(*(_QWORD *)(v13 + 16) + 9LL) < 0 )
    {
      v11 = *(_QWORD *)(v10 + 16);
      goto LABEL_12;
    }
    return 0;
  }
  if ( (*(_BYTE *)(v10 + 4) & 1) != 0 )
    goto LABEL_12;
  if ( v12 && v13 != v12 || (*(_DWORD *)v10 & 0xFFF00) != 0 )
    return 0;
  v12 = *(_QWORD *)(v10 + 16);
LABEL_12:
  while ( 1 )
  {
    v10 = *(_QWORD *)(v10 + 32);
    if ( !v10 )
      break;
    if ( (*(_BYTE *)(v10 + 4) & 8) == 0 )
      goto LABEL_6;
  }
  if ( v12 == 0 || v11 == 0 )
    return 0;
  v14 = v12;
  v15 = *(_QWORD *)(*(_QWORD *)(a1 + 32) + 272LL);
  if ( (*(_BYTE *)(v12 + 46) & 4) != 0 )
  {
    do
      v14 = *(_QWORD *)v14 & 0xFFFFFFFFFFFFFFF8LL;
    while ( (*(_BYTE *)(v14 + 46) & 4) != 0 );
  }
  v16 = *(_QWORD *)(v15 + 368);
  v17 = *(_DWORD *)(v15 + 384);
  if ( v17 )
  {
    LODWORD(a5) = v17 - 1;
    v18 = (v17 - 1) & (((unsigned int)v14 >> 9) ^ ((unsigned int)v14 >> 4));
    v19 = (__int64 *)(v16 + 16LL * v18);
    v20 = *v19;
    if ( *v19 == v14 )
      goto LABEL_18;
    v44 = 1;
    while ( v20 != -8 )
    {
      v45 = v44 + 1;
      v18 = a5 & (v44 + v18);
      v19 = (__int64 *)(v16 + 16LL * v18);
      v20 = *v19;
      if ( *v19 == v14 )
        goto LABEL_18;
      v44 = v45;
    }
  }
  v19 = (__int64 *)(v16 + 16LL * v17);
LABEL_18:
  v21 = v19[1];
  v22 = v11;
  if ( (*(_BYTE *)(v11 + 46) & 4) != 0 )
  {
    do
      v22 = *(_QWORD *)v22 & 0xFFFFFFFFFFFFFFF8LL;
    while ( (*(_BYTE *)(v22 + 46) & 4) != 0 );
  }
  if ( !v17 )
  {
LABEL_38:
    v24 = (__int64 *)(v16 + 16LL * v17);
    goto LABEL_22;
  }
  v23 = (v17 - 1) & (((unsigned int)v22 >> 9) ^ ((unsigned int)v22 >> 4));
  v24 = (__int64 *)(v16 + 16LL * v23);
  a5 = *v24;
  if ( v22 != *v24 )
  {
    v43 = 1;
    while ( a5 != -8 )
    {
      v23 = (v17 - 1) & (v43 + v23);
      v51 = v43 + 1;
      v24 = (__int64 *)(v16 + 16LL * v23);
      a5 = *v24;
      if ( *v24 == v22 )
        goto LABEL_22;
      v43 = v51;
    }
    goto LABEL_38;
  }
LABEL_22:
  v48 = v12 == 0 || v11 == 0;
  if ( !(unsigned __int8)sub_2100010(a1, v11, v24[1], v21, a5, v21) )
    return 0;
  v52 = 1;
  v49 = sub_1E17B50(v11, 0, &v52);
  if ( !v49 )
    return 0;
  v25 = *(_DWORD *)(a2 + 112);
  v56 = v58;
  v57 = 0x800000000LL;
  v26 = sub_1E166B0(v12, v25, (__int64)&v56);
  v27 = v48;
  if ( !HIBYTE(v26) )
  {
    v29 = sub_1F3B570(*(__int64 **)(a1 + 48), v12, v56, (unsigned int)v57, v11, *(_QWORD *)(a1 + 32));
    v27 = v48;
    v30 = v29;
    if ( v29 )
    {
      v31 = *(_QWORD *)(*(_QWORD *)(a1 + 32) + 272LL);
      v32 = *(_DWORD *)(v31 + 384);
      if ( v32 )
      {
        v33 = *(_QWORD *)(v31 + 368);
        v34 = (v32 - 1) & (((unsigned int)v12 >> 9) ^ ((unsigned int)v12 >> 4));
        v35 = (__int64 *)(v33 + 16LL * v34);
        v36 = *v35;
        if ( v12 == *v35 )
        {
LABEL_42:
          if ( v35 != (__int64 *)(v33 + 16LL * v32) )
          {
            v37 = v35[1];
            *(_QWORD *)((v37 & 0xFFFFFFFFFFFFFFF8LL) + 16) = v30;
            *v35 = -16;
            --*(_DWORD *)(v31 + 376);
            ++*(_DWORD *)(v31 + 380);
            v53 = v30;
            v54 = v37;
            sub_20EB8E0((__int64)v55, v31 + 360, &v53, &v54);
          }
        }
        else
        {
          v46 = 1;
          while ( v36 != -8 )
          {
            v47 = v46 + 1;
            v34 = (v32 - 1) & (v46 + v34);
            v35 = (__int64 *)(v33 + 16LL * v34);
            v36 = *v35;
            if ( v12 == *v35 )
              goto LABEL_42;
            v46 = v47;
          }
        }
      }
      sub_1E16240(v12);
      sub_1E1B440(v11, *(_DWORD *)(a2 + 112), 0, 0, v38, v39);
      v42 = *(unsigned int *)(a3 + 8);
      if ( (unsigned int)v42 >= *(_DWORD *)(a3 + 12) )
      {
        sub_16CD150(a3, (const void *)(a3 + 16), 0, 8, v40, v41);
        v42 = *(unsigned int *)(a3 + 8);
      }
      v27 = v49;
      *(_QWORD *)(*(_QWORD *)a3 + 8 * v42) = v11;
      ++*(_DWORD *)(a3 + 8);
    }
  }
  if ( v56 != v58 )
  {
    v50 = v27;
    _libc_free((unsigned __int64)v56);
    return v50;
  }
  return v27;
}
