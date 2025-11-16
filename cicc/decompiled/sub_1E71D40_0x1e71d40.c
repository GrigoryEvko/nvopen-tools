// Function: sub_1E71D40
// Address: 0x1e71d40
//
void __fastcall sub_1E71D40(__int64 a1, __int64 a2, char a3)
{
  unsigned __int64 v6; // r15
  __int64 v7; // rdx
  __int64 v8; // rsi
  __int64 v9; // rdi
  __int64 v10; // rax
  bool v11; // zf
  __int64 v12; // rdx
  __int64 v13; // r8
  __int64 v14; // rcx
  unsigned __int64 v15; // rdi
  unsigned __int64 v16; // rsi
  __int64 v17; // rax
  __int64 v18; // rdx
  __int64 v19; // r8
  __int64 v20; // rcx
  __int64 v21; // rsi
  unsigned __int64 v22; // rax
  __int64 i; // rcx
  __int64 v24; // rdi
  __int64 v25; // rcx
  unsigned int v26; // r10d
  __int64 *v27; // rdx
  __int64 v28; // r8
  __int64 v29; // rsi
  unsigned __int64 v30; // rax
  __int64 j; // rcx
  __int64 v32; // rdi
  __int64 v33; // rcx
  unsigned int v34; // r10d
  __int64 *v35; // rdx
  __int64 v36; // r8
  __int64 v37; // rdi
  __int64 v38; // rcx
  __int64 *v39; // r8
  __int64 v40; // rdi
  __int64 v41; // rax
  int v42; // edx
  int v43; // r11d
  int v44; // edx
  int v45; // r9d
  unsigned int *v46; // [rsp+20h] [rbp-170h] BYREF
  __int64 v47; // [rsp+28h] [rbp-168h]
  _BYTE v48[64]; // [rsp+30h] [rbp-160h] BYREF
  _BYTE *v49; // [rsp+70h] [rbp-120h] BYREF
  __int64 v50; // [rsp+78h] [rbp-118h]
  _BYTE v51[64]; // [rsp+80h] [rbp-110h] BYREF
  _BYTE *v52; // [rsp+C0h] [rbp-D0h]
  __int64 v53; // [rsp+C8h] [rbp-C8h]
  _BYTE v54[64]; // [rsp+D0h] [rbp-C0h] BYREF
  _BYTE *v55; // [rsp+110h] [rbp-80h]
  __int64 v56; // [rsp+118h] [rbp-78h]
  _BYTE v57[112]; // [rsp+120h] [rbp-70h] BYREF

  v6 = *(_QWORD *)(a2 + 8);
  v7 = *(_QWORD *)(a1 + 2240);
  if ( !a3 )
  {
    v16 = sub_1E6C1C0(*(_QWORD *)(a1 + 2248), v7);
    if ( v16 == v6 )
    {
      *(_QWORD *)(a1 + 2248) = v6;
    }
    else
    {
      v17 = *(_QWORD *)(a1 + 2240);
      if ( v6 == v17 )
      {
        if ( !v6 )
          BUG();
        if ( (*(_BYTE *)v6 & 4) == 0 && (*(_BYTE *)(v6 + 46) & 8) != 0 )
        {
          do
            v17 = *(_QWORD *)(v17 + 8);
          while ( (*(_BYTE *)(v17 + 46) & 8) != 0 );
        }
        v40 = *(_QWORD *)(v17 + 8);
        *(_QWORD *)(a1 + 2240) = v40;
        v41 = sub_1E6BEE0(v40, v16);
        *(_QWORD *)(a1 + 2240) = v41;
        *(_QWORD *)(a1 + 3352) = v41;
      }
      sub_1E705C0(a1, (unsigned __int64 *)v6, *(__int64 **)(a1 + 2248));
      *(_QWORD *)(a1 + 2248) = v6;
      *(_QWORD *)(a1 + 3840) = v6;
    }
    if ( !*(_BYTE *)(a1 + 2568) )
      return;
    v18 = *(_QWORD *)(a1 + 24);
    v19 = *(unsigned __int8 *)(a1 + 2569);
    v52 = v54;
    v55 = v57;
    v20 = *(_QWORD *)(a1 + 40);
    v49 = v51;
    v50 = 0x800000000LL;
    v53 = 0x800000000LL;
    v56 = 0x800000000LL;
    sub_1EE65F0(&v49, v6, v18, v20, v19, 0);
    if ( !*(_BYTE *)(a1 + 2569) )
    {
      sub_1EE69C0(&v49, v6, *(_QWORD *)(a1 + 2112));
LABEL_36:
      v37 = a1 + 3776;
      if ( *(_QWORD *)(a1 + 3840) != *(_QWORD *)(a1 + 2248) )
      {
        sub_1EE76C0();
        v37 = a1 + 3776;
      }
      v46 = (unsigned int *)v48;
      v47 = 0x800000000LL;
      sub_1EE8590(v37, &v49, &v46);
      sub_1E70E90((_QWORD *)a1, a2, *(_QWORD **)(a1 + 3824));
      sub_1E70F80((_QWORD *)a1, v46, (unsigned int)v47, v38, v39);
      if ( v46 != (unsigned int *)v48 )
        _libc_free((unsigned __int64)v46);
      v15 = (unsigned __int64)v55;
      if ( v55 == v57 )
        goto LABEL_14;
      goto LABEL_13;
    }
    v21 = *(_QWORD *)(a1 + 2112);
    v22 = v6;
    for ( i = *(_QWORD *)(v21 + 272); (*(_BYTE *)(v22 + 46) & 4) != 0; v22 = *(_QWORD *)v22 & 0xFFFFFFFFFFFFFFF8LL )
      ;
    v24 = *(_QWORD *)(i + 368);
    v25 = *(unsigned int *)(i + 384);
    if ( (_DWORD)v25 )
    {
      v26 = (v25 - 1) & (((unsigned int)v22 >> 9) ^ ((unsigned int)v22 >> 4));
      v27 = (__int64 *)(v24 + 16LL * v26);
      v28 = *v27;
      if ( *v27 == v22 )
      {
LABEL_27:
        sub_1EE6D60(&v49, v21, *(_QWORD *)(a1 + 40), v27[1] & 0xFFFFFFFFFFFFFFF8LL | 4, v6);
        goto LABEL_36;
      }
      v44 = 1;
      while ( v28 != -8 )
      {
        v45 = v44 + 1;
        v26 = (v25 - 1) & (v44 + v26);
        v27 = (__int64 *)(v24 + 16LL * v26);
        v28 = *v27;
        if ( *v27 == v22 )
          goto LABEL_27;
        v44 = v45;
      }
    }
    v27 = (__int64 *)(v24 + 16 * v25);
    goto LABEL_27;
  }
  if ( v6 != v7 )
  {
    sub_1E705C0(a1, *(unsigned __int64 **)(a2 + 8), (__int64 *)v7);
    *(_QWORD *)(a1 + 3352) = v6;
    if ( !*(_BYTE *)(a1 + 2568) )
      return;
LABEL_10:
    v12 = *(_QWORD *)(a1 + 24);
    v13 = *(unsigned __int8 *)(a1 + 2569);
    v52 = v54;
    v55 = v57;
    v14 = *(_QWORD *)(a1 + 40);
    v49 = v51;
    v50 = 0x800000000LL;
    v53 = 0x800000000LL;
    v56 = 0x800000000LL;
    sub_1EE65F0(&v49, v6, v12, v14, v13, 0);
    if ( !*(_BYTE *)(a1 + 2569) )
    {
      sub_1EE69C0(&v49, v6, *(_QWORD *)(a1 + 2112));
      goto LABEL_12;
    }
    v29 = *(_QWORD *)(a1 + 2112);
    v30 = v6;
    for ( j = *(_QWORD *)(v29 + 272); (*(_BYTE *)(v30 + 46) & 4) != 0; v30 = *(_QWORD *)v30 & 0xFFFFFFFFFFFFFFF8LL )
      ;
    v32 = *(_QWORD *)(j + 368);
    v33 = *(unsigned int *)(j + 384);
    if ( (_DWORD)v33 )
    {
      v34 = (v33 - 1) & (((unsigned int)v30 >> 9) ^ ((unsigned int)v30 >> 4));
      v35 = (__int64 *)(v32 + 16LL * v34);
      v36 = *v35;
      if ( v30 == *v35 )
      {
LABEL_32:
        sub_1EE6D60(&v49, v29, *(_QWORD *)(a1 + 40), v35[1] & 0xFFFFFFFFFFFFFFF8LL | 4, v6);
LABEL_12:
        sub_1EE80E0(a1 + 3288, &v49);
        sub_1E70E90((_QWORD *)a1, a2, *(_QWORD **)(a1 + 3336));
        v15 = (unsigned __int64)v55;
        if ( v55 == v57 )
        {
LABEL_14:
          if ( v52 != v54 )
            _libc_free((unsigned __int64)v52);
          if ( v49 != v51 )
            _libc_free((unsigned __int64)v49);
          return;
        }
LABEL_13:
        _libc_free(v15);
        goto LABEL_14;
      }
      v42 = 1;
      while ( v36 != -8 )
      {
        v43 = v42 + 1;
        v34 = (v33 - 1) & (v42 + v34);
        v35 = (__int64 *)(v32 + 16LL * v34);
        v36 = *v35;
        if ( *v35 == v30 )
          goto LABEL_32;
        v42 = v43;
      }
    }
    v35 = (__int64 *)(v32 + 16 * v33);
    goto LABEL_32;
  }
  v8 = *(_QWORD *)(a1 + 2248);
  if ( !v6 )
    BUG();
  if ( (*(_BYTE *)v6 & 4) == 0 && (*(_BYTE *)(v6 + 46) & 8) != 0 )
  {
    do
      v7 = *(_QWORD *)(v7 + 8);
    while ( (*(_BYTE *)(v7 + 46) & 8) != 0 );
  }
  v9 = *(_QWORD *)(v7 + 8);
  *(_QWORD *)(a1 + 2240) = v9;
  v10 = sub_1E6BEE0(v9, v8);
  v11 = *(_BYTE *)(a1 + 2568) == 0;
  *(_QWORD *)(a1 + 2240) = v10;
  if ( !v11 )
    goto LABEL_10;
}
