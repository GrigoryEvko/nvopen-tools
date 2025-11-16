// Function: sub_31C1720
// Address: 0x31c1720
//
__int64 __fastcall sub_31C1720(__int64 a1, __int64 *a2, unsigned __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 *v8; // r12
  __int64 *v9; // r14
  __int64 v10; // rax
  unsigned int v11; // esi
  __int64 *v12; // rcx
  __int64 v13; // r10
  __int64 v14; // r8
  unsigned __int64 v15; // rdx
  __int64 v16; // rdx
  __int64 v17; // rdi
  __int64 v18; // r8
  _QWORD *v19; // rax
  __int64 v20; // r9
  __int64 v21; // r15
  __int64 *v22; // rdi
  unsigned int v23; // r14d
  unsigned int v24; // esi
  __int64 v25; // rdi
  int v26; // r11d
  __int64 v27; // r9
  __int64 *v28; // rdx
  unsigned int v29; // r8d
  __int64 *v30; // rax
  __int64 v31; // rcx
  unsigned __int64 v32; // r12
  __int64 *v33; // rax
  unsigned __int64 v34; // rdi
  __int64 v35; // rdx
  int v37; // ecx
  int v38; // r11d
  __int64 *v39; // r12
  __int64 v40; // r8
  size_t v41; // rdx
  __int64 *v42; // rsi
  __int64 *v43; // r14
  __int64 v44; // rdi
  int v45; // eax
  int v46; // eax
  int v47; // ecx
  int v48; // eax
  int v49; // esi
  __int64 v50; // r8
  unsigned int v51; // eax
  __int64 v52; // rdi
  int v53; // r10d
  __int64 *v54; // r9
  int v55; // eax
  int v56; // eax
  __int64 v57; // rdi
  __int64 *v58; // r8
  unsigned int v59; // r12d
  int v60; // r9d
  __int64 v61; // rsi
  __int64 v62; // [rsp+8h] [rbp-68h]
  __int64 v63; // [rsp+8h] [rbp-68h]
  __int64 *v64; // [rsp+10h] [rbp-60h] BYREF
  __int64 v65; // [rsp+18h] [rbp-58h]
  _BYTE src[80]; // [rsp+20h] [rbp-50h] BYREF

  v8 = a2;
  v64 = (__int64 *)src;
  v65 = 0x400000000LL;
  if ( a3 > 4 )
    sub_C8D5F0((__int64)&v64, src, a3, 8u, a5, a6);
  v9 = &a2[a3];
  v10 = (unsigned int)v65;
  if ( v9 != a2 )
  {
    do
    {
      v16 = *(unsigned int *)(a1 + 64);
      v17 = *v8;
      v18 = *(_QWORD *)(a1 + 48);
      if ( (_DWORD)v16 )
      {
        a6 = (unsigned int)(v16 - 1);
        v11 = a6 & (((unsigned int)v17 >> 9) ^ ((unsigned int)v17 >> 4));
        v12 = (__int64 *)(v18 + 16LL * v11);
        v13 = *v12;
        if ( v17 == *v12 )
        {
LABEL_6:
          if ( v12 != (__int64 *)(v18 + 16 * v16) )
          {
            v14 = v12[1];
            v15 = v10 + 1;
            if ( v10 + 1 > (unsigned __int64)HIDWORD(v65) )
              goto LABEL_11;
            goto LABEL_8;
          }
        }
        else
        {
          v37 = 1;
          while ( v13 != -4096 )
          {
            v38 = v37 + 1;
            v11 = a6 & (v37 + v11);
            v12 = (__int64 *)(v18 + 16LL * v11);
            v13 = *v12;
            if ( v17 == *v12 )
              goto LABEL_6;
            v37 = v38;
          }
        }
      }
      v15 = v10 + 1;
      v14 = 0;
      if ( v10 + 1 > (unsigned __int64)HIDWORD(v65) )
      {
LABEL_11:
        v62 = v14;
        sub_C8D5F0((__int64)&v64, src, v15, 8u, v14, a6);
        v10 = (unsigned int)v65;
        v14 = v62;
      }
LABEL_8:
      ++v8;
      v64[v10] = v14;
      v10 = (unsigned int)(v65 + 1);
      LODWORD(v65) = v65 + 1;
    }
    while ( v9 != v8 );
  }
  v19 = (_QWORD *)sub_22077B0(0x30u);
  v21 = (__int64)v19;
  if ( v19 )
  {
    v22 = v19 + 2;
    v23 = v65;
    *v19 = v19 + 2;
    v19[1] = 0x400000000LL;
    if ( v23 )
    {
      v39 = v64;
      v40 = v23;
      if ( v64 != (__int64 *)src )
      {
        v45 = HIDWORD(v65);
        *(_QWORD *)v21 = v64;
        *(_DWORD *)(v21 + 8) = v23;
        *(_DWORD *)(v21 + 12) = v45;
        v64 = (__int64 *)src;
        v65 = 0;
LABEL_34:
        v43 = &v39[v40];
        do
        {
          v44 = *v39++;
          sub_31B8E10(v44, v21);
        }
        while ( v43 != v39 );
        goto LABEL_14;
      }
      v41 = 8LL * v23;
      v42 = (__int64 *)src;
      if ( v23 > 4 )
      {
        sub_C8D5F0((__int64)v19, v22, v23, 8u, v23, v20);
        v39 = *(__int64 **)v21;
        v42 = v64;
        v40 = v23;
        v41 = 8LL * (unsigned int)v65;
        if ( !v41 )
          goto LABEL_33;
        v22 = *(__int64 **)v21;
      }
      v63 = v40;
      memcpy(v22, v42, v41);
      v39 = *(__int64 **)v21;
      v40 = v63;
LABEL_33:
      *(_DWORD *)(v21 + 8) = v23;
      LODWORD(v65) = 0;
      goto LABEL_34;
    }
  }
LABEL_14:
  v24 = *(_DWORD *)(a1 + 240);
  v25 = a1 + 216;
  if ( !v24 )
  {
    ++*(_QWORD *)(a1 + 216);
    goto LABEL_54;
  }
  v26 = 1;
  v27 = *(_QWORD *)(a1 + 224);
  v28 = 0;
  v29 = (v24 - 1) & (((unsigned int)v21 >> 9) ^ ((unsigned int)v21 >> 4));
  v30 = (__int64 *)(v27 + 16LL * v29);
  v31 = *v30;
  if ( *v30 == v21 )
  {
LABEL_16:
    v32 = v30[1];
    v30[1] = v21;
    if ( v32 )
    {
      v33 = *(__int64 **)v32;
      v34 = *(_QWORD *)v32 + 8LL * *(unsigned int *)(v32 + 8);
      if ( *(_QWORD *)v32 != v34 )
      {
        do
        {
          v35 = *v33++;
          *(_QWORD *)(v35 + 32) = 0;
        }
        while ( (__int64 *)v34 != v33 );
        v34 = *(_QWORD *)v32;
      }
      if ( v34 != v32 + 16 )
        _libc_free(v34);
      j_j___libc_free_0(v32);
    }
    goto LABEL_23;
  }
  while ( v31 != -4096 )
  {
    if ( v31 == -8192 && !v28 )
      v28 = v30;
    v29 = (v24 - 1) & (v26 + v29);
    v30 = (__int64 *)(v27 + 16LL * v29);
    v31 = *v30;
    if ( v21 == *v30 )
      goto LABEL_16;
    ++v26;
  }
  if ( !v28 )
    v28 = v30;
  v46 = *(_DWORD *)(a1 + 232);
  ++*(_QWORD *)(a1 + 216);
  v47 = v46 + 1;
  if ( 4 * (v46 + 1) >= 3 * v24 )
  {
LABEL_54:
    sub_31C14D0(v25, 2 * v24);
    v48 = *(_DWORD *)(a1 + 240);
    if ( v48 )
    {
      v49 = v48 - 1;
      v50 = *(_QWORD *)(a1 + 224);
      v51 = (v48 - 1) & (((unsigned int)v21 >> 9) ^ ((unsigned int)v21 >> 4));
      v47 = *(_DWORD *)(a1 + 232) + 1;
      v28 = (__int64 *)(v50 + 16LL * v51);
      v52 = *v28;
      if ( *v28 != v21 )
      {
        v53 = 1;
        v54 = 0;
        while ( v52 != -4096 )
        {
          if ( v52 == -8192 && !v54 )
            v54 = v28;
          v51 = v49 & (v53 + v51);
          v28 = (__int64 *)(v50 + 16LL * v51);
          v52 = *v28;
          if ( v21 == *v28 )
            goto LABEL_48;
          ++v53;
        }
        if ( v54 )
          v28 = v54;
      }
      goto LABEL_48;
    }
    goto LABEL_77;
  }
  if ( v24 - *(_DWORD *)(a1 + 236) - v47 <= v24 >> 3 )
  {
    sub_31C14D0(v25, v24);
    v55 = *(_DWORD *)(a1 + 240);
    if ( v55 )
    {
      v56 = v55 - 1;
      v57 = *(_QWORD *)(a1 + 224);
      v58 = 0;
      v59 = v56 & (((unsigned int)v21 >> 9) ^ ((unsigned int)v21 >> 4));
      v60 = 1;
      v47 = *(_DWORD *)(a1 + 232) + 1;
      v28 = (__int64 *)(v57 + 16LL * v59);
      v61 = *v28;
      if ( v21 != *v28 )
      {
        while ( v61 != -4096 )
        {
          if ( !v58 && v61 == -8192 )
            v58 = v28;
          v59 = v56 & (v60 + v59);
          v28 = (__int64 *)(v57 + 16LL * v59);
          v61 = *v28;
          if ( v21 == *v28 )
            goto LABEL_48;
          ++v60;
        }
        if ( v58 )
          v28 = v58;
      }
      goto LABEL_48;
    }
LABEL_77:
    ++*(_DWORD *)(a1 + 232);
    BUG();
  }
LABEL_48:
  *(_DWORD *)(a1 + 232) = v47;
  if ( *v28 != -4096 )
    --*(_DWORD *)(a1 + 236);
  *v28 = v21;
  v28[1] = v21;
LABEL_23:
  if ( v64 != (__int64 *)src )
    _libc_free((unsigned __int64)v64);
  return v21;
}
