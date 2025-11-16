// Function: sub_321B470
// Address: 0x321b470
//
__int64 __fastcall sub_321B470(__int64 a1, __int64 a2)
{
  __int64 v3; // r14
  __int64 v4; // r8
  char v5; // dl
  _BYTE *v6; // r9
  __int64 v7; // rbx
  char *v8; // r10
  char *v9; // rbx
  int *v10; // rax
  bool v11; // dl
  int v12; // ecx
  const __m128i *v13; // r12
  __int64 v14; // rdx
  char *v15; // rcx
  __m128i *v16; // rdx
  char v17; // dl
  __int64 v18; // rdx
  const __m128i *v19; // r12
  char *v20; // rcx
  __int64 v21; // rdx
  __m128i *v22; // rdx
  __int64 v23; // rax
  void *v24; // rdi
  void *v25; // r12
  unsigned __int64 v26; // rax
  int v27; // ebx
  size_t v28; // r13
  _BYTE *v29; // rdi
  bool v31; // zf
  __int64 v32; // rax
  int v33; // edx
  int v34; // ecx
  __int64 v35; // rdx
  __int64 v36; // rdx
  signed __int64 v37; // r12
  signed __int64 v38; // r12
  char *v39; // [rsp+0h] [rbp-F0h]
  __m128i *v40; // [rsp+0h] [rbp-F0h]
  char *v41; // [rsp+0h] [rbp-F0h]
  __m128i *v42; // [rsp+0h] [rbp-F0h]
  int *v43; // [rsp+8h] [rbp-E8h]
  char *v44; // [rsp+8h] [rbp-E8h]
  int *v45; // [rsp+8h] [rbp-E8h]
  char *v46; // [rsp+8h] [rbp-E8h]
  _BYTE *v47; // [rsp+10h] [rbp-E0h]
  _BYTE *v48; // [rsp+10h] [rbp-E0h]
  _BYTE *v49; // [rsp+10h] [rbp-E0h]
  _BYTE *v50; // [rsp+10h] [rbp-E0h]
  _BYTE *v51; // [rsp+10h] [rbp-E0h]
  _BYTE *v52; // [rsp+10h] [rbp-E0h]
  char v53; // [rsp+1Fh] [rbp-D1h]
  __int64 v54; // [rsp+20h] [rbp-D0h]
  int v55; // [rsp+30h] [rbp-C0h] BYREF
  __int64 v56; // [rsp+38h] [rbp-B8h]
  int v57; // [rsp+40h] [rbp-B0h]
  int v58; // [rsp+44h] [rbp-ACh]
  void *src; // [rsp+50h] [rbp-A0h] BYREF
  __int64 v60; // [rsp+58h] [rbp-98h]
  _BYTE v61[144]; // [rsp+60h] [rbp-90h] BYREF

  v3 = sub_2E891C0(a2);
  v54 = sub_B0D520(v3);
  v53 = v5 ^ 1;
  if ( v5 )
  {
    if ( *(_WORD *)(a2 + 68) != 14 )
    {
      v6 = v61;
      v3 = v54;
      src = v61;
      v60 = 0x400000000LL;
      goto LABEL_4;
    }
    v9 = *(char **)(a2 + 32);
    v6 = v61;
    src = v61;
    v60 = 0x400000000LL;
    v8 = v9 + 40;
LABEL_24:
    if ( v8 != v9 )
      goto LABEL_5;
LABEL_25:
    v32 = (unsigned int)v60;
    v24 = (void *)(a1 + 24);
    *(_QWORD *)a1 = v3;
    *(_QWORD *)(a1 + 8) = a1 + 24;
    v25 = src;
    *(_QWORD *)(a1 + 16) = 0x200000000LL;
    v28 = 24 * v32;
    v27 = v32;
    if ( 24 * v32 )
      goto LABEL_26;
    goto LABEL_19;
  }
  v6 = v61;
  v31 = *(_WORD *)(a2 + 68) == 14;
  src = v61;
  v60 = 0x400000000LL;
  if ( v31 )
  {
    v9 = *(char **)(a2 + 32);
    v8 = v9 + 40;
    goto LABEL_24;
  }
LABEL_4:
  v7 = *(_QWORD *)(a2 + 32);
  v8 = (char *)(v7 + 40LL * (*(_DWORD *)(a2 + 40) & 0xFFFFFF));
  v9 = (char *)(v7 + 80);
  if ( v8 == v9 )
    goto LABEL_25;
LABEL_5:
  v10 = &v55;
  do
  {
    while ( 1 )
    {
      v17 = *v9;
      if ( !*v9 )
      {
        v11 = 1;
        if ( *(_WORD *)(a2 + 68) == 14 )
          v11 = *(_BYTE *)(*(_QWORD *)(a2 + 32) + 40LL) != 1;
        LOBYTE(v57) = v11;
        v12 = *((_DWORD *)v9 + 2);
        v13 = (const __m128i *)v10;
        v14 = (unsigned int)v60;
        v55 = 0;
        v58 = v12;
        v15 = (char *)src;
        if ( (unsigned __int64)(unsigned int)v60 + 1 <= HIDWORD(v60) )
          goto LABEL_9;
        if ( src > v10 || v10 >= (int *)((char *)src + 24 * (unsigned int)v60) )
          goto LABEL_48;
        goto LABEL_45;
      }
      if ( v17 != 7 )
        break;
      v33 = *((_DWORD *)v9 + 2);
      v34 = *((_DWORD *)v9 + 6);
      v13 = (const __m128i *)v10;
      v55 = 4;
      v58 = v33;
      v14 = (unsigned int)v60;
      v57 = v34;
      v15 = (char *)src;
      if ( (unsigned __int64)(unsigned int)v60 + 1 <= HIDWORD(v60) )
        goto LABEL_9;
      if ( src > v10 || v10 >= (int *)((char *)src + 24 * (unsigned int)v60) )
      {
LABEL_48:
        v42 = (__m128i *)v10;
        v46 = v8;
        v52 = v6;
        sub_C8D5F0((__int64)&src, v6, (unsigned int)v60 + 1LL, 0x18u, v4, (__int64)v6);
        v10 = (int *)v42;
        v15 = (char *)src;
        v14 = (unsigned int)v60;
        v8 = v46;
        v6 = v52;
        v13 = v42;
        goto LABEL_9;
      }
LABEL_45:
      v38 = (char *)v10 - (_BYTE *)src;
      v41 = v8;
      v45 = v10;
      v51 = v6;
      sub_C8D5F0((__int64)&src, v6, (unsigned int)v60 + 1LL, 0x18u, v4, (__int64)v6);
      v15 = (char *)src;
      v14 = (unsigned int)v60;
      v6 = v51;
      v10 = v45;
      v8 = v41;
      v13 = (const __m128i *)((char *)src + v38);
LABEL_9:
      v16 = (__m128i *)&v15[24 * v14];
      *v16 = _mm_loadu_si128(v13);
      v16[1].m128i_i64[0] = v13[1].m128i_i64[0];
      LODWORD(v60) = v60 + 1;
      v9 += 40;
      if ( v8 == v9 )
        goto LABEL_17;
    }
    if ( v17 == 1 )
    {
      v35 = *((_QWORD *)v9 + 3);
      v13 = (const __m128i *)v10;
      v55 = 1;
      v15 = (char *)src;
      v56 = v35;
      v14 = (unsigned int)v60;
      if ( (unsigned __int64)(unsigned int)v60 + 1 <= HIDWORD(v60) )
        goto LABEL_9;
      if ( src > v10 || v10 >= (int *)((char *)src + 24 * (unsigned int)v60) )
        goto LABEL_48;
      goto LABEL_45;
    }
    if ( v17 == 3 )
    {
      v36 = *((_QWORD *)v9 + 3);
      v13 = (const __m128i *)v10;
      v55 = 2;
      v15 = (char *)src;
      v56 = v36;
      v14 = (unsigned int)v60;
      if ( (unsigned __int64)(unsigned int)v60 + 1 <= HIDWORD(v60) )
        goto LABEL_9;
      if ( src > v10 || v10 >= (int *)((char *)src + 24 * (unsigned int)v60) )
        goto LABEL_48;
      goto LABEL_45;
    }
    if ( v17 != 2 )
      BUG();
    v18 = *((_QWORD *)v9 + 3);
    v19 = (const __m128i *)v10;
    v55 = 3;
    v20 = (char *)src;
    v56 = v18;
    v21 = (unsigned int)v60;
    if ( (unsigned __int64)(unsigned int)v60 + 1 > HIDWORD(v60) )
    {
      if ( src > v10 || v10 >= (int *)((char *)src + 24 * (unsigned int)v60) )
      {
        v40 = (__m128i *)v10;
        v44 = v8;
        v50 = v6;
        sub_C8D5F0((__int64)&src, v6, (unsigned int)v60 + 1LL, 0x18u, v4, (__int64)v6);
        v10 = (int *)v40;
        v20 = (char *)src;
        v21 = (unsigned int)v60;
        v8 = v44;
        v6 = v50;
        v19 = v40;
      }
      else
      {
        v37 = (char *)v10 - (_BYTE *)src;
        v39 = v8;
        v43 = v10;
        v49 = v6;
        sub_C8D5F0((__int64)&src, v6, (unsigned int)v60 + 1LL, 0x18u, v4, (__int64)v6);
        v20 = (char *)src;
        v21 = (unsigned int)v60;
        v6 = v49;
        v10 = v43;
        v8 = v39;
        v19 = (const __m128i *)((char *)src + v37);
      }
    }
    v9 += 40;
    v22 = (__m128i *)&v20[24 * v21];
    *v22 = _mm_loadu_si128(v19);
    v22[1].m128i_i64[0] = v19[1].m128i_i64[0];
    LODWORD(v60) = v60 + 1;
  }
  while ( v8 != v9 );
LABEL_17:
  v23 = (unsigned int)v60;
  v24 = (void *)(a1 + 24);
  *(_QWORD *)a1 = v3;
  *(_QWORD *)(a1 + 8) = a1 + 24;
  v25 = src;
  v26 = 3 * v23;
  *(_QWORD *)(a1 + 16) = 0x200000000LL;
  v27 = -1431655765 * v26;
  v28 = 8 * v26;
  if ( v26 > 6 )
  {
    v48 = v6;
    sub_C8D5F0(a1 + 8, (const void *)(a1 + 24), 0xAAAAAAAAAAAAAAABLL * v26, 0x18u, v4, (__int64)v6);
    v6 = v48;
    v24 = (void *)(*(_QWORD *)(a1 + 8) + 24LL * *(unsigned int *)(a1 + 16));
  }
  else if ( !v28 )
  {
    goto LABEL_19;
  }
LABEL_26:
  v47 = v6;
  memcpy(v24, v25, v28);
  LODWORD(v28) = *(_DWORD *)(a1 + 16);
  v6 = v47;
LABEL_19:
  v29 = src;
  *(_DWORD *)(a1 + 16) = v28 + v27;
  *(_BYTE *)(a1 + 72) = v53;
  if ( v29 != v6 )
    _libc_free((unsigned __int64)v29);
  return a1;
}
