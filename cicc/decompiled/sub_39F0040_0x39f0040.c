// Function: sub_39F0040
// Address: 0x39f0040
//
void __fastcall sub_39F0040(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // r12
  __int64 v4; // rbx
  __int64 v5; // r15
  __int64 v6; // r14
  __int64 v7; // rsi
  __int64 v8; // rax
  __int64 v9; // r14
  __int64 v10; // rax
  __int64 v11; // r8
  __int64 v12; // r15
  int v13; // r9d
  __int64 v14; // rax
  __int64 v15; // rsi
  unsigned __int64 v16; // r8
  __int64 *v17; // rdx
  __int64 v18; // rcx
  __int64 v19; // rax
  unsigned __int64 v20; // r14
  const void *v21; // rsi
  __int64 v22; // rbx
  __int64 v23; // r12
  __int64 v24; // r13
  const __m128i *v25; // r15
  __int64 v26; // rdx
  __m128i *v27; // rdx
  __int64 v28; // rdi
  __int64 v29; // rdx
  size_t v30; // r15
  void *v31; // r9
  int v32; // eax
  __int64 v33; // rax
  __int64 v34; // rax
  __int64 v35; // rax
  unsigned __int64 v36; // r8
  unsigned __int64 v37; // rdi
  unsigned __int64 v38; // rdi
  __int64 v39; // rax
  __int64 v40; // rax
  __int64 v41; // r14
  __int64 v42; // rbx
  int v43; // r8d
  int v44; // r9d
  __int64 v45; // rax
  __int64 v46; // rsi
  __int64 *v47; // rdx
  __int64 v48; // rcx
  __int64 v49; // rax
  __int64 v50; // rdi
  __int64 v51; // rdx
  size_t v52; // r12
  void *v53; // r15
  int v54; // eax
  void *v55; // [rsp+18h] [rbp-208h]
  __int64 v56; // [rsp+20h] [rbp-200h]
  _QWORD *v57; // [rsp+20h] [rbp-200h]
  __int64 v58; // [rsp+20h] [rbp-200h]
  unsigned __int64 v59; // [rsp+20h] [rbp-200h]
  unsigned __int64 v60; // [rsp+20h] [rbp-200h]
  unsigned __int64 v61; // [rsp+20h] [rbp-200h]
  __int64 v62; // [rsp+20h] [rbp-200h]
  unsigned __int64 v64; // [rsp+38h] [rbp-1E8h]
  unsigned __int64 v65; // [rsp+38h] [rbp-1E8h]
  _QWORD v66[4]; // [rsp+40h] [rbp-1E0h] BYREF
  int v67; // [rsp+60h] [rbp-1C0h]
  void **p_src; // [rsp+68h] [rbp-1B8h]
  _BYTE *v69; // [rsp+70h] [rbp-1B0h] BYREF
  __int64 v70; // [rsp+78h] [rbp-1A8h]
  _BYTE v71[96]; // [rsp+80h] [rbp-1A0h] BYREF
  void *src; // [rsp+E0h] [rbp-140h] BYREF
  size_t n; // [rsp+E8h] [rbp-138h]
  _BYTE v74[304]; // [rsp+F0h] [rbp-130h] BYREF

  v3 = a1;
  v4 = *(_QWORD *)(a1 + 264);
  v69 = v71;
  v70 = 0x400000000LL;
  src = v74;
  n = 0x10000000000LL;
  v66[0] = &unk_49EFC48;
  p_src = &src;
  v67 = 1;
  memset(&v66[1], 0, 24);
  sub_16E7A40((__int64)v66, 0, 0, 0);
  (*(void (__fastcall **)(_QWORD, __int64, _QWORD *, _BYTE **, __int64))(**(_QWORD **)(v4 + 16) + 24LL))(
    *(_QWORD *)(v4 + 16),
    a2,
    v66,
    &v69,
    a3);
  if ( (_DWORD)v70 )
  {
    v5 = 0;
    v6 = 24LL * (unsigned int)v70;
    do
    {
      v7 = *(_QWORD *)&v69[v5];
      v5 += 24;
      sub_39EFF60(a1, v7);
    }
    while ( v6 != v5 );
  }
  if ( !*(_DWORD *)(v4 + 480) )
  {
    v16 = sub_38D4BB0(a1, a3);
LABEL_20:
    if ( (_DWORD)v70 )
    {
      v20 = v16;
      v58 = v4;
      v21 = (const void *)(v16 + 128);
      v22 = 0;
      v23 = 24LL * (unsigned int)v70;
      v24 = v16 + 112;
      do
      {
        v25 = (const __m128i *)&v69[v22];
        *(_DWORD *)&v69[v22 + 8] += *(_DWORD *)(v20 + 72);
        v26 = *(unsigned int *)(v20 + 120);
        if ( (unsigned int)v26 >= *(_DWORD *)(v20 + 124) )
        {
          sub_16CD150(v24, v21, 0, 24, v16, v13);
          v26 = *(unsigned int *)(v20 + 120);
        }
        v22 += 24;
        v27 = (__m128i *)(*(_QWORD *)(v20 + 112) + 24 * v26);
        *v27 = _mm_loadu_si128(v25);
        v27[1].m128i_i64[0] = v25[1].m128i_i64[0];
        ++*(_DWORD *)(v20 + 120);
      }
      while ( v22 != v23 );
      v4 = v58;
      v3 = a1;
      v16 = v20;
    }
    v28 = *(unsigned int *)(v16 + 72);
    v29 = *(unsigned int *)(v16 + 76);
    *(_BYTE *)(v16 + 17) = 1;
    v30 = (unsigned int)n;
    v31 = src;
    *(_QWORD *)(v16 + 56) = a3;
    v32 = v28;
    if ( v30 > v29 - v28 )
    {
      v55 = v31;
      v61 = v16;
      sub_16CD150(v16 + 64, (const void *)(v16 + 80), v30 + v28, 1, v16, (int)v31);
      v16 = v61;
      v31 = v55;
      v28 = *(unsigned int *)(v61 + 72);
      v32 = *(_DWORD *)(v61 + 72);
    }
    if ( (_DWORD)v30 )
    {
      v59 = v16;
      memcpy((void *)(*(_QWORD *)(v16 + 64) + v28), v31, v30);
      v16 = v59;
      v32 = *(_DWORD *)(v59 + 72);
    }
    *(_DWORD *)(v16 + 72) = v32 + v30;
    if ( *(_DWORD *)(v4 + 480) )
    {
      if ( (*(_BYTE *)(v4 + 484) & 1) != 0 )
      {
        v62 = v16;
        if ( !sub_39EF7F0(v3) )
        {
          v35 = sub_38D4BB0(v3, a3);
          sub_39EFA40(v3, v35, v62);
          v36 = v62;
          v37 = *(_QWORD *)(v62 + 112);
          if ( v37 != v62 + 128 )
          {
            _libc_free(v37);
            v36 = v62;
          }
          v38 = *(_QWORD *)(v36 + 64);
          if ( v38 != v36 + 80 )
          {
            v64 = v36;
            _libc_free(v38);
            v36 = v64;
          }
          v65 = v36;
          nullsub_1930();
          j_j___libc_free_0(v65);
        }
      }
    }
    goto LABEL_32;
  }
  v8 = *(unsigned int *)(a1 + 120);
  v9 = 0;
  if ( (_DWORD)v8 )
    v9 = *(_QWORD *)(*(_QWORD *)(a1 + 112) + 32 * v8 - 32);
  if ( (*(_BYTE *)(v4 + 484) & 1) != 0 )
  {
    if ( sub_39EF7F0(a1) )
    {
      v16 = *(_QWORD *)(*(_QWORD *)(a1 + 328) + 8LL * *(unsigned int *)(a1 + 336) - 8);
      v34 = *(_QWORD *)(v16 + 56);
      if ( a3 != v34 && v34 )
LABEL_45:
        sub_16BD130("A Bundle can only have one Subtarget.", 1u);
LABEL_17:
      if ( *(_DWORD *)(v9 + 36) == 2 )
        *(_BYTE *)(v16 + 48) = 1;
      *(_BYTE *)(v9 + 44) &= ~1u;
      goto LABEL_20;
    }
    if ( (*(_BYTE *)(v4 + 484) & 1) != 0 && !sub_39EF7F0(a1) )
    {
      v33 = sub_22077B0(0xE0u);
      v16 = v33;
      if ( v33 )
      {
        v60 = v33;
        sub_38CF760(v33, 1, 0, 0);
        v16 = v60;
        *(_WORD *)(v60 + 48) = 0;
        *(_QWORD *)(v60 + 64) = v60 + 80;
        *(_QWORD *)(v60 + 72) = 0x2000000000LL;
        *(_QWORD *)(v60 + 112) = v60 + 128;
        *(_QWORD *)(v60 + 56) = 0;
        *(_QWORD *)(v60 + 120) = 0x400000000LL;
      }
      goto LABEL_17;
    }
  }
  if ( sub_39EF7F0(a1) && (*(_BYTE *)(v9 + 44) & 1) == 0 )
  {
    v16 = sub_38D4B30(a1);
    v39 = *(_QWORD *)(v16 + 56);
    if ( v39 && a3 != v39 )
      goto LABEL_45;
    goto LABEL_17;
  }
  if ( sub_39EF7F0(a1) || (_DWORD)v70 )
  {
    v10 = sub_22077B0(0xE0u);
    v11 = v10;
    if ( v10 )
    {
      v56 = v10;
      v12 = v10;
      sub_38CF760(v10, 1, 0, 0);
      v11 = v56;
      *(_WORD *)(v56 + 48) = 0;
      *(_QWORD *)(v56 + 64) = v56 + 80;
      *(_QWORD *)(v56 + 72) = 0x2000000000LL;
      *(_QWORD *)(v56 + 112) = v56 + 128;
      *(_QWORD *)(v56 + 56) = 0;
      *(_QWORD *)(v56 + 120) = 0x400000000LL;
    }
    else
    {
      v12 = 0;
    }
    v57 = (_QWORD *)v11;
    sub_38D4150(a1, v11, 0);
    v14 = *(unsigned int *)(a1 + 120);
    v15 = 0;
    v16 = (unsigned __int64)v57;
    if ( (_DWORD)v14 )
      v15 = *(_QWORD *)(*(_QWORD *)(a1 + 112) + 32 * v14 - 32);
    v17 = *(__int64 **)(a1 + 272);
    v18 = *v17;
    v19 = *v57 & 7LL;
    v57[1] = v17;
    v18 &= 0xFFFFFFFFFFFFFFF8LL;
    *v57 = v18 | v19;
    *(_QWORD *)(v18 + 8) = v12;
    *v17 = *v17 & 7 | v12;
    v57[3] = v15;
    goto LABEL_17;
  }
  v40 = sub_22077B0(0x58u);
  v41 = v40;
  if ( v40 )
  {
    v42 = v40;
    sub_38CF760(v40, 2, 1, 0);
    *(_QWORD *)(v41 + 56) = 0;
    *(_QWORD *)(v41 + 64) = v41 + 80;
    *(_WORD *)(v41 + 48) = 0;
    *(_QWORD *)(v41 + 72) = 0x400000000LL;
  }
  else
  {
    v42 = 0;
  }
  sub_38D4150(a1, v41, 0);
  v45 = *(unsigned int *)(a1 + 120);
  v46 = 0;
  if ( (_DWORD)v45 )
    v46 = *(_QWORD *)(*(_QWORD *)(a1 + 112) + 32 * v45 - 32);
  v47 = *(__int64 **)(a1 + 272);
  v48 = *v47;
  v49 = *(_QWORD *)v41 & 7LL;
  *(_QWORD *)(v41 + 8) = v47;
  v48 &= 0xFFFFFFFFFFFFFFF8LL;
  *(_QWORD *)v41 = v48 | v49;
  *(_QWORD *)(v48 + 8) = v42;
  *v47 = *v47 & 7 | v42;
  v50 = *(unsigned int *)(v41 + 72);
  v51 = *(unsigned int *)(v41 + 76);
  v52 = (unsigned int)n;
  *(_QWORD *)(v41 + 24) = v46;
  v53 = src;
  v54 = v50;
  if ( v52 > v51 - v50 )
  {
    sub_16CD150(v41 + 64, (const void *)(v41 + 80), v52 + v50, 1, v43, v44);
    v50 = *(unsigned int *)(v41 + 72);
    v54 = *(_DWORD *)(v41 + 72);
  }
  if ( (_DWORD)v52 )
  {
    memcpy((void *)(*(_QWORD *)(v41 + 64) + v50), v53, v52);
    v54 = *(_DWORD *)(v41 + 72);
  }
  *(_BYTE *)(v41 + 17) = 1;
  *(_DWORD *)(v41 + 72) = v54 + v52;
  *(_QWORD *)(v41 + 56) = a3;
LABEL_32:
  v66[0] = &unk_49EFD28;
  sub_16E7960((__int64)v66);
  if ( src != v74 )
    _libc_free((unsigned __int64)src);
  if ( v69 != v71 )
    _libc_free((unsigned __int64)v69);
}
