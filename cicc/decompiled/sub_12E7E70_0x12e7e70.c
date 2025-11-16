// Function: sub_12E7E70
// Address: 0x12e7e70
//
_QWORD *__fastcall sub_12E7E70(
        __int64 a1,
        _QWORD *a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        _QWORD *a6,
        __int64 a7,
        _QWORD *a8)
{
  __int64 v10; // r13
  struct __jmp_buf_tag *v11; // r12
  int v12; // eax
  const __m128i *v13; // rdx
  char v14; // bl
  char *v15; // rax
  __int64 v16; // rdx
  __int64 v17; // rcx
  __int64 v18; // r8
  __int64 v19; // r9
  char v20; // al
  __int64 v21; // rdx
  __int64 v22; // rcx
  __int64 v23; // r8
  __int64 v24; // r9
  __m128i *v25; // r12
  __int64 v26; // rax
  unsigned __int64 v27; // rbx
  char *v28; // rdi
  __m128i *v29; // rsi
  __m128i *v30; // rax
  const __m128i *v31; // rcx
  const __m128i *v32; // rdx
  __m128i *v33; // rcx
  __int64 v34; // rsi
  _DWORD *v35; // rax
  __int64 v36; // rdi
  __int64 v37; // rdx
  __int64 v38; // rcx
  __int64 v39; // r8
  __int64 v40; // r9
  char v41; // bl
  __m128i *v42; // rax
  __int64 v43; // rcx
  char *v44; // rsi
  __int64 *v45; // rdx
  unsigned __int64 v46; // rdx
  __int64 *v47; // r13
  __int64 v48; // r12
  __int64 v49; // rdx
  __int64 v50; // rcx
  __int64 v51; // r8
  __int64 v52; // r9
  _DWORD *v53; // rax
  _DWORD *v54; // rax
  __int64 v55; // rsi
  __int64 v56; // rdx
  __int64 v57; // rcx
  __int64 v58; // r8
  __int64 v59; // r9
  __int64 v60; // rdx
  __int64 v61; // rcx
  __int64 v62; // r8
  __int64 v63; // r9
  __int64 *v64; // rdi
  void *v65; // rdi
  __int64 v66; // r14
  __int64 v67; // rax
  _QWORD *v68; // rbx
  const __m128i *v69; // rcx
  const __m128i *v70; // r13
  unsigned __int64 v71; // r15
  __m128i *v72; // rax
  __int8 *v73; // r15
  __int64 v74; // r12
  __int64 v75; // rdi
  int v76; // r12d
  __int64 v77; // rax
  __m128i *v78; // rax
  __m128i *v79; // rcx
  __m128i *v80; // rcx
  const __m128i *v81; // [rsp+8h] [rbp-1238h]
  __int64 v88; // [rsp+48h] [rbp-11F8h] BYREF
  __int64 *v89; // [rsp+50h] [rbp-11F0h] BYREF
  __int64 v90; // [rsp+58h] [rbp-11E8h]
  _QWORD v91[2]; // [rsp+60h] [rbp-11E0h] BYREF
  _DWORD v92[1120]; // [rsp+70h] [rbp-11D0h] BYREF
  __int64 v93; // [rsp+11F0h] [rbp-50h] BYREF
  __m128i *v94; // [rsp+11F8h] [rbp-48h]
  __m128i *v95; // [rsp+1200h] [rbp-40h]
  __int8 *v96; // [rsp+1208h] [rbp-38h]

  if ( (unsigned __int8)sub_1BF83F0() )
  {
    sub_1C3E9C0(a3);
    return 0;
  }
  v10 = sub_1C3E710();
  v11 = (struct __jmp_buf_tag *)sub_16D40F0(v10);
  if ( !v11 )
  {
    v65 = (void *)sub_1C42D70(200, 8);
    memset(v65, 0, 0xC8u);
    sub_16D40E0(v10, v65);
    v11 = (struct __jmp_buf_tag *)sub_16D40F0(v10);
  }
  v12 = _setjmp(v11);
  if ( v12 )
  {
    if ( v12 == 1 )
    {
      sub_1C3E9C0(a3);
      return 0;
    }
    goto LABEL_21;
  }
  v25 = *(__m128i **)(a1 + 296);
  if ( !v25 )
  {
    v66 = *(_QWORD *)(a1 + 288);
    v67 = sub_22077B0(4512);
    v68 = (_QWORD *)v67;
    if ( v67 )
    {
      sub_12D6300(v67, v66);
      v68[560] = 0;
      v68[561] = 0;
      v68[562] = 0;
      v68[563] = 0;
    }
    v13 = (const __m128i *)(v66 + 488);
    if ( (_QWORD *)(v66 + 488) == v68 + 561 )
      goto LABEL_70;
    v69 = *(const __m128i **)(v66 + 496);
    v70 = *(const __m128i **)(v66 + 488);
    v28 = (char *)v68[561];
    v71 = (char *)v69 - (char *)v70;
    v29 = (__m128i *)(v68[563] - (_QWORD)v28);
    if ( (char *)v69 - (char *)v70 > (unsigned __int64)v29 )
    {
      if ( v71 )
      {
        if ( v71 > 0x7FFFFFFFFFFFFFF0LL )
          goto LABEL_96;
        v81 = *(const __m128i **)(v66 + 496);
        v77 = sub_22077B0((char *)v81 - (char *)v70);
        v28 = (char *)v68[561];
        v69 = v81;
        v25 = (__m128i *)v77;
        v29 = (__m128i *)(v68[563] - (_QWORD)v28);
      }
      if ( v69 != v70 )
      {
        v13 = v70;
        v78 = v25;
        v79 = (__m128i *)((char *)v25 + (char *)v69 - (char *)v70);
        do
        {
          if ( v78 )
            *v78 = _mm_loadu_si128(v13);
          ++v78;
          ++v13;
        }
        while ( v78 != v79 );
      }
      if ( v28 )
        j_j___libc_free_0(v28, v29);
      v73 = &v25->m128i_i8[v71];
      v68[561] = v25;
      v68[563] = v73;
      goto LABEL_69;
    }
    v72 = (__m128i *)v68[562];
    v13 = (const __m128i *)((char *)v72 - v28);
    if ( v71 > (char *)v72 - v28 )
    {
      if ( v13 )
      {
        memmove(v28, *(const void **)(v66 + 488), (size_t)v13);
        v72 = (__m128i *)v68[562];
        v28 = (char *)v68[561];
        v69 = *(const __m128i **)(v66 + 496);
        v70 = *(const __m128i **)(v66 + 488);
        v13 = (const __m128i *)((char *)v72 - v28);
      }
      v13 = (const __m128i *)((char *)v13 + (_QWORD)v70);
      if ( v13 != v69 )
      {
        v80 = (__m128i *)((char *)v72 + (char *)v69 - (char *)v13);
        do
        {
          if ( v72 )
            *v72 = _mm_loadu_si128(v13);
          ++v72;
          ++v13;
        }
        while ( v72 != v80 );
        v73 = (__int8 *)(v68[561] + v71);
        goto LABEL_69;
      }
    }
    else if ( v69 != v70 )
    {
      memmove(v28, *(const void **)(v66 + 488), *(_QWORD *)(v66 + 496) - (_QWORD)v70);
      v28 = (char *)v68[561];
    }
    v73 = &v28[v71];
LABEL_69:
    v68[562] = v73;
LABEL_70:
    v74 = *(_QWORD *)(a1 + 296);
    *(_QWORD *)(a1 + 296) = v68;
    if ( v74 )
    {
      v75 = *(_QWORD *)(v74 + 4488);
      if ( v75 )
        j_j___libc_free_0(v75, *(_QWORD *)(v74 + 4504) - v75);
      j_j___libc_free_0(v74, 4512);
      v68 = *(_QWORD **)(a1 + 296);
    }
    v68[560] = *(_QWORD *)(a1 + 304);
    v25 = *(__m128i **)(a1 + 296);
  }
  v26 = v25[280].m128i_i64[0];
  v27 = v25[281].m128i_i64[0] - v25[280].m128i_i64[1];
  qmemcpy(v92, v25, sizeof(v92));
  v29 = v25 + 280;
  v28 = (char *)&v93;
  v93 = v26;
  v94 = 0;
  v95 = 0;
  v96 = 0;
  if ( v27 )
  {
    if ( v27 <= 0x7FFFFFFFFFFFFFF0LL )
    {
      v30 = (__m128i *)sub_22077B0(v27);
      goto LABEL_27;
    }
LABEL_96:
    sub_4261EA(v28, v29, v13);
  }
  v27 = 0;
  v30 = 0;
LABEL_27:
  v95 = v30;
  v96 = &v30->m128i_i8[v27];
  v31 = (const __m128i *)v25[281].m128i_i64[0];
  v94 = v30;
  v32 = (const __m128i *)v25[280].m128i_i64[1];
  if ( v31 == v32 )
  {
    v33 = v30;
  }
  else
  {
    v33 = (__m128i *)((char *)v30 + (char *)v31 - (char *)v32);
    do
    {
      if ( v30 )
        *v30 = _mm_loadu_si128(v32);
      ++v30;
      ++v32;
    }
    while ( v30 != v33 );
  }
  v76 = v92[1026];
  v95 = v33;
  v34 = *(unsigned __int8 *)(a1 + 248);
  if ( v92[1026] < 0 )
    v76 = v92[1036];
  sub_16033C0(*a2, v34);
  v14 = v92[882];
  if ( LOBYTE(v92[882]) )
  {
    v14 = 0;
  }
  else if ( !LOBYTE(v92[892]) )
  {
    v14 = LOBYTE(v92[902]) ^ 1;
  }
  v15 = (char *)sub_16D40F0(qword_4FBB510);
  if ( v15 )
    v20 = *v15;
  else
    v20 = qword_4FBB510[2];
  if ( v20 || v14 )
  {
    if ( sub_16DA870(qword_4FBB510, v34, v16, v17, v18, v19) )
      goto LABEL_38;
  }
  else
  {
    if ( !v76 )
      v76 = sub_22420F0();
    if ( sub_16DA870(qword_4FBB510, v34, v16, v17, v18, v19) )
    {
      if ( v76 > 1 )
      {
LABEL_18:
        a2 = sub_12E7B90(a2, (__int64 *)(a1 + 8), v76, (__int64)v92, a4, a5, a6, a7, a8);
        goto LABEL_19;
      }
LABEL_38:
      if ( sub_16DA870(qword_4FBB510, v34, v21, v22, v23, v24) )
        sub_16DB3F0("Phase I", 7, byte_3F871B3, 0);
      v35 = (_DWORD *)sub_1C42D70(4, 4);
      *v35 = 1;
      sub_16D40E0(qword_4FBB3B0, v35);
      sub_12E54A0(a2, *(_QWORD *)(a1 + 24), *(_QWORD *)(a1 + 32), (__int64)v92, a8);
      if ( *a8 )
      {
        v36 = a8[1];
        if ( ((unsigned int (__fastcall *)(__int64, _QWORD))*a8)(v36, 0) )
        {
          if ( sub_16DA870(v36, 0, v37, v38, v39, v40) )
            sub_16DB5E0();
          if ( v94 )
            j_j___libc_free_0(v94, v96 - (__int8 *)v94);
          return a2;
        }
      }
      v88 = 27;
      v41 = sub_12D4250(a2, (__int64)v92);
      v89 = v91;
      v42 = (__m128i *)sub_22409D0(&v89, &v88, 0);
      v89 = (__int64 *)v42;
      v44 = "Yes";
      v91[0] = v88;
      *v42 = _mm_load_si128((const __m128i *)&xmmword_4281B30);
      v45 = v89;
      qmemcpy(&v42[1], "Concurrent=", 11);
      v90 = v88;
      *((_BYTE *)v45 + v88) = 0;
      v46 = 3LL - (v41 == 0);
      if ( !v41 )
        v44 = "No";
      if ( v46 > 0x3FFFFFFFFFFFFFFFLL - v90 )
        sub_4262D8((__int64)"basic_string::append");
      sub_2241490(&v89, v44, v46, v43);
      v47 = v89;
      v48 = v90;
      if ( sub_16DA870(&v89, v44, v49, v50, v51, v52) )
        sub_16DB3F0("Phase II", 8, v47, v48);
      v53 = (_DWORD *)sub_1C42D70(4, 4);
      *v53 = 2;
      sub_16D40E0(qword_4FBB3B0, v53);
      sub_12E54A0(a2, *(_QWORD *)(a1 + 24), *(_QWORD *)(a1 + 32), (__int64)v92, a8);
      v54 = (_DWORD *)sub_1C42D70(4, 4);
      *v54 = 3;
      v55 = (__int64)v54;
      sub_16D40E0(qword_4FBB3B0, v54);
      if ( sub_16DA870(qword_4FBB3B0, v55, v56, v57, v58, v59) )
        sub_16DB5E0();
      v64 = v89;
      if ( v89 != v91 )
      {
        v55 = v91[0] + 1LL;
        j_j___libc_free_0(v89, v91[0] + 1LL);
      }
      if ( sub_16DA870(v64, v55, v60, v61, v62, v63) )
        sub_16DB5E0();
      goto LABEL_19;
    }
    if ( v76 > 1 )
      goto LABEL_18;
  }
  sub_12E54A0(a2, *(_QWORD *)(a1 + 24), *(_QWORD *)(a1 + 32), (__int64)v92, a8);
LABEL_19:
  if ( v94 )
    j_j___libc_free_0(v94, v96 - (__int8 *)v94);
LABEL_21:
  sub_1C3E9C0(a3);
  return a2;
}
