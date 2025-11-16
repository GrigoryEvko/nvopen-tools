// Function: sub_123DE00
// Address: 0x123de00
//
__int64 __fastcall sub_123DE00(
        __int64 a1,
        _QWORD *a2,
        unsigned __int64 a3,
        int a4,
        unsigned int a5,
        __int64 *a6,
        unsigned __int64 a7)
{
  __int64 v10; // rdi
  unsigned __int64 v11; // rdx
  _BYTE *v13; // rsi
  __int64 v14; // r14
  _QWORD *v15; // rax
  _QWORD *v16; // r14
  unsigned __int64 v17; // rax
  size_t v18; // rdx
  _BYTE *v19; // rsi
  __int64 v20; // r14
  __int64 v21; // rdx
  _QWORD *v22; // rax
  _QWORD *v23; // r14
  __int64 v24; // rax
  _QWORD *v25; // rax
  _QWORD *v26; // r14
  __int64 v27; // rax
  __int64 v28; // rdi
  __int64 v29; // rcx
  __int64 v30; // rdx
  unsigned __int64 **v31; // r10
  unsigned __int64 **i; // rdx
  unsigned __int64 *v33; // rax
  unsigned __int64 v34; // rcx
  unsigned __int64 v35; // rsi
  __int64 v36; // rax
  __int64 v37; // rdi
  __int64 v38; // r14
  __int64 v39; // rax
  __int64 v40; // rdi
  __int64 v41; // rcx
  __int64 v42; // rdx
  __int64 *v43; // rsi
  __int64 *v44; // rax
  unsigned __int64 j; // r8
  __int64 v46; // rdx
  __int64 v47; // rcx
  __int64 v48; // rax
  __int64 v49; // rdi
  __int64 v50; // r14
  __int64 v51; // rax
  __int64 v52; // rdi
  char *v53; // rsi
  __int64 v54; // rdx
  unsigned __int64 v55; // rax
  unsigned __int64 v57; // rcx
  char *v58; // rax
  __int64 v59; // rcx
  __m128i *v60; // rax
  __int64 v61; // [rsp+0h] [rbp-E0h]
  _QWORD *v62; // [rsp+8h] [rbp-D8h]
  char *v63; // [rsp+8h] [rbp-D8h]
  _QWORD *v64; // [rsp+10h] [rbp-D0h]
  _QWORD *v65; // [rsp+10h] [rbp-D0h]
  __int64 v66; // [rsp+18h] [rbp-C8h]
  _QWORD *v67; // [rsp+18h] [rbp-C8h]
  _QWORD *v68; // [rsp+18h] [rbp-C8h]
  __int64 v69; // [rsp+20h] [rbp-C0h]
  _QWORD *v70; // [rsp+28h] [rbp-B8h]
  __int64 v71; // [rsp+28h] [rbp-B8h]
  _QWORD *v72; // [rsp+28h] [rbp-B8h]
  _QWORD *v73; // [rsp+28h] [rbp-B8h]
  unsigned __int64 v74; // [rsp+38h] [rbp-A8h] BYREF
  _QWORD v75[2]; // [rsp+40h] [rbp-A0h] BYREF
  __int64 v76; // [rsp+50h] [rbp-90h] BYREF
  unsigned __int64 v77[2]; // [rsp+60h] [rbp-80h] BYREF
  __m128i v78; // [rsp+70h] [rbp-70h] BYREF
  __m128i v79; // [rsp+80h] [rbp-60h] BYREF
  _QWORD *v80; // [rsp+90h] [rbp-50h] BYREF
  _QWORD *v81; // [rsp+98h] [rbp-48h]
  __int64 v82; // [rsp+A0h] [rbp-40h]

  v74 = 0;
  if ( a3 )
  {
    v24 = *(_QWORD *)(a1 + 352);
    v77[0] = a3;
    v69 = v24;
    if ( *(_BYTE *)(v24 + 343) )
    {
      v79.m128i_i64[0] = 0;
    }
    else
    {
      v79.m128i_i64[1] = 0;
      v79.m128i_i64[0] = (__int64)byte_3F871B3;
    }
    v80 = 0;
    v81 = 0;
    v82 = 0;
    v25 = sub_9CA390((_QWORD *)v24, v77, &v79);
    v26 = v80;
    v68 = v25 + 4;
    v73 = v81;
    if ( v81 != v80 )
    {
      do
      {
        if ( *v26 )
          (*(void (__fastcall **)(_QWORD))(*(_QWORD *)*v26 + 8LL))(*v26);
        ++v26;
      }
      while ( v73 != v26 );
      v26 = v80;
    }
    if ( v26 )
      j_j___libc_free_0(v26, v82 - (_QWORD)v26);
    v17 = (unsigned __int64)v68;
  }
  else
  {
    v10 = *(_QWORD *)(a1 + 344);
    v11 = a2[1];
    v13 = (_BYTE *)*a2;
    if ( v10 )
    {
      v66 = sub_BA8B30(v10, (__int64)v13, v11);
      if ( !v66 )
      {
        sub_8FD6D0((__int64)v75, "Reference to undefined global \"", a2);
        if ( v75[1] == 0x3FFFFFFFFFFFFFFFLL )
          sub_4262D8((__int64)"basic_string::append");
        v60 = (__m128i *)sub_2241490(v75, "\"", 1, v59);
        v77[0] = (unsigned __int64)&v78;
        if ( (__m128i *)v60->m128i_i64[0] == &v60[1] )
        {
          v78 = _mm_loadu_si128(v60 + 1);
        }
        else
        {
          v77[0] = v60->m128i_i64[0];
          v78.m128i_i64[0] = v60[1].m128i_i64[0];
        }
        v77[1] = v60->m128i_u64[1];
        v60->m128i_i64[0] = (__int64)v60[1].m128i_i64;
        v60->m128i_i64[1] = 0;
        v60[1].m128i_i8[0] = 0;
        LOWORD(v82) = 260;
        v79.m128i_i64[0] = (__int64)v77;
        sub_11FD800(a1 + 176, a7, (__int64)&v79, 1);
        if ( (__m128i *)v77[0] != &v78 )
          j_j___libc_free_0(v77[0], v78.m128i_i64[0] + 1);
        if ( (__int64 *)v75[0] != &v76 )
          j_j___libc_free_0(v75[0], v76 + 1);
        return 1;
      }
      v69 = *(_QWORD *)(a1 + 352);
      sub_B2F930(&v79, v66);
      v14 = sub_B2F650(v79.m128i_i64[0], v79.m128i_i64[1]);
      if ( (_QWORD **)v79.m128i_i64[0] != &v80 )
        j_j___libc_free_0(v79.m128i_i64[0], (char *)v80 + 1);
      v77[0] = v14;
      if ( *(_BYTE *)(v69 + 343) )
      {
        v79.m128i_i64[0] = 0;
      }
      else
      {
        v79.m128i_i64[1] = 0;
        v79.m128i_i64[0] = (__int64)byte_3F871B3;
      }
      v80 = 0;
      v81 = 0;
      v82 = 0;
      v15 = sub_9CA390((_QWORD *)v69, v77, &v79);
      v16 = v80;
      v64 = v15;
      v62 = v15 + 4;
      v70 = v81;
      if ( v81 != v80 )
      {
        do
        {
          if ( *v16 )
            (*(void (__fastcall **)(_QWORD))(*(_QWORD *)*v16 + 8LL))(*v16);
          ++v16;
        }
        while ( v70 != v16 );
        v16 = v80;
      }
      if ( v16 )
        j_j___libc_free_0(v16, v82 - (_QWORD)v16);
      v64[5] = v66;
      v17 = (unsigned __int64)v62;
    }
    else
    {
      sub_B2F7A0(&v79, v13, v11, a4, *(_QWORD *)(a1 + 1752), *(_QWORD *)(a1 + 1760));
      v71 = sub_B2F650(v79.m128i_i64[0], v79.m128i_i64[1]);
      if ( (_QWORD **)v79.m128i_i64[0] != &v80 )
        j_j___libc_free_0(v79.m128i_i64[0], (char *)v80 + 1);
      v18 = a2[1];
      v19 = (_BYTE *)*a2;
      v20 = *(_QWORD *)(a1 + 352);
      v69 = v20;
      v63 = sub_C948A0((char ***)(v20 + 512), v19, v18);
      v61 = v21;
      v77[0] = v71;
      if ( *(_BYTE *)(v20 + 343) )
      {
        v79.m128i_i64[0] = 0;
      }
      else
      {
        v79.m128i_i64[1] = 0;
        v79.m128i_i64[0] = (__int64)byte_3F871B3;
      }
      v80 = 0;
      v81 = 0;
      v82 = 0;
      v22 = sub_9CA390((_QWORD *)v20, v77, &v79);
      v23 = v80;
      v67 = v22;
      v65 = v22 + 4;
      v72 = v81;
      if ( v81 != v80 )
      {
        do
        {
          if ( *v23 )
            (*(void (__fastcall **)(_QWORD))(*(_QWORD *)*v23 + 8LL))(*v23);
          ++v23;
        }
        while ( v72 != v23 );
        v23 = v80;
      }
      if ( v23 )
        j_j___libc_free_0(v23, v82 - (_QWORD)v23);
      v67[5] = v63;
      v67[6] = v61;
      v17 = (unsigned __int64)v65;
    }
  }
  v74 = *(unsigned __int8 *)(v69 + 343) | v17 & 0xFFFFFFFFFFFFFFF8LL;
  v27 = *(_QWORD *)(a1 + 1544);
  if ( v27 )
  {
    v28 = a1 + 1536;
    do
    {
      while ( 1 )
      {
        v29 = *(_QWORD *)(v27 + 16);
        v30 = *(_QWORD *)(v27 + 24);
        if ( *(_DWORD *)(v27 + 32) >= a5 )
          break;
        v27 = *(_QWORD *)(v27 + 24);
        if ( !v30 )
          goto LABEL_43;
      }
      v28 = v27;
      v27 = *(_QWORD *)(v27 + 16);
    }
    while ( v29 );
LABEL_43:
    if ( a1 + 1536 != v28 && *(_DWORD *)(v28 + 32) <= a5 )
    {
      v31 = *(unsigned __int64 ***)(v28 + 48);
      for ( i = *(unsigned __int64 ***)(v28 + 40); v31 != i; i += 2 )
      {
        v33 = *i;
        v34 = v74;
        v35 = **i;
        **i = v74;
        if ( (v35 & 2) != 0 )
          *v33 = v34 | 2;
        if ( (v35 & 4) != 0 )
          *v33 |= 4u;
      }
      v36 = sub_220F330(v28, a1 + 1536);
      v37 = *(_QWORD *)(v36 + 40);
      v38 = v36;
      if ( v37 )
        j_j___libc_free_0(v37, *(_QWORD *)(v36 + 56) - v37);
      j_j___libc_free_0(v38, 64);
      --*(_QWORD *)(a1 + 1568);
    }
  }
  v39 = *(_QWORD *)(a1 + 1592);
  if ( v39 )
  {
    v40 = a1 + 1584;
    do
    {
      while ( 1 )
      {
        v41 = *(_QWORD *)(v39 + 16);
        v42 = *(_QWORD *)(v39 + 24);
        if ( *(_DWORD *)(v39 + 32) >= a5 )
          break;
        v39 = *(_QWORD *)(v39 + 24);
        if ( !v42 )
          goto LABEL_59;
      }
      v40 = v39;
      v39 = *(_QWORD *)(v39 + 16);
    }
    while ( v41 );
LABEL_59:
    if ( a1 + 1584 != v40 && *(_DWORD *)(v40 + 32) <= a5 )
    {
      v43 = *(__int64 **)(v40 + 48);
      v44 = *(__int64 **)(v40 + 40);
      for ( j = v74; v43 != v44; *(_QWORD *)(v46 + 64) = v47 )
      {
        v46 = *v44;
        v47 = *a6;
        v44 += 2;
        *(_QWORD *)(v46 + 56) = j;
      }
      v48 = sub_220F330(v40, a1 + 1584);
      v49 = *(_QWORD *)(v48 + 40);
      v50 = v48;
      if ( v49 )
        j_j___libc_free_0(v49, *(_QWORD *)(v48 + 56) - v49);
      j_j___libc_free_0(v50, 64);
      --*(_QWORD *)(a1 + 1616);
    }
  }
  v51 = *a6;
  if ( *a6 )
  {
    v52 = *(_QWORD *)(a1 + 352);
    *a6 = 0;
    v79.m128i_i64[0] = v51;
    sub_9D8550(v52, v74, &v79);
    if ( v79.m128i_i64[0] )
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v79.m128i_i64[0] + 8LL))(v79.m128i_i64[0]);
  }
  v53 = *(char **)(a1 + 1632);
  v54 = *(_QWORD *)(a1 + 1624);
  v55 = (__int64)&v53[-v54] >> 3;
  if ( a5 == v55 )
  {
    if ( v53 == *(char **)(a1 + 1640) )
    {
      sub_BAF2D0((char **)(a1 + 1624), v53, &v74);
      return 0;
    }
    else
    {
      if ( v53 )
      {
        *(_QWORD *)v53 = v74;
        v53 = *(char **)(a1 + 1632);
      }
      *(_QWORD *)(a1 + 1632) = v53 + 8;
      return 0;
    }
  }
  else
  {
    if ( a5 > v55 )
    {
      v57 = a5 + 1;
      if ( v57 > v55 )
      {
        sub_1213760((char **)(a1 + 1624), v57 - v55);
        v54 = *(_QWORD *)(a1 + 1624);
      }
      else if ( v57 < v55 )
      {
        v58 = (char *)(v54 + 8 * v57);
        if ( v53 != v58 )
          *(_QWORD *)(a1 + 1632) = v58;
      }
    }
    *(_QWORD *)(v54 + 8LL * a5) = v74;
    return 0;
  }
}
