// Function: sub_31099F0
// Address: 0x31099f0
//
void __fastcall sub_31099F0(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        _QWORD *a5,
        const __m128i *a6,
        char a7,
        const char **a8)
{
  __m128i v11; // xmm4
  __m128i v12; // xmm2
  unsigned __int64 *v13; // rax
  __int64 v14; // rsi
  __int64 v15; // rdi
  unsigned __int64 v16; // rax
  __int64 *v17; // rax
  size_t v18; // rbx
  unsigned __int64 v19; // rax
  unsigned __int64 v20; // rbx
  size_t v21; // rsi
  __int64 v22; // rax
  unsigned __int64 v23; // rdx
  unsigned __int64 v24; // rax
  void *v25; // r14
  bool v26; // r13
  void *v27; // r12
  size_t v28; // rbx
  unsigned __int64 v29; // rax
  void *v30; // r12
  size_t v31; // r13
  int v32; // eax
  unsigned int v33; // r8d
  __int64 *v34; // rcx
  __int64 v35; // rdx
  __m128i v36; // xmm0
  __int64 v37; // rax
  unsigned int v38; // r8d
  __int64 *v39; // rcx
  __int64 v40; // r14
  __int64 *v41; // rax
  __int64 *v42; // rax
  _WORD *v43; // rdi
  __m128i v44; // xmm1
  int v45; // eax
  __int64 v46; // r13
  _QWORD *v47; // rcx
  _QWORD *v48; // r14
  __int64 *v51; // [rsp+60h] [rbp-1B0h]
  char v52; // [rsp+6Bh] [rbp-1A5h]
  unsigned int v53; // [rsp+6Ch] [rbp-1A4h]
  void *src; // [rsp+78h] [rbp-198h]
  __int64 *v55; // [rsp+80h] [rbp-190h]
  char v56; // [rsp+88h] [rbp-188h]
  _QWORD *v57; // [rsp+88h] [rbp-188h]
  __m128i v58; // [rsp+A0h] [rbp-170h] BYREF
  __int64 v59[2]; // [rsp+B0h] [rbp-160h] BYREF
  char v60; // [rsp+C0h] [rbp-150h]
  __int64 v61[2]; // [rsp+D0h] [rbp-140h] BYREF
  _QWORD v62[2]; // [rsp+E0h] [rbp-130h] BYREF
  _WORD *v63[2]; // [rsp+F0h] [rbp-120h] BYREF
  _QWORD v64[2]; // [rsp+100h] [rbp-110h] BYREF
  __m128i v65; // [rsp+110h] [rbp-100h] BYREF
  __int64 v66; // [rsp+120h] [rbp-F0h] BYREF
  unsigned __int64 v67; // [rsp+128h] [rbp-E8h]
  __m128i v68; // [rsp+130h] [rbp-E0h] BYREF
  void *v69; // [rsp+140h] [rbp-D0h] BYREF
  size_t n; // [rsp+148h] [rbp-C8h]
  void *v71; // [rsp+150h] [rbp-C0h] BYREF
  size_t v72; // [rsp+158h] [rbp-B8h]
  __m128i v73; // [rsp+160h] [rbp-B0h] BYREF
  void *v74[2]; // [rsp+170h] [rbp-A0h] BYREF
  __m128i v75; // [rsp+180h] [rbp-90h] BYREF
  __int16 v76; // [rsp+190h] [rbp-80h]
  const char **v77; // [rsp+1A0h] [rbp-70h] BYREF
  __int64 v78; // [rsp+1A8h] [rbp-68h]
  __int16 v79; // [rsp+1C0h] [rbp-50h]
  __m128i v80; // [rsp+1D0h] [rbp-40h]

  LOBYTE(v78) = 1;
  v77 = a8;
  sub_30CBEF0(a1, a2, a3, (__int64)a8, v78);
  v79 = 261;
  *(_QWORD *)a1 = &unk_4A329D8;
  *(_QWORD *)(a1 + 80) = *a5;
  *a5 = 0;
  v11 = _mm_loadu_si128(a6 + 1);
  v12 = _mm_loadu_si128(a6);
  *(_QWORD *)(a1 + 152) = 0x1000000000LL;
  *(_BYTE *)(a1 + 88) = 0;
  *(_BYTE *)(a1 + 128) = a7;
  *(_QWORD *)(a1 + 136) = 0;
  *(_QWORD *)(a1 + 144) = 0;
  *(_QWORD *)(a1 + 160) = 0;
  *(_QWORD *)(a1 + 168) = 0;
  *(_QWORD *)(a1 + 176) = 0x800000000LL;
  *(__m128i *)(a1 + 96) = v12;
  *(__m128i *)(a1 + 112) = v11;
  v77 = (const char **)a6->m128i_i64[0];
  v78 = a6->m128i_i64[1];
  sub_C7EAD0((__int64)v59, &v77, 0, 1u, 0);
  if ( (v60 & 1) != 0 && LODWORD(v59[0]) )
  {
    (*(void (__fastcall **)(void **))(*(_QWORD *)v59[1] + 32LL))(v74);
    v13 = sub_2241130((unsigned __int64 *)v74, 0, 0, "Could not open remarks file: ", 0x1Du);
    v71 = &v73;
    if ( (unsigned __int64 *)*v13 == v13 + 2 )
    {
      v73 = _mm_loadu_si128((const __m128i *)v13 + 1);
    }
    else
    {
      v71 = (void *)*v13;
      v73.m128i_i64[0] = v13[2];
    }
    v72 = v13[1];
    *v13 = (unsigned __int64)(v13 + 2);
    v13[1] = 0;
    *((_BYTE *)v13 + 16) = 0;
    v79 = 260;
    v77 = (const char **)&v71;
    sub_B6ECE0(a4, (__int64)&v77);
    if ( v71 != &v73 )
      j_j___libc_free_0((unsigned __int64)v71);
    if ( v74[0] != &v75 )
      j_j___libc_free_0((unsigned __int64)v74[0]);
    goto LABEL_9;
  }
  sub_C7C840((__int64)&v77, v59[0], 1, 0);
  v61[0] = (__int64)v62;
  sub_3109320(v61, "' inlined into '", (__int64)"");
  v63[0] = v64;
  sub_3109320((__int64 *)v63, "' will not be inlined into '", (__int64)"");
  v52 = v79;
  if ( (_BYTE)v79 )
  {
    v55 = (__int64 *)(a1 + 136);
    while ( 1 )
    {
      v58 = v80;
      v16 = sub_C931B0(v58.m128i_i64, " at callsite ", 0xDu, 0);
      if ( v16 == -1 )
      {
        v36 = _mm_load_si128(&v58);
        v66 = 0;
        v67 = 0;
        v65 = v36;
      }
      else
      {
        v14 = v16 + 13;
        if ( v16 + 13 > v58.m128i_i64[1] )
        {
          v14 = v58.m128i_i64[1];
          v15 = 0;
        }
        else
        {
          v15 = v58.m128i_i64[1] - v14;
        }
        v65.m128i_i64[0] = v58.m128i_i64[0];
        if ( v16 > v58.m128i_i64[1] )
          v16 = v58.m128i_u64[1];
        v67 = v15;
        v66 = v14 + v58.m128i_i64[0];
        v65.m128i_i64[1] = v16;
      }
      if ( sub_C931B0(v65.m128i_i64, v63[0], (size_t)v63[1], 0) == -1 )
      {
        v56 = v52;
        v17 = v61;
      }
      else
      {
        v56 = 0;
        v17 = (__int64 *)v63;
      }
      v18 = v17[1];
      v19 = sub_C931B0(v65.m128i_i64, (_WORD *)*v17, v18, 0);
      if ( v19 == -1 )
      {
        v44 = _mm_load_si128(&v65);
        v69 = 0;
        n = 0;
        v68 = v44;
      }
      else
      {
        v20 = v19 + v18;
        if ( v20 > v65.m128i_i64[1] )
        {
          v20 = v65.m128i_u64[1];
          v21 = 0;
        }
        else
        {
          v21 = v65.m128i_i64[1] - v20;
        }
        v68.m128i_i64[0] = v65.m128i_i64[0];
        if ( v19 > v65.m128i_i64[1] )
          v19 = v65.m128i_u64[1];
        n = v21;
        v69 = (void *)(v65.m128i_i64[0] + v20);
        v68.m128i_i64[1] = v19;
      }
      v22 = sub_C93460(v68.m128i_i64, ": '", 3u);
      if ( v22 == -1 )
      {
        v26 = v52;
        v25 = 0;
        v27 = 0;
      }
      else
      {
        v23 = v68.m128i_u64[1];
        v24 = v22 + 3;
        if ( v24 > v68.m128i_i64[1] )
        {
          v26 = v52;
          v25 = 0;
        }
        else
        {
          v25 = (void *)(v68.m128i_i64[1] - v24);
          v26 = v68.m128i_i64[1] == v24;
          v23 = v24;
        }
        v27 = (void *)(v23 + v68.m128i_i64[0]);
      }
      v28 = sub_C93460((__int64 *)&v69, "'", 1u);
      src = v69;
      if ( v28 == -1 )
      {
        v28 = n;
      }
      else if ( n <= v28 )
      {
        v28 = n;
      }
      v29 = sub_C931B0(&v66, ";", 1u, 0);
      if ( v29 == -1 )
      {
        v29 = v67;
        if ( v26 || v28 == 0 )
          goto LABEL_56;
      }
      else
      {
        if ( v67 <= v29 )
          v29 = v67;
        if ( v26 || v28 == 0 )
        {
LABEL_56:
          v76 = 1283;
          v74[0] = "Invalid remark format: ";
          v75 = v58;
          sub_B6ECE0(a4, (__int64)v74);
          v43 = v63[0];
          if ( (_QWORD *)v63[0] != v64 )
            goto LABEL_57;
          goto LABEL_58;
        }
      }
      if ( !v29 )
        goto LABEL_56;
      v75.m128i_i64[0] = v66;
      v76 = 1285;
      v74[0] = v27;
      v75.m128i_i64[1] = v29;
      v74[1] = v25;
      sub_CA0F50((__int64 *)&v71, v74);
      v30 = v71;
      v31 = v72;
      v32 = sub_C92610();
      v33 = sub_C92740((__int64)v55, v30, v31, v32);
      v34 = (__int64 *)(*(_QWORD *)(a1 + 136) + 8LL * v33);
      v35 = *v34;
      if ( !*v34 )
        goto LABEL_47;
      if ( v35 == -8 )
        break;
LABEL_40:
      *(_BYTE *)(v35 + 8) = v56;
      if ( !a6[1].m128i_i32[0] )
      {
        v45 = sub_C92610();
        v46 = (unsigned int)sub_C92740(a1 + 160, src, v28, v45);
        v47 = (_QWORD *)(*(_QWORD *)(a1 + 160) + 8 * v46);
        if ( !*v47 )
          goto LABEL_69;
        if ( *v47 == -8 )
        {
          --*(_DWORD *)(a1 + 176);
LABEL_69:
          v57 = v47;
          v48 = (_QWORD *)sub_C7D670(v28 + 9, 8);
          memcpy(v48 + 1, src, v28);
          *((_BYTE *)v48 + v28 + 8) = 0;
          *v48 = v28;
          *v57 = v48;
          ++*(_DWORD *)(a1 + 172);
          sub_C929D0((__int64 *)(a1 + 160), v46);
        }
      }
      if ( v71 != &v73 )
        j_j___libc_free_0((unsigned __int64)v71);
      sub_C7C5C0((__int64)&v77);
      if ( !(_BYTE)v79 )
        goto LABEL_74;
    }
    --*(_DWORD *)(a1 + 152);
LABEL_47:
    v51 = v34;
    v53 = v33;
    v37 = sub_C7D670(v31 + 17, 8);
    v38 = v53;
    v39 = v51;
    v40 = v37;
    if ( v31 )
    {
      memcpy((void *)(v37 + 16), v30, v31);
      v38 = v53;
      v39 = v51;
    }
    *(_BYTE *)(v40 + v31 + 16) = 0;
    *(_QWORD *)v40 = v31;
    *(_BYTE *)(v40 + 8) = 0;
    *v39 = v40;
    ++*(_DWORD *)(a1 + 148);
    v41 = (__int64 *)(*(_QWORD *)(a1 + 136) + 8LL * (unsigned int)sub_C929D0(v55, v38));
    v35 = *v41;
    if ( *v41 == -8 || !v35 )
    {
      v42 = v41 + 1;
      do
      {
        do
          v35 = *v42++;
        while ( !v35 );
      }
      while ( v35 == -8 );
    }
    goto LABEL_40;
  }
LABEL_74:
  *(_BYTE *)(a1 + 88) = 1;
  v43 = v63[0];
  if ( (_QWORD *)v63[0] != v64 )
LABEL_57:
    j_j___libc_free_0((unsigned __int64)v43);
LABEL_58:
  if ( (_QWORD *)v61[0] != v62 )
  {
    j_j___libc_free_0(v61[0]);
    if ( (v60 & 1) != 0 )
      return;
    goto LABEL_60;
  }
LABEL_9:
  if ( (v60 & 1) != 0 )
    return;
LABEL_60:
  if ( v59[0] )
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v59[0] + 8LL))(v59[0]);
}
