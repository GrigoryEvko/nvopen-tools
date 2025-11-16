// Function: sub_EB7A90
// Address: 0xeb7a90
//
__int64 __fastcall sub_EB7A90(__int64 a1, unsigned __int64 a2)
{
  unsigned int v3; // r13d
  _DWORD *v4; // rax
  char v5; // r14
  _DWORD *v6; // rax
  _DWORD *v8; // rax
  __int64 v9; // rax
  bool v10; // cc
  _QWORD *v11; // rax
  __int64 v12; // rbx
  __int64 v13; // rdx
  char v14; // bl
  unsigned __int64 v15; // rdi
  __m128i *v16; // rax
  unsigned __int64 v17; // rsi
  int v18; // ecx
  unsigned __int64 v19; // r10
  char v20; // dl
  __int64 v21; // rsi
  __m128i v22; // xmm3
  void (__fastcall *v23)(__int64 *, __int64, _QWORD, _QWORD *, __int64, _QWORD, _QWORD *, __int64, __int64, __int64, char, _QWORD, _QWORD, __int64); // rax
  __int64 v24; // rax
  __int64 v25; // rsi
  __int64 v26; // rax
  __int64 v27; // rdi
  __m128i v28; // xmm2
  void (__fastcall *v29)(__int64, _QWORD *, __int64, _QWORD *, __int64, _QWORD, __int64, __int64, char, _QWORD, _QWORD, __int64); // rax
  __int64 v30; // rax
  __int64 v31; // rcx
  __int64 v32; // rax
  __int64 v33; // rdx
  _QWORD *v34; // rdx
  size_t v35; // r10
  __int64 v36; // rdi
  __int64 v37; // rsi
  unsigned __int64 v38; // rax
  void *v39; // rcx
  void *v40; // rax
  __int64 v41; // rax
  __int64 v42; // r12
  __int64 v43; // rax
  _QWORD *v44; // r9
  _QWORD *v45; // rbx
  __int64 v46; // rdx
  _QWORD *v47; // rax
  _QWORD *v48; // r9
  _QWORD *v49; // rbx
  _BYTE *v50; // rax
  __int64 v51; // rax
  _QWORD *v52; // [rsp+38h] [rbp-1B8h]
  _QWORD *v53; // [rsp+38h] [rbp-1B8h]
  __int64 v54; // [rsp+40h] [rbp-1B0h]
  _QWORD *v55; // [rsp+48h] [rbp-1A8h]
  _QWORD *v57; // [rsp+60h] [rbp-190h]
  __int64 v58; // [rsp+68h] [rbp-188h]
  char v59; // [rsp+87h] [rbp-169h]
  __int64 v60; // [rsp+88h] [rbp-168h]
  const char *v61; // [rsp+98h] [rbp-158h] BYREF
  const char *v62; // [rsp+A0h] [rbp-150h] BYREF
  __int64 v63; // [rsp+A8h] [rbp-148h] BYREF
  __int64 v64; // [rsp+B0h] [rbp-140h] BYREF
  char v65; // [rsp+B8h] [rbp-138h]
  __m128i v66; // [rsp+C0h] [rbp-130h] BYREF
  char v67; // [rsp+D0h] [rbp-120h]
  __int128 v68; // [rsp+E0h] [rbp-110h]
  __int64 v69; // [rsp+F0h] [rbp-100h]
  _QWORD *v70; // [rsp+100h] [rbp-F0h] BYREF
  __int64 v71; // [rsp+108h] [rbp-E8h]
  _QWORD v72[2]; // [rsp+110h] [rbp-E0h] BYREF
  _QWORD *v73; // [rsp+120h] [rbp-D0h] BYREF
  __int64 v74; // [rsp+128h] [rbp-C8h]
  _QWORD v75[2]; // [rsp+130h] [rbp-C0h] BYREF
  void *src; // [rsp+140h] [rbp-B0h] BYREF
  size_t n; // [rsp+148h] [rbp-A8h]
  _QWORD v78[2]; // [rsp+150h] [rbp-A0h] BYREF
  _QWORD v79[2]; // [rsp+160h] [rbp-90h] BYREF
  __int64 v80; // [rsp+170h] [rbp-80h] BYREF
  char v81; // [rsp+180h] [rbp-70h]
  char v82; // [rsp+181h] [rbp-6Fh]
  __m128i v83[2]; // [rsp+190h] [rbp-60h] BYREF
  __int16 v84; // [rsp+1B0h] [rbp-40h]

  v60 = -1;
  if ( **(_DWORD **)(a1 + 48) == 4 )
  {
    v9 = sub_ECD7B0(a1);
    v10 = *(_DWORD *)(v9 + 32) <= 0x40u;
    v11 = *(_QWORD **)(v9 + 24);
    if ( !v10 )
      v11 = (_QWORD *)*v11;
    v12 = (__int64)v11;
    v60 = (__int64)v11;
    sub_EABFE0(a1);
    if ( v12 < 0 )
    {
      v83[0].m128i_i64[0] = (__int64)"negative file number";
      v84 = 259;
      return (unsigned int)sub_ECE0E0(a1, v83, 0, 0);
    }
  }
  v71 = 0;
  v70 = v72;
  LOBYTE(v72[0]) = 0;
  v3 = sub_EAE3B0((_QWORD *)a1, &v70);
  if ( !(_BYTE)v3 )
  {
    v74 = 0;
    v73 = v75;
    v4 = *(_DWORD **)(a1 + 48);
    LOBYTE(v75[0]) = 0;
    v55 = 0;
    v54 = 0;
    if ( *v4 == 3 )
    {
      v83[0].m128i_i64[0] = (__int64)"explicit path specified, but no file number";
      v84 = 259;
      if ( (unsigned __int8)sub_ECE0A0(a1, v60 == -1, v83) || (unsigned __int8)sub_EAE3B0((_QWORD *)a1, &v73) )
      {
        v3 = 1;
LABEL_14:
        if ( v73 != v75 )
          j_j___libc_free_0(v73, v75[0] + 1LL);
        goto LABEL_16;
      }
      v57 = v73;
      v58 = v74;
      v55 = v70;
      v54 = v71;
    }
    else
    {
      v57 = v70;
      v58 = v71;
    }
    v5 = 0;
    v69 = 0;
    src = v78;
    n = 0;
    LOBYTE(v78[0]) = 0;
    v59 = 0;
    v68 = 0;
    while ( !(unsigned __int8)sub_ECE2A0(a1, 9) )
    {
      v66 = 0u;
      v83[0].m128i_i64[0] = (__int64)"unexpected token in '.file' directive";
      v84 = 259;
      v6 = (_DWORD *)sub_ECD7B0(a1);
      if ( (unsigned __int8)sub_ECE0A0(a1, *v6 != 2, v83) || (unsigned __int8)sub_EB61F0(a1, v66.m128i_i64) )
        goto LABEL_30;
      if ( v66.m128i_i64[1] == 3 )
      {
        if ( *(_WORD *)v66.m128i_i64[0] != 25709 || *(_BYTE *)(v66.m128i_i64[0] + 2) != 53 )
          goto LABEL_11;
        v83[0].m128i_i64[0] = (__int64)"MD5 checksum specified, but no file number";
        v84 = 259;
        if ( (unsigned __int8)sub_ECE0A0(a1, v60 == -1, v83) || (unsigned __int8)sub_EAFAF0(a1, &v61, &v62) )
          goto LABEL_30;
        v59 = 1;
      }
      else
      {
        if ( v66.m128i_i64[1] != 6
          || *(_DWORD *)v66.m128i_i64[0] != 1920298867
          || *(_WORD *)(v66.m128i_i64[0] + 4) != 25955 )
        {
LABEL_11:
          v83[0].m128i_i64[0] = (__int64)"unexpected token in '.file' directive";
          v84 = 259;
          v3 = sub_ECE0E0(a1, v83, 0, 0);
          goto LABEL_12;
        }
        v82 = 1;
        v79[0] = "source specified, but no file number";
        v81 = 3;
        if ( (unsigned __int8)sub_ECE0A0(a1, v60 == -1, v79)
          || (v83[0].m128i_i64[0] = (__int64)"unexpected token in '.file' directive",
              v84 = 259,
              v8 = (_DWORD *)sub_ECD7B0(a1),
              (unsigned __int8)sub_ECE0A0(a1, *v8 != 3, v83))
          || (unsigned __int8)sub_EAE3B0((_QWORD *)a1, &src) )
        {
LABEL_30:
          v3 = 1;
          goto LABEL_12;
        }
        v5 = 1;
      }
    }
    v13 = *(_QWORD *)(a1 + 224);
    if ( v60 == -1 )
    {
      if ( *(_BYTE *)(*(_QWORD *)(v13 + 152) + 290LL) )
        (*(void (__fastcall **)(_QWORD, _QWORD *, __int64))(**(_QWORD **)(a1 + 232) + 632LL))(
          *(_QWORD *)(a1 + 232),
          v57,
          v58);
      else
        v3 = 0;
LABEL_12:
      if ( src != v78 )
        j_j___libc_free_0(src, v78[0] + 1LL);
      goto LABEL_14;
    }
    if ( !*(_BYTE *)(v13 + 1793) )
    {
LABEL_41:
      v14 = 0;
      v67 = 0;
      v66 = 0;
      if ( v59 )
      {
        v15 = (unsigned __int64)v61;
        v16 = v83;
        v17 = (unsigned __int64)v62;
        v18 = 56;
        do
        {
          v16 = (__m128i *)((char *)v16 + 1);
          v16[-1].m128i_i8[15] = v15 >> v18;
          v19 = v17 >> v18;
          v18 -= 8;
          v16->m128i_i8[7] = v19;
        }
        while ( &v83[0].m128i_u64[1] != (unsigned __int64 *)v16 );
        v14 = 1;
        v66 = _mm_loadu_si128(v83);
      }
      v20 = 0;
      if ( v5 )
      {
        v34 = *(_QWORD **)(a1 + 224);
        v35 = n;
        v36 = v34[24];
        v37 = (unsigned int)n;
        v34[34] += (unsigned int)n;
        v38 = (v36 + 7) & 0xFFFFFFFFFFFFFFF8LL;
        if ( v34[25] >= v37 + v38 && v36 )
        {
          v34[24] = v37 + v38;
          v39 = (void *)((v36 + 7) & 0xFFFFFFFFFFFFFFF8LL);
        }
        else
        {
          v51 = sub_9D1E70((__int64)(v34 + 24), v37, v37, 3);
          v35 = n;
          v39 = (void *)v51;
        }
        v40 = memcpy(v39, src, v35);
        v20 = 1;
        *(_QWORD *)&v68 = v40;
        *((_QWORD *)&v68 + 1) = n;
      }
      if ( v60 )
      {
        v21 = *(_QWORD *)(a1 + 232);
        LOBYTE(v69) = v20;
        v22 = _mm_loadu_si128(&v66);
        v23 = *(void (__fastcall **)(__int64 *, __int64, _QWORD, _QWORD *, __int64, _QWORD, _QWORD *, __int64, __int64, __int64, char, _QWORD, _QWORD, __int64))(*(_QWORD *)v21 + 656LL);
        v67 = v14;
        v23(
          &v64,
          v21,
          (unsigned int)v60,
          v55,
          v54,
          0,
          v57,
          v58,
          v22.m128i_i64[0],
          v22.m128i_i64[1],
          v14,
          v68,
          *((_QWORD *)&v68 + 1),
          v69);
        if ( (v65 & 1) != 0 )
        {
          v65 &= ~2u;
          v24 = v64;
          v64 = 0;
          v63 = v24 | 1;
          sub_C64870((__int64)v79, &v63);
          v25 = a2;
          v84 = 260;
          v83[0].m128i_i64[0] = (__int64)v79;
          v3 = sub_ECDA70(a1, a2, v83, 0, 0);
          if ( (__int64 *)v79[0] != &v80 )
          {
            v25 = v80 + 1;
            j_j___libc_free_0(v79[0], v80 + 1);
          }
          if ( (v63 & 1) != 0 || (v63 & 0xFFFFFFFFFFFFFFFELL) != 0 )
            sub_C63C30(&v63, v25);
          if ( (v65 & 2) != 0 )
            sub_9CE230(&v64);
          if ( (v65 & 1) != 0 && v64 )
            (*(void (__fastcall **)(__int64))(*(_QWORD *)v64 + 8LL))(v64);
          goto LABEL_12;
        }
      }
      else
      {
        v26 = *(_QWORD *)(a1 + 224);
        if ( *(_WORD *)(v26 + 1904) <= 4u )
          *(_WORD *)(v26 + 1904) = 5;
        v27 = *(_QWORD *)(a1 + 232);
        LOBYTE(v69) = v20;
        v28 = _mm_loadu_si128(&v66);
        v29 = *(void (__fastcall **)(__int64, _QWORD *, __int64, _QWORD *, __int64, _QWORD, __int64, __int64, char, _QWORD, _QWORD, __int64))(*(_QWORD *)v27 + 664LL);
        v67 = v14;
        v29(v27, v55, v54, v57, v58, 0, v28.m128i_i64[0], v28.m128i_i64[1], v14, v68, *((_QWORD *)&v68 + 1), v69);
      }
      if ( !*(_BYTE *)(a1 + 870) )
      {
        v30 = *(_QWORD *)(a1 + 224);
        v31 = v30 + 1736;
        v32 = *(_QWORD *)(v30 + 1744);
        if ( v32 )
        {
          do
          {
            v33 = v32;
            v32 = *(_QWORD *)(v32 + 16);
          }
          while ( v32 );
          if ( v33 != v31 && !*(_DWORD *)(v33 + 32) )
            v31 = v33;
        }
        if ( *(_DWORD *)(v31 + 168) && *(_BYTE *)(v31 + 553) != *(_BYTE *)(v31 + 554) )
        {
          *(_BYTE *)(a1 + 870) = 1;
          v83[0].m128i_i64[0] = (__int64)"inconsistent use of MD5 checksums";
          v84 = 259;
          v3 = sub_EA8060((_QWORD *)a1, a2, (__int64)v83, 0, 0);
        }
      }
      goto LABEL_12;
    }
    LODWORD(v79[0]) = 0;
    v41 = *(_QWORD *)(v13 + 1744);
    if ( v41 )
    {
      do
      {
        v42 = v41;
        v41 = *(_QWORD *)(v41 + 16);
      }
      while ( v41 );
      if ( v13 + 1736 != v42 && !*(_DWORD *)(v42 + 32) )
        goto LABEL_77;
    }
    else
    {
      v42 = v13 + 1736;
    }
    v83[0].m128i_i64[0] = (__int64)v79;
    v42 = sub_EAA600((_QWORD *)(v13 + 1728), v42, (unsigned int **)v83);
LABEL_77:
    v43 = *(_QWORD *)(v42 + 48);
    if ( v43 != v43 + 32LL * *(unsigned int *)(v42 + 56) )
    {
      v44 = (_QWORD *)(v43 + 32LL * *(unsigned int *)(v42 + 56));
      v45 = *(_QWORD **)(v42 + 48);
      do
      {
        v44 -= 4;
        if ( (_QWORD *)*v44 != v44 + 2 )
        {
          v52 = v44;
          j_j___libc_free_0(*v44, v44[2] + 1LL);
          v44 = v52;
        }
      }
      while ( v45 != v44 );
    }
    v46 = *(unsigned int *)(v42 + 168);
    v47 = *(_QWORD **)(v42 + 160);
    *(_DWORD *)(v42 + 56) = 0;
    if ( v47 != &v47[10 * v46] )
    {
      v48 = &v47[10 * v46];
      v49 = v47;
      do
      {
        v48 -= 10;
        if ( (_QWORD *)*v48 != v48 + 2 )
        {
          v53 = v48;
          j_j___libc_free_0(*v48, v48[2] + 1LL);
          v48 = v53;
        }
      }
      while ( v49 != v48 );
    }
    v50 = *(_BYTE **)(v42 + 472);
    *(_DWORD *)(v42 + 168) = 0;
    *(_QWORD *)(v42 + 480) = 0;
    *v50 = 0;
    *(_WORD *)(v42 + 552) = 256;
    *(_BYTE *)(v42 + 554) = 0;
    *(_BYTE *)(*(_QWORD *)(a1 + 224) + 1793LL) = 0;
    goto LABEL_41;
  }
LABEL_16:
  if ( v70 != v72 )
    j_j___libc_free_0(v70, v72[0] + 1LL);
  return v3;
}
