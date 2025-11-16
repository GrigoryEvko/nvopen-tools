// Function: sub_2F151D0
// Address: 0x2f151d0
//
void __fastcall sub_2F151D0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v5; // rbx
  unsigned int v6; // r13d
  __int64 v7; // rdx
  __int64 v8; // rax
  unsigned int v9; // edi
  void (__fastcall *v10)(char **, __int64, _QWORD, __int64); // rax
  __int64 v11; // rsi
  __m128i v12; // xmm5
  __m128i v13; // xmm6
  char *v14; // rbx
  __int64 v15; // rdx
  __m128i *v16; // rax
  __int64 v17; // rdx
  __int64 v18; // rcx
  _BYTE *v19; // rsi
  __m128i v20; // xmm0
  unsigned __int64 v21; // r12
  unsigned __int64 *v22; // rbx
  unsigned int *i; // r14
  __m128i v24; // xmm2
  __m128i *v25; // rdi
  __m128i v26; // xmm3
  __m128i *v27; // rsi
  unsigned int v28; // ecx
  __int64 v29; // rax
  unsigned int v30; // edi
  _WORD *v31; // rax
  _WORD *v32; // r13
  char *v33; // rcx
  char *v34; // rdx
  __int64 v35; // rax
  unsigned __int64 v36; // r13
  unsigned __int64 *v37; // r12
  unsigned __int64 *v38; // rbx
  unsigned __int64 *v39; // rbx
  unsigned __int64 *v40; // r12
  int v42; // [rsp+18h] [rbp-1B8h]
  unsigned int *v46; // [rsp+48h] [rbp-188h]
  __int32 v47; // [rsp+50h] [rbp-180h]
  unsigned int v48; // [rsp+54h] [rbp-17Ch]
  char *v49; // [rsp+58h] [rbp-178h]
  unsigned int v50; // [rsp+58h] [rbp-178h]
  __m128i *v51; // [rsp+60h] [rbp-170h] BYREF
  __int64 v52; // [rsp+68h] [rbp-168h]
  __m128i v53; // [rsp+70h] [rbp-160h] BYREF
  __int64 v54; // [rsp+80h] [rbp-150h]
  __int64 v55; // [rsp+88h] [rbp-148h]
  __m128i v56; // [rsp+90h] [rbp-140h] BYREF
  __m128i v57; // [rsp+A0h] [rbp-130h] BYREF
  __m128i v58; // [rsp+B0h] [rbp-120h] BYREF
  __int64 v59; // [rsp+C0h] [rbp-110h]
  __int64 v60; // [rsp+C8h] [rbp-108h]
  char *v61; // [rsp+D0h] [rbp-100h] BYREF
  char *v62; // [rsp+D8h] [rbp-F8h]
  _QWORD v63[6]; // [rsp+E0h] [rbp-F0h] BYREF
  __m128i v64; // [rsp+110h] [rbp-C0h] BYREF
  __m128i v65; // [rsp+120h] [rbp-B0h] BYREF
  __m128i v66; // [rsp+130h] [rbp-A0h] BYREF
  __m128i *v67; // [rsp+140h] [rbp-90h] BYREF
  __m128i v68; // [rsp+148h] [rbp-88h] BYREF
  __m128i *v69; // [rsp+158h] [rbp-78h] BYREF
  __m128i v70; // [rsp+160h] [rbp-70h] BYREF
  __m128i v71; // [rsp+178h] [rbp-58h] BYREF
  unsigned __int64 *v72; // [rsp+188h] [rbp-48h] BYREF
  unsigned __int64 v73; // [rsp+190h] [rbp-40h]
  __int64 v74; // [rsp+198h] [rbp-38h]

  v47 = 0;
  *(_BYTE *)(a2 + 23) = (*(_QWORD *)(*(_QWORD *)a4 + 344LL) & 4LL) != 0;
  v42 = *(_DWORD *)(a4 + 64);
  if ( v42 )
  {
    while ( 1 )
    {
      v70.m128i_i8[8] = 0;
      v64.m128i_i64[1] = 0;
      v65.m128i_i64[1] = (__int64)&v66.m128i_i64[1];
      v65.m128i_i64[0] = 0;
      v69 = (__m128i *)&v70.m128i_u64[1];
      v66.m128i_i64[0] = 0;
      v66.m128i_i8[8] = 0;
      v5 = v47 & 0x7FFFFFFF;
      v68 = 0u;
      v70.m128i_i64[0] = 0;
      v71 = 0u;
      v72 = 0;
      v73 = 0;
      v74 = 0;
      v64.m128i_i32[0] = v47;
      if ( *(_DWORD *)(a4 + 104) <= (unsigned int)v5 || !*(_QWORD *)(*(_QWORD *)(a4 + 96) + 32LL * (unsigned int)v5 + 8) )
        break;
LABEL_4:
      if ( ++v47 == v42 )
        goto LABEL_49;
    }
    v62 = 0;
    v63[3] = 0x100000000LL;
    v61 = (char *)&unk_49DD210;
    v6 = v47 | 0x80000000;
    memset(v63, 0, 24);
    v63[4] = &v65.m128i_i64[1];
    sub_CB5980((__int64)&v61, 0, 0, 0);
    sub_2FF63B0(&v57, v47 | 0x80000000, a4, a5);
    if ( !v58.m128i_i64[0] )
      sub_4263D6(&v57, v6, v7);
    ((void (__fastcall *)(__m128i *, char **))v58.m128i_i64[1])(&v57, &v61);
    if ( v58.m128i_i64[0] )
      ((void (__fastcall *)(__m128i *, __m128i *, __int64))v58.m128i_i64[0])(&v57, &v57, 3);
    v61 = (char *)&unk_49DD210;
    sub_CB5840((__int64)&v61);
    if ( (unsigned int)v5 < *(_DWORD *)(a4 + 248) )
    {
      v8 = *(_QWORD *)(a4 + 240) + 40 * v5;
      if ( *(_DWORD *)(v8 + 16) )
      {
        v9 = **(_DWORD **)(v8 + 8);
        if ( !*(_DWORD *)v8 )
        {
          if ( v9 )
            sub_2F07630(v9, (__int64)&v69, a5);
        }
      }
    }
    v10 = *(void (__fastcall **)(char **, __int64, _QWORD, __int64))(*(_QWORD *)a5 + 712LL);
    if ( (char *)v10 == (char *)sub_2F07210 )
    {
LABEL_13:
      v11 = *(_QWORD *)(a2 + 56);
      if ( v11 == *(_QWORD *)(a2 + 64) )
      {
        sub_2F14850((unsigned __int64 *)(a2 + 48), v11, (__int64)&v64);
        v21 = v73;
        v22 = v72;
      }
      else
      {
        if ( v11 )
        {
          *(__m128i *)v11 = _mm_load_si128(&v64);
          *(_QWORD *)(v11 + 16) = v65.m128i_i64[0];
          *(_QWORD *)(v11 + 24) = v11 + 40;
          if ( (unsigned __int64 *)v65.m128i_i64[1] == &v66.m128i_u64[1] )
          {
            *(__m128i *)(v11 + 40) = _mm_loadu_si128((const __m128i *)&v66.m128i_u64[1]);
          }
          else
          {
            *(_QWORD *)(v11 + 24) = v65.m128i_i64[1];
            *(_QWORD *)(v11 + 40) = v66.m128i_i64[1];
          }
          *(_QWORD *)(v11 + 32) = v66.m128i_i64[0];
          v12 = _mm_loadu_si128(&v68);
          v66.m128i_i8[8] = 0;
          v65.m128i_i64[1] = (__int64)&v66.m128i_i64[1];
          v66.m128i_i64[0] = 0;
          *(_QWORD *)(v11 + 72) = v11 + 88;
          *(__m128i *)(v11 + 56) = v12;
          if ( v69 == (__m128i *)&v70.m128i_u64[1] )
          {
            *(__m128i *)(v11 + 88) = _mm_loadu_si128((const __m128i *)&v70.m128i_u64[1]);
          }
          else
          {
            *(_QWORD *)(v11 + 72) = v69;
            *(_QWORD *)(v11 + 88) = v70.m128i_i64[1];
          }
          *(_QWORD *)(v11 + 80) = v70.m128i_i64[0];
          v13 = _mm_loadu_si128(&v71);
          v70.m128i_i64[0] = 0;
          v69 = (__m128i *)&v70.m128i_u64[1];
          v70.m128i_i8[8] = 0;
          *(__m128i *)(v11 + 104) = v13;
          *(_QWORD *)(v11 + 120) = v72;
          *(_QWORD *)(v11 + 128) = v73;
          *(_QWORD *)(v11 + 136) = v74;
          *(_QWORD *)(a2 + 56) += 144LL;
LABEL_20:
          if ( v69 != (__m128i *)&v70.m128i_u64[1] )
            j_j___libc_free_0((unsigned __int64)v69);
          if ( (unsigned __int64 *)v65.m128i_i64[1] != &v66.m128i_u64[1] )
            j_j___libc_free_0(v65.m128i_u64[1]);
          goto LABEL_4;
        }
        v21 = v73;
        v22 = v72;
        *(_QWORD *)(a2 + 56) = 144;
      }
      if ( v22 != (unsigned __int64 *)v21 )
      {
        do
        {
          if ( (unsigned __int64 *)*v22 != v22 + 2 )
            j_j___libc_free_0(*v22);
          v22 += 6;
        }
        while ( v22 != (unsigned __int64 *)v21 );
        v21 = (unsigned __int64)v72;
      }
      if ( v21 )
        j_j___libc_free_0(v21);
      goto LABEL_20;
    }
    v10(&v61, a5, v6, a3);
    v14 = v61;
    v49 = &v61[16 * (unsigned int)v62];
    if ( v61 == v49 )
    {
LABEL_39:
      if ( v49 != (char *)v63 )
        _libc_free((unsigned __int64)v49);
      goto LABEL_13;
    }
    while ( 1 )
    {
      v19 = *(_BYTE **)v14;
      if ( !*(_QWORD *)v14 )
        break;
      v15 = *((_QWORD *)v14 + 1);
      v51 = &v53;
      sub_2F07580((__int64 *)&v51, v19, (__int64)&v19[v15]);
      v16 = v51;
      v17 = v52;
      if ( v51 == &v53 )
        goto LABEL_36;
      v18 = v53.m128i_i64[0];
      v54 = (__int64)v51;
      v55 = v52;
      v56.m128i_i64[0] = v53.m128i_i64[0];
      v51 = &v53;
      v52 = 0;
      v53.m128i_i8[0] = 0;
      v57.m128i_i64[0] = (__int64)&v58;
      if ( v16 == &v56 )
        goto LABEL_37;
      v57.m128i_i64[0] = (__int64)v16;
      v58.m128i_i64[0] = v18;
LABEL_29:
      v57.m128i_i64[1] = v17;
      v59 = 0;
      v60 = 0;
      sub_2F147D0((unsigned __int64 *)&v72, &v57);
      if ( (__m128i *)v57.m128i_i64[0] != &v58 )
        j_j___libc_free_0(v57.m128i_u64[0]);
      if ( v51 != &v53 )
        j_j___libc_free_0((unsigned __int64)v51);
      v14 += 16;
      if ( v49 == v14 )
      {
        v49 = v61;
        goto LABEL_39;
      }
    }
    v53.m128i_i8[0] = 0;
    v17 = 0;
LABEL_36:
    v20 = _mm_load_si128(&v53);
    v51 = &v53;
    v52 = 0;
    v53.m128i_i8[0] = 0;
    v57.m128i_i64[0] = (__int64)&v58;
    v56 = v20;
LABEL_37:
    v58 = _mm_load_si128(&v56);
    goto LABEL_29;
  }
LABEL_49:
  v46 = *(unsigned int **)(a4 + 496);
  if ( *(unsigned int **)(a4 + 488) != v46 )
  {
    for ( i = *(unsigned int **)(a4 + 488); v46 != i; i += 2 )
    {
      v28 = i[1];
      v29 = *(_QWORD *)i;
      v30 = *i;
      v68.m128i_i8[8] = 0;
      v50 = v28;
      v48 = HIDWORD(v29);
      v64 = (__m128i)(unsigned __int64)&v65;
      v65.m128i_i8[0] = 0;
      v66 = 0u;
      v67 = (__m128i *)&v68.m128i_u64[1];
      v68.m128i_i64[0] = 0;
      v70 = 0u;
      sub_2F07630(v30, (__int64)&v64, a5);
      if ( v50 )
      {
        sub_2F07630(v48, (__int64)&v67, a5);
        v27 = *(__m128i **)(a2 + 80);
        if ( v27 != *(__m128i **)(a2 + 88) )
        {
LABEL_64:
          if ( v27 )
          {
            v27->m128i_i64[0] = (__int64)v27[1].m128i_i64;
            if ( (__m128i *)v64.m128i_i64[0] == &v65 )
            {
              v27[1] = _mm_load_si128(&v65);
            }
            else
            {
              v27->m128i_i64[0] = v64.m128i_i64[0];
              v27[1].m128i_i64[0] = v65.m128i_i64[0];
            }
            v27->m128i_i64[1] = v64.m128i_i64[1];
            v24 = _mm_load_si128(&v66);
            v64 = (__m128i)(unsigned __int64)&v65;
            v65.m128i_i8[0] = 0;
            v27[3].m128i_i64[0] = (__int64)v27[4].m128i_i64;
            v27[2] = v24;
            if ( v67 == (__m128i *)&v68.m128i_u64[1] )
            {
              v27[4] = _mm_load_si128((const __m128i *)&v68.m128i_u64[1]);
            }
            else
            {
              v27[3].m128i_i64[0] = (__int64)v67;
              v27[4].m128i_i64[0] = v68.m128i_i64[1];
            }
            v25 = (__m128i *)&v68.m128i_u64[1];
            v27[3].m128i_i64[1] = v68.m128i_i64[0];
            v26 = _mm_load_si128(&v70);
            v67 = (__m128i *)&v68.m128i_u64[1];
            v68.m128i_i64[0] = 0;
            v68.m128i_i8[8] = 0;
            v27[5] = v26;
            v27 = *(__m128i **)(a2 + 80);
          }
          else
          {
            v25 = v67;
          }
          *(_QWORD *)(a2 + 80) = v27 + 6;
          goto LABEL_57;
        }
      }
      else
      {
        v27 = *(__m128i **)(a2 + 80);
        if ( v27 != *(__m128i **)(a2 + 88) )
          goto LABEL_64;
      }
      sub_2F14DC0((unsigned __int64 *)(a2 + 72), v27, &v64);
      v25 = v67;
LABEL_57:
      if ( v25 != (__m128i *)&v68.m128i_u64[1] )
        j_j___libc_free_0((unsigned __int64)v25);
      if ( (__m128i *)v64.m128i_i64[0] != &v65 )
        j_j___libc_free_0(v64.m128i_u64[0]);
    }
  }
  if ( *(_BYTE *)(a4 + 176) )
  {
    v31 = sub_2EBFBC0((_QWORD *)a4);
    v61 = 0;
    v62 = 0;
    v32 = v31;
    v63[0] = 0;
    if ( *v31 )
    {
      do
      {
        v64 = (__m128i)(unsigned __int64)&v65;
        v65.m128i_i8[0] = 0;
        v66 = 0u;
        sub_2F07630((unsigned __int16)*v32, (__int64)&v64, a5);
        sub_2F147D0((unsigned __int64 *)&v61, &v64);
        if ( (__m128i *)v64.m128i_i64[0] != &v65 )
          j_j___libc_free_0(v64.m128i_u64[0]);
        ++v32;
      }
      while ( *v32 );
      v33 = v61;
      v34 = v62;
      v35 = v63[0];
    }
    else
    {
      v35 = 0;
      v34 = 0;
      v33 = 0;
    }
    if ( *(_BYTE *)(a2 + 120) )
    {
      v36 = *(_QWORD *)(a2 + 96);
      v37 = *(unsigned __int64 **)(a2 + 104);
      *(_QWORD *)(a2 + 96) = v33;
      *(_QWORD *)(a2 + 104) = v34;
      *(_QWORD *)(a2 + 112) = v35;
      v38 = (unsigned __int64 *)v36;
      v61 = 0;
      v62 = 0;
      for ( v63[0] = 0; v37 != v38; v38 += 6 )
      {
        if ( (unsigned __int64 *)*v38 != v38 + 2 )
          j_j___libc_free_0(*v38);
      }
      if ( v36 )
        j_j___libc_free_0(v36);
      v39 = (unsigned __int64 *)v62;
      v40 = (unsigned __int64 *)v61;
      if ( v62 != v61 )
      {
        do
        {
          if ( (unsigned __int64 *)*v40 != v40 + 2 )
            j_j___libc_free_0(*v40);
          v40 += 6;
        }
        while ( v39 != v40 );
        v40 = (unsigned __int64 *)v61;
      }
      if ( v40 )
        j_j___libc_free_0((unsigned __int64)v40);
    }
    else
    {
      *(_QWORD *)(a2 + 96) = v33;
      *(_QWORD *)(a2 + 104) = v34;
      *(_QWORD *)(a2 + 112) = v35;
      *(_BYTE *)(a2 + 120) = 1;
    }
  }
}
