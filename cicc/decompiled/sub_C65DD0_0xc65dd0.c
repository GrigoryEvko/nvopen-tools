// Function: sub_C65DD0
// Address: 0xc65dd0
//
__int64 __fastcall sub_C65DD0(__int64 a1, char *a2, size_t a3)
{
  __int64 v3; // rbx
  size_t v4; // r15
  _BYTE *v7; // rax
  size_t v8; // rax
  char v9; // dl
  __int8 v10; // r9
  __int32 v11; // r10d
  __int32 v12; // eax
  __int32 v13; // r8d
  unsigned __int64 v14; // rcx
  unsigned __int64 v15; // rsi
  unsigned __int64 v16; // rdi
  int v17; // eax
  unsigned __int64 v18; // rdx
  __m128i v19; // xmm1
  __m128i v20; // xmm3
  __m128i v21; // xmm2
  char v22; // al
  char v23; // si
  __int64 v24; // rdx
  char *v25; // rcx
  unsigned __int64 v26; // rax
  unsigned __int64 v27; // rax
  _BYTE *v29; // rax
  size_t v30; // rcx
  _BYTE *v31; // rax
  size_t v32; // rax
  const __m128i *v33; // r8
  __m128i v34; // xmm5
  __m128i v35; // xmm6
  __m128i *v36; // rax
  char *v37; // r14
  unsigned __int64 v38; // rax
  size_t v39; // rax
  size_t v40; // rcx
  unsigned __int64 v41; // rax
  unsigned __int64 v42; // rcx
  void *v43; // r8
  unsigned __int64 v44; // rax
  unsigned __int64 v45; // rax
  unsigned __int64 v46; // rax
  unsigned __int64 v47; // rcx
  unsigned __int64 v48; // rsi
  unsigned __int64 v49; // rax
  char *v50; // rcx
  unsigned __int64 v51; // rax
  __int8 v52; // r9
  unsigned __int64 v53; // rax
  unsigned __int64 v54; // rcx
  void *v55; // r11
  unsigned __int64 v56; // rax
  unsigned __int64 v57; // rax
  int v58; // r8d
  char v59; // r9
  char v60; // al
  char v61; // dl
  unsigned __int64 v62; // rdx
  char *v63; // rax
  __int32 v64; // [rsp+8h] [rbp-118h]
  __int32 v65; // [rsp+14h] [rbp-10Ch]
  __int32 v66; // [rsp+14h] [rbp-10Ch]
  int v67; // [rsp+14h] [rbp-10Ch]
  size_t v68; // [rsp+18h] [rbp-108h]
  __int32 v69; // [rsp+18h] [rbp-108h]
  __int32 v70; // [rsp+18h] [rbp-108h]
  __int8 v71; // [rsp+18h] [rbp-108h]
  __int32 v72; // [rsp+18h] [rbp-108h]
  __int64 v73; // [rsp+20h] [rbp-100h]
  void *s; // [rsp+28h] [rbp-F8h]
  void *sa; // [rsp+28h] [rbp-F8h]
  __int32 sb; // [rsp+28h] [rbp-F8h]
  __int8 sc; // [rsp+28h] [rbp-F8h]
  void *sd; // [rsp+28h] [rbp-F8h]
  char se; // [rsp+28h] [rbp-F8h]
  const char *v80; // [rsp+30h] [rbp-F0h]
  __int64 v81; // [rsp+38h] [rbp-E8h]
  __int8 v82; // [rsp+38h] [rbp-E8h]
  unsigned __int64 v83; // [rsp+40h] [rbp-E0h]
  __int32 v84; // [rsp+48h] [rbp-D8h]
  __int32 v85; // [rsp+4Ch] [rbp-D4h]
  const char *v86; // [rsp+50h] [rbp-D0h] BYREF
  unsigned __int64 v87; // [rsp+58h] [rbp-C8h]
  char *v88; // [rsp+60h] [rbp-C0h] BYREF
  unsigned __int64 v89; // [rsp+68h] [rbp-B8h]
  _OWORD v90[3]; // [rsp+70h] [rbp-B0h] BYREF
  __int64 v91; // [rsp+A0h] [rbp-80h]
  __m128i v92; // [rsp+B0h] [rbp-70h] BYREF
  __m128i v93; // [rsp+C0h] [rbp-60h] BYREF
  __m128i v94; // [rsp+D0h] [rbp-50h] BYREF
  __int128 v95; // [rsp+E0h] [rbp-40h]

  v4 = a3;
  v73 = a1 + 16;
  *(_QWORD *)a1 = a1 + 16;
  *(_QWORD *)(a1 + 8) = 0x200000000LL;
  v84 = 0;
  while ( v4 )
  {
    v22 = *a2;
    v92 = 0;
    v93 = 0;
    v94 = 0;
    v95 = 0;
    if ( v22 != 123 )
    {
      v7 = memchr(a2, 123, v4);
      if ( v7 )
      {
        v8 = v7 - a2;
        if ( v8 > v4 )
          v8 = v4;
        v3 = v8;
        v4 -= v8;
      }
      else
      {
        v3 = v4;
        v4 = 0;
      }
LABEL_7:
      v80 = a2;
      a2 += v3;
LABEL_8:
      v9 = 1;
      v10 = 0;
      v11 = 0;
      v12 = 1;
      v85 = 0;
      v13 = 2;
      v83 = 0;
      v81 = 0;
      goto LABEL_9;
    }
    v23 = 123;
    s = a2 + 1;
    while ( sub_C65DC0((__int64)v90, v23) )
    {
      if ( v24 == 1 )
      {
        v26 = v4;
        if ( v4 <= 1 )
          goto LABEL_34;
LABEL_26:
        v3 = v26 >> 1;
        if ( v26 >> 1 > v4 )
          v3 = v4;
        v27 = v26 & 0xFFFFFFFFFFFFFFFELL;
        if ( v27 > v4 )
        {
          v27 = v4;
          v4 = 0;
        }
        else
        {
          v4 -= v27;
        }
        v80 = a2;
        a2 += v27;
        goto LABEL_8;
      }
      v23 = *v25;
    }
    v26 = v4 - v24;
    if ( v4 - v24 > v4 )
      v26 = v4;
    if ( v26 > 1 )
      goto LABEL_26;
LABEL_34:
    v29 = memchr(a2, 125, v4);
    if ( !v29 || (v30 = v29 - a2, v29 - a2 == -1) )
    {
      v4 = 0;
      a2 = 0;
      v10 = 0;
      v80 = "Unterminated brace sequence. Escape with {{ for a literal brace.";
      v9 = 1;
      v13 = 2;
      v11 = 0;
      v83 = 0;
      v3 = 64;
      v12 = 1;
      v81 = 0;
      v85 = 0;
      goto LABEL_9;
    }
    if ( v4 != 1 )
    {
      v68 = v29 - a2;
      v31 = memchr(s, 123, v4 - 1);
      v30 = v68;
      if ( v31 )
      {
        v32 = v31 - a2;
        if ( v68 > v32 )
        {
          if ( v32 > v4 )
            v32 = v4;
          v3 = v32;
          v4 -= v32;
          goto LABEL_7;
        }
      }
    }
    v38 = 0;
    if ( v30 )
    {
      v39 = v4;
      if ( v30 <= v4 )
        v39 = v30;
      v38 = v39 - 1;
    }
    v40 = v30 + 1;
    if ( v40 > v4 )
    {
      v40 = v4;
      v4 = 0;
    }
    else
    {
      v4 -= v40;
    }
    a2 += v40;
    v87 = v38;
    v86 = (const char *)s;
    v41 = sub_C935B0(&v86, "{}", 2, 0);
    v42 = v87;
    v43 = 0;
    if ( v41 < v87 )
    {
      v43 = (void *)(v87 - v41);
      v42 = v41;
    }
    *(_QWORD *)&v90[0] = &v86[v42];
    *((_QWORD *)&v90[0] + 1) = v43;
    sa = v43;
    v44 = sub_C93740(v90, "{}", 2, -1) + 1;
    v88 = *(char **)&v90[0];
    if ( v44 > *((_QWORD *)&v90[0] + 1) )
      v44 = *((_QWORD *)&v90[0] + 1);
    v45 = *((_QWORD *)&v90[0] + 1) - (_QWORD)sa + v44;
    if ( v45 > *((_QWORD *)&v90[0] + 1) )
      v45 = *((_QWORD *)&v90[0] + 1);
    v89 = v45;
    v46 = sub_C935B0(&v88, &unk_3F15413, 6, 0);
    v47 = v89;
    v48 = 0;
    if ( v46 < v89 )
    {
      v48 = v89 - v46;
      v47 = v46;
    }
    v89 = v48;
    v88 += v47;
    if ( (unsigned __int8)sub_C93B20(&v88, 0, v90) || (v11 = v90[0], *(_QWORD *)&v90[0] != LODWORD(v90[0])) )
      v11 = -1;
    v49 = v89;
    if ( !v89 || (v50 = v88, *v88 != 44) || (++v88, --v89, v49 == 1) )
    {
      v85 = 0;
      v13 = 2;
      v10 = 32;
      goto LABEL_68;
    }
    v58 = 2;
    if ( v49 == 2 )
      goto LABEL_83;
    v61 = v50[2];
    v59 = v50[1];
    switch ( v61 )
    {
      case '-':
        v58 = 0;
        break;
      case '=':
        v58 = 1;
        break;
      case '+':
        v58 = 2;
        break;
      default:
        if ( v59 == 45 )
        {
          v58 = 0;
        }
        else
        {
          if ( v59 != 61 )
          {
            v58 = 2;
            if ( v59 == 43 )
              goto LABEL_94;
LABEL_83:
            v59 = 32;
            goto LABEL_84;
          }
          v58 = 1;
        }
LABEL_94:
        v88 = v50 + 2;
        v89 = v49 - 2;
        goto LABEL_83;
    }
    v88 = v50 + 3;
    v89 = v49 - 3;
LABEL_84:
    v67 = v58;
    v72 = v11;
    se = v59;
    v60 = sub_C93B20(&v88, 0, v90);
    v10 = se;
    v11 = v72;
    v13 = v67;
    if ( v60 || *(_QWORD *)&v90[0] != LODWORD(v90[0]) )
    {
      v9 = 0;
      v12 = 0;
      goto LABEL_9;
    }
    v85 = v90[0];
LABEL_68:
    v69 = v13;
    sb = v11;
    v82 = v10;
    v51 = sub_C935B0(&v88, &unk_3F15413, 6, 0);
    v52 = v82;
    if ( v51 < v89 )
    {
      v62 = v89 - v51;
      v63 = &v88[v51];
      v88 = v63;
      v89 = v62;
      if ( *v63 == 58 )
      {
        v88 = 0;
        v83 = v62 - 1;
        v81 = (__int64)(v63 + 1);
        v89 = 0;
        goto LABEL_71;
      }
    }
    else
    {
      v88 += v89;
      v89 = 0;
    }
    v83 = 0;
    v81 = 0;
LABEL_71:
    v65 = v69;
    v70 = sb;
    sc = v52;
    v53 = sub_C935B0(&v88, &unk_3F15413, 6, 0);
    v54 = v89;
    v55 = 0;
    if ( v53 < v89 )
    {
      v55 = (void *)(v89 - v53);
      v54 = v53;
    }
    v64 = v65;
    v66 = v70;
    v71 = sc;
    *((_QWORD *)&v90[0] + 1) = v55;
    sd = v55;
    *(_QWORD *)&v90[0] = &v88[v54];
    v56 = sub_C93740(v90, &unk_3F15413, 6, -1) + 1;
    v10 = v71;
    v11 = v66;
    v13 = v64;
    if ( v56 > *((_QWORD *)&v90[0] + 1) )
      v56 = *((_QWORD *)&v90[0] + 1);
    v57 = *((_QWORD *)&v90[0] + 1) - (_QWORD)sd + v56;
    if ( v57 > *((_QWORD *)&v90[0] + 1) )
      v57 = *((_QWORD *)&v90[0] + 1);
    v9 = 0;
    if ( !v57 )
    {
      v3 = v87;
      v9 = 1;
      v80 = v86;
    }
    v12 = 0;
LABEL_9:
    v92.m128i_i32[0] = v12;
    v93.m128i_i64[0] = v3;
    v92.m128i_i64[1] = (__int64)v80;
    v94.m128i_i32[0] = v13;
    v93.m128i_i32[3] = v85;
    v94.m128i_i8[4] = v10;
    v94.m128i_i64[1] = v81;
    *(_QWORD *)&v95 = v83;
    if ( v9 )
    {
      if ( !v12 && v11 == -1 )
        v11 = v84++;
      v14 = *(unsigned int *)(a1 + 8);
      v15 = *(_QWORD *)a1;
      v16 = *(unsigned int *)(a1 + 12);
      v17 = *(_DWORD *)(a1 + 8);
      v18 = *(_QWORD *)a1 + 56 * v14;
      if ( v14 >= v16 )
      {
        v93.m128i_i32[2] = v11;
        v33 = (const __m128i *)v90;
        v34 = _mm_loadu_si128(&v93);
        v35 = _mm_loadu_si128(&v94);
        v90[0] = _mm_loadu_si128(&v92);
        v90[1] = v34;
        v91 = v95;
        v90[2] = v35;
        if ( v16 < v14 + 1 )
        {
          if ( v15 > (unsigned __int64)v90 || v18 <= (unsigned __int64)v90 )
          {
            sub_C8D5F0(a1, v73, v14 + 1, 56);
            v15 = *(_QWORD *)a1;
            v14 = *(unsigned int *)(a1 + 8);
            v33 = (const __m128i *)v90;
          }
          else
          {
            v37 = (char *)v90 - v15;
            sub_C8D5F0(a1, v73, v14 + 1, 56);
            v15 = *(_QWORD *)a1;
            v14 = *(unsigned int *)(a1 + 8);
            v33 = (const __m128i *)&v37[*(_QWORD *)a1];
          }
        }
        v36 = (__m128i *)(v15 + 56 * v14);
        *v36 = _mm_loadu_si128(v33);
        v36[1] = _mm_loadu_si128(v33 + 1);
        v36[2] = _mm_loadu_si128(v33 + 2);
        v36[3].m128i_i64[0] = v33[3].m128i_i64[0];
        ++*(_DWORD *)(a1 + 8);
      }
      else
      {
        if ( v18 )
        {
          v19 = _mm_loadu_si128(&v92);
          v93.m128i_i32[2] = v11;
          v20 = _mm_loadu_si128(&v94);
          v21 = _mm_loadu_si128(&v93);
          *(_QWORD *)(v18 + 48) = v95;
          *(__m128i *)v18 = v19;
          *(__m128i *)(v18 + 16) = v21;
          *(__m128i *)(v18 + 32) = v20;
          v17 = *(_DWORD *)(a1 + 8);
        }
        *(_DWORD *)(a1 + 8) = v17 + 1;
      }
    }
  }
  return a1;
}
