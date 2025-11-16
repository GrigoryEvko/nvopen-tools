// Function: sub_16FC3B0
// Address: 0x16fc3b0
//
__int64 __fastcall sub_16FC3B0(__int64 a1, unsigned __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v6; // rax
  int v7; // edx
  __m128i v8; // xmm0
  _BYTE *v9; // rsi
  __int64 v10; // rdx
  __int64 v11; // rdx
  __int64 v12; // rcx
  __int64 v13; // r8
  int v14; // eax
  __int64 v15; // r15
  __int64 v16; // r14
  __int64 v17; // r14
  __int64 v18; // r9
  _QWORD *v20; // rsi
  __int64 v21; // r8
  _QWORD *v22; // rcx
  __m128i v23; // xmm3
  _BYTE *v24; // rdi
  size_t v25; // rdx
  __int64 v26; // rax
  _QWORD *v27; // rsi
  __int64 v28; // r8
  _QWORD *v29; // rcx
  __m128i v30; // xmm1
  _BYTE *v31; // rdi
  size_t v32; // rdx
  __int64 v33; // rax
  __int64 v34; // r9
  __int64 v35; // r8
  __int64 v36; // rcx
  __int128 v37; // kr00_16
  __int64 v38; // r9
  __int64 v39; // r8
  __int64 v40; // rcx
  __int128 v41; // kr10_16
  __int64 v42; // r9
  __int64 v43; // r8
  __int64 v44; // rcx
  __int128 v45; // kr20_16
  __int64 v46; // r9
  __int64 v47; // r8
  __int64 v48; // rcx
  __int128 v49; // kr30_16
  __int64 v50; // r9
  __int64 v51; // r8
  __int64 v52; // rcx
  __int128 v53; // kr40_16
  __int64 v54; // r9
  __int64 v55; // r8
  __int64 v56; // rcx
  __int128 v57; // kr50_16
  __int64 v58; // r9
  __int64 v59; // r8
  __int64 v60; // rcx
  __m128i v61; // kr60_16
  __int64 v62; // r9
  __int64 v63; // r14
  __int64 *v64; // r10
  void *v65; // r15
  __int64 v66; // r9
  __int64 v67; // r8
  __int64 v68; // rcx
  __int64 v69; // r9
  const char *v70; // rax
  __int128 v71; // [rsp-10h] [rbp-180h]
  __int64 v72; // [rsp+0h] [rbp-170h]
  __int64 v73; // [rsp+8h] [rbp-168h]
  __int64 v74; // [rsp+10h] [rbp-160h]
  __int64 v75; // [rsp+10h] [rbp-160h]
  __int64 v76; // [rsp+18h] [rbp-158h]
  __int64 v77; // [rsp+18h] [rbp-158h]
  void *v78; // [rsp+20h] [rbp-150h]
  void *v79; // [rsp+20h] [rbp-150h]
  void *v80; // [rsp+20h] [rbp-150h]
  void *v81; // [rsp+20h] [rbp-150h]
  void *v82; // [rsp+20h] [rbp-150h]
  void *v83; // [rsp+20h] [rbp-150h]
  void *v84; // [rsp+20h] [rbp-150h]
  void *v85; // [rsp+20h] [rbp-150h]
  __int64 v86; // [rsp+28h] [rbp-148h]
  __int64 v87; // [rsp+28h] [rbp-148h]
  __int64 v88; // [rsp+28h] [rbp-148h]
  __int64 v89; // [rsp+28h] [rbp-148h]
  __int64 v90; // [rsp+28h] [rbp-148h]
  __int64 v91; // [rsp+28h] [rbp-148h]
  __int128 v92; // [rsp+28h] [rbp-148h]
  size_t v93; // [rsp+28h] [rbp-148h]
  __int64 v94; // [rsp+28h] [rbp-148h]
  __int64 v95; // [rsp+30h] [rbp-140h]
  __int64 v96; // [rsp+30h] [rbp-140h]
  int v97; // [rsp+40h] [rbp-130h] BYREF
  __m128i v98; // [rsp+48h] [rbp-128h]
  _QWORD *v99; // [rsp+58h] [rbp-118h] BYREF
  __int64 v100; // [rsp+60h] [rbp-110h]
  _QWORD v101[3]; // [rsp+68h] [rbp-108h] BYREF
  int v102; // [rsp+80h] [rbp-F0h]
  __m128i v103; // [rsp+88h] [rbp-E8h]
  void *dest; // [rsp+98h] [rbp-D8h]
  size_t v105; // [rsp+A0h] [rbp-D0h]
  _QWORD v106[3]; // [rsp+A8h] [rbp-C8h] BYREF
  int v107; // [rsp+C0h] [rbp-B0h]
  __int128 v108; // [rsp+C8h] [rbp-A8h]
  void *v109; // [rsp+D8h] [rbp-98h]
  size_t v110; // [rsp+E0h] [rbp-90h]
  _QWORD v111[3]; // [rsp+E8h] [rbp-88h] BYREF
  const char *v112; // [rsp+100h] [rbp-70h] BYREF
  __m128i v113; // [rsp+108h] [rbp-68h] BYREF
  _QWORD *v114; // [rsp+118h] [rbp-58h]
  size_t n; // [rsp+120h] [rbp-50h]
  _QWORD src[9]; // [rsp+128h] [rbp-48h] BYREF

  v6 = sub_16FC330((__int64 **)a1, a2, a3, a4, a5);
  v7 = *(_DWORD *)v6;
  v8 = _mm_loadu_si128((const __m128i *)(v6 + 8));
  v99 = v101;
  v9 = *(_BYTE **)(v6 + 24);
  v97 = v7;
  v10 = *(_QWORD *)(v6 + 32);
  v98 = v8;
  sub_16F6740((__int64 *)&v99, v9, (__int64)&v9[v10]);
  dest = v106;
  v102 = 0;
  v103 = 0u;
  v105 = 0;
  LOBYTE(v106[0]) = 0;
  v107 = 0;
  v108 = 0u;
  v109 = v111;
  v110 = 0;
  LOBYTE(v111[0]) = 0;
  while ( 1 )
  {
    v14 = v97;
    if ( v97 != 21 )
    {
      while ( 1 )
      {
        if ( v14 != 22 )
        {
          if ( v14 == 20 )
          {
            sub_16FC210((__int64)&v112, (unsigned __int64 **)a1, v11, v12, v13);
            if ( v114 != src )
              j_j___libc_free_0(v114, src[0] + 1LL);
            v15 = v98.m128i_i64[1];
            v16 = 0;
            if ( v98.m128i_i64[1] )
            {
              v15 = v98.m128i_i64[1] - 1;
              v16 = 1;
            }
            v95 = v16 + v98.m128i_i64[0];
            v17 = sub_145CBF0((__int64 *)(a1 + 8), 88, 16);
            sub_16FC350(v17, 6u, *(_QWORD *)a1 + 8LL, 0, 0, v18, 0);
            *(_QWORD *)(v17 + 80) = v15;
            *(_QWORD *)(v17 + 72) = v95;
            *(_QWORD *)v17 = &unk_49EFE78;
          }
          else
          {
            switch ( v14 )
            {
              case 0:
                v17 = 0;
                break;
              case 7:
                v35 = v103.m128i_i64[1];
                v36 = 0;
                if ( v103.m128i_i64[1] )
                {
                  v35 = v103.m128i_i64[1] - 1;
                  v36 = 1;
                }
                v78 = (void *)v35;
                v86 = v103.m128i_i64[0] + v36;
                v37 = v108;
                v17 = sub_145CBF0((__int64 *)(a1 + 8), 88, 16);
                sub_16FC350(v17, 5u, *(_QWORD *)a1 + 8LL, v86, (__int64)v78, v38, v37);
                *(_BYTE *)(v17 + 78) = 1;
                *(_DWORD *)(v17 + 72) = 2;
                *(_WORD *)(v17 + 76) = 1;
                *(_QWORD *)v17 = &unk_49EFE58;
                *(_QWORD *)(v17 + 80) = 0;
                break;
              case 9:
                sub_16FC210((__int64)&v112, (unsigned __int64 **)a1, v11, v12, v13);
                if ( v114 != src )
                  j_j___libc_free_0(v114, src[0] + 1LL);
                v39 = v103.m128i_i64[1];
                v40 = 0;
                if ( v103.m128i_i64[1] )
                {
                  v39 = v103.m128i_i64[1] - 1;
                  v40 = 1;
                }
                v79 = (void *)v39;
                v87 = v103.m128i_i64[0] + v40;
                v41 = v108;
                v17 = sub_145CBF0((__int64 *)(a1 + 8), 88, 16);
                sub_16FC350(v17, 5u, *(_QWORD *)a1 + 8LL, v87, (__int64)v79, v42, v41);
                *(_BYTE *)(v17 + 78) = 1;
                *(_DWORD *)(v17 + 72) = 0;
                *(_WORD *)(v17 + 76) = 1;
                *(_QWORD *)v17 = &unk_49EFE58;
                *(_QWORD *)(v17 + 80) = 0;
                break;
              case 10:
                sub_16FC210((__int64)&v112, (unsigned __int64 **)a1, v11, v12, v13);
                if ( v114 != src )
                  j_j___libc_free_0(v114, src[0] + 1LL);
                v43 = v103.m128i_i64[1];
                v44 = 0;
                if ( v103.m128i_i64[1] )
                {
                  v43 = v103.m128i_i64[1] - 1;
                  v44 = 1;
                }
                v80 = (void *)v43;
                v88 = v103.m128i_i64[0] + v44;
                v45 = v108;
                v17 = sub_145CBF0((__int64 *)(a1 + 8), 88, 16);
                sub_16FC350(v17, 4u, *(_QWORD *)a1 + 8LL, v88, (__int64)v80, v46, v45);
                *(_DWORD *)(v17 + 72) = 0;
                *(_WORD *)(v17 + 76) = 1;
                *(_QWORD *)(v17 + 80) = 0;
                *(_QWORD *)v17 = &unk_49EFE38;
                break;
              case 12:
                sub_16FC210((__int64)&v112, (unsigned __int64 **)a1, v11, v12, v13);
                if ( v114 != src )
                  j_j___libc_free_0(v114, src[0] + 1LL);
                v47 = v103.m128i_i64[1];
                v48 = 0;
                if ( v103.m128i_i64[1] )
                {
                  v47 = v103.m128i_i64[1] - 1;
                  v48 = 1;
                }
                v81 = (void *)v47;
                v89 = v103.m128i_i64[0] + v48;
                v49 = v108;
                v17 = sub_145CBF0((__int64 *)(a1 + 8), 88, 16);
                sub_16FC350(v17, 5u, *(_QWORD *)a1 + 8LL, v89, (__int64)v81, v50, v49);
                *(_BYTE *)(v17 + 78) = 1;
                *(_DWORD *)(v17 + 72) = 1;
                *(_WORD *)(v17 + 76) = 1;
                *(_QWORD *)v17 = &unk_49EFE58;
                *(_QWORD *)(v17 + 80) = 0;
                break;
              case 14:
                sub_16FC210((__int64)&v112, (unsigned __int64 **)a1, v11, v12, v13);
                if ( v114 != src )
                  j_j___libc_free_0(v114, src[0] + 1LL);
                v51 = v103.m128i_i64[1];
                v52 = 0;
                if ( v103.m128i_i64[1] )
                {
                  v51 = v103.m128i_i64[1] - 1;
                  v52 = 1;
                }
                v82 = (void *)v51;
                v90 = v103.m128i_i64[0] + v52;
                v53 = v108;
                v17 = sub_145CBF0((__int64 *)(a1 + 8), 88, 16);
                sub_16FC350(v17, 4u, *(_QWORD *)a1 + 8LL, v90, (__int64)v82, v54, v53);
                *(_DWORD *)(v17 + 72) = 1;
                *(_WORD *)(v17 + 76) = 1;
                *(_QWORD *)(v17 + 80) = 0;
                *(_QWORD *)v17 = &unk_49EFE38;
                break;
              case 16:
                v55 = v103.m128i_i64[1];
                v56 = 0;
                if ( v103.m128i_i64[1] )
                {
                  v55 = v103.m128i_i64[1] - 1;
                  v56 = 1;
                }
                v83 = (void *)v55;
                v91 = v103.m128i_i64[0] + v56;
                v57 = v108;
                v17 = sub_145CBF0((__int64 *)(a1 + 8), 88, 16);
                sub_16FC350(v17, 4u, *(_QWORD *)a1 + 8LL, v91, (__int64)v83, v58, v57);
                *(_DWORD *)(v17 + 72) = 2;
                *(_WORD *)(v17 + 76) = 1;
                *(_QWORD *)(v17 + 80) = 0;
                *(_QWORD *)v17 = &unk_49EFE38;
                break;
              case 18:
                sub_16FC210((__int64)&v112, (unsigned __int64 **)a1, v11, v12, v13);
                if ( v114 != src )
                  j_j___libc_free_0(v114, src[0] + 1LL);
                v59 = v103.m128i_i64[1];
                v60 = 0;
                if ( v103.m128i_i64[1] )
                {
                  v59 = v103.m128i_i64[1] - 1;
                  v60 = 1;
                }
                v76 = v103.m128i_i64[0] + v60;
                v74 = v59;
                v61 = v98;
                v92 = v108;
                v17 = sub_145CBF0((__int64 *)(a1 + 8), 88, 16);
                sub_16FC350(v17, 1u, *(_QWORD *)a1 + 8LL, v76, v74, v62, v92);
                *(_QWORD *)(v17 + 72) = v98.m128i_i64[0];
                *(_QWORD *)(v17 + 16) = v61.m128i_i64[0];
                *(_QWORD *)(v17 + 80) = v61.m128i_i64[1];
                *(_QWORD *)v17 = &unk_49EFDD8;
                *(_QWORD *)(v17 + 24) = v61.m128i_i64[0] + v61.m128i_i64[1];
                break;
              case 19:
                sub_16FC210((__int64)&v112, (unsigned __int64 **)a1, v11, v12, v13);
                if ( v114 != src )
                  j_j___libc_free_0(v114, src[0] + 1LL);
                v63 = v100;
                v64 = (__int64 *)(a1 + 8);
                v65 = 0;
                v66 = v100 + 1;
                if ( v100 != -1 )
                {
                  v84 = v99;
                  v93 = v100 + 1;
                  v65 = (void *)sub_145CBF0((__int64 *)(a1 + 8), v100 + 1, 1);
                  memmove(v65, v84, v93);
                  v64 = (__int64 *)(a1 + 8);
                  v66 = v63;
                }
                v67 = v103.m128i_i64[1];
                v68 = 0;
                if ( v103.m128i_i64[1] )
                {
                  v67 = v103.m128i_i64[1] - 1;
                  v68 = 1;
                }
                v75 = v103.m128i_i64[0] + v68;
                v94 = v98.m128i_i64[1];
                v73 = v66;
                v77 = v98.m128i_i64[0];
                v72 = v67;
                v96 = *((_QWORD *)&v108 + 1);
                v85 = (void *)v108;
                v17 = sub_145CBF0(v64, 88, 16);
                *((_QWORD *)&v71 + 1) = v96;
                *(_QWORD *)&v71 = v85;
                sub_16FC350(v17, 2u, *(_QWORD *)a1 + 8LL, v75, v72, v69, v71);
                *(_QWORD *)(v17 + 72) = v65;
                *(_QWORD *)(v17 + 16) = v77;
                *(_QWORD *)v17 = &unk_49EFDF8;
                *(_QWORD *)(v17 + 80) = v73;
                *(_QWORD *)(v17 + 24) = v77 + v94;
                break;
              default:
                v17 = sub_145CBF0((__int64 *)(a1 + 8), 72, 16);
                sub_16FC350(v17, 0, *(_QWORD *)a1 + 8LL, 0, 0, v34, 0);
                *(_QWORD *)v17 = &unk_49EFDB8;
                break;
            }
          }
          goto LABEL_11;
        }
        if ( v107 == 22 )
        {
          v113.m128i_i8[9] = 1;
          v70 = "Already encountered a tag for this node!";
          goto LABEL_89;
        }
        v20 = (_QWORD *)a1;
        sub_16FC210((__int64)&v112, (unsigned __int64 **)a1, v11, v12, v13);
        v22 = src;
        v23 = _mm_loadu_si128(&v113);
        v24 = v109;
        v107 = (int)v112;
        v108 = (__int128)v23;
        if ( v114 == src )
        {
          v25 = n;
          if ( n )
          {
            if ( n == 1 )
            {
              *(_BYTE *)v109 = src[0];
              v25 = n;
              v24 = v109;
            }
            else
            {
              v20 = src;
              memcpy(v109, src, n);
              v25 = n;
              v24 = v109;
              v22 = src;
            }
          }
          v110 = v25;
          v24[v25] = 0;
          v24 = v114;
          goto LABEL_23;
        }
        v25 = src[0];
        v20 = (_QWORD *)n;
        if ( v109 == v111 )
          break;
        v21 = v111[0];
        v109 = v114;
        v110 = n;
        v111[0] = src[0];
        if ( !v24 )
          goto LABEL_45;
        v114 = v24;
        src[0] = v21;
LABEL_23:
        n = 0;
        *v24 = 0;
        if ( v114 != src )
        {
          v20 = (_QWORD *)(src[0] + 1LL);
          j_j___libc_free_0(v114, src[0] + 1LL);
        }
        v26 = sub_16FC330((__int64 **)a1, (unsigned __int64)v20, v25, (__int64)v22, v21);
        v97 = *(_DWORD *)v26;
        v98 = _mm_loadu_si128((const __m128i *)(v26 + 8));
        sub_2240AE0(&v99, v26 + 24);
        v14 = v97;
        if ( v97 == 21 )
          goto LABEL_26;
      }
      v109 = v114;
      v110 = n;
      v111[0] = src[0];
LABEL_45:
      v114 = src;
      v22 = src;
      v24 = src;
      goto LABEL_23;
    }
LABEL_26:
    if ( v102 == 21 )
      break;
    v27 = (_QWORD *)a1;
    sub_16FC210((__int64)&v112, (unsigned __int64 **)a1, v11, v12, v13);
    v29 = src;
    v30 = _mm_loadu_si128(&v113);
    v31 = dest;
    v102 = (int)v112;
    v103 = v30;
    if ( v114 == src )
    {
      v32 = n;
      if ( n )
      {
        if ( n == 1 )
        {
          *(_BYTE *)dest = src[0];
          v32 = n;
          v31 = dest;
        }
        else
        {
          v27 = src;
          memcpy(dest, src, n);
          v32 = n;
          v31 = dest;
          v29 = src;
        }
      }
      v105 = v32;
      v31[v32] = 0;
      v31 = v114;
    }
    else
    {
      v32 = src[0];
      v27 = (_QWORD *)n;
      if ( dest == v106 )
      {
        dest = v114;
        v105 = n;
        v106[0] = src[0];
      }
      else
      {
        v28 = v106[0];
        dest = v114;
        v105 = n;
        v106[0] = src[0];
        if ( v31 )
        {
          v114 = v31;
          src[0] = v28;
          goto LABEL_31;
        }
      }
      v114 = src;
      v29 = src;
      v31 = src;
    }
LABEL_31:
    n = 0;
    *v31 = 0;
    if ( v114 != src )
    {
      v27 = (_QWORD *)(src[0] + 1LL);
      j_j___libc_free_0(v114, src[0] + 1LL);
    }
    v33 = sub_16FC330((__int64 **)a1, (unsigned __int64)v27, v32, (__int64)v29, v28);
    v97 = *(_DWORD *)v33;
    v98 = _mm_loadu_si128((const __m128i *)(v33 + 8));
    sub_2240AE0(&v99, v33 + 24);
  }
  v113.m128i_i8[9] = 1;
  v70 = "Already encountered an anchor for this node!";
LABEL_89:
  v112 = v70;
  v17 = 0;
  v113.m128i_i8[8] = 3;
  sub_16F82E0((__int64 **)a1, (__int64)&v112, (__int64)&v97, v12, v13);
LABEL_11:
  if ( v109 != v111 )
    j_j___libc_free_0(v109, v111[0] + 1LL);
  if ( dest != v106 )
    j_j___libc_free_0(dest, v106[0] + 1LL);
  if ( v99 != v101 )
    j_j___libc_free_0(v99, v101[0] + 1LL);
  return v17;
}
