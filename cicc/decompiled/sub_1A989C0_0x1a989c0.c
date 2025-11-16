// Function: sub_1A989C0
// Address: 0x1a989c0
//
void __fastcall sub_1A989C0(__int64 a1, __int64 *a2, char **a3, __int64 a4, __m128i **a5)
{
  __int64 v5; // rax
  __int64 v6; // rsi
  __int64 v7; // rax
  unsigned __int64 v8; // rdi
  unsigned __int8 v9; // al
  __int64 v10; // rax
  __int64 v11; // rdx
  unsigned __int64 v12; // r14
  __int64 v13; // rax
  __int64 v14; // rdx
  __int64 v15; // rbx
  __int64 v16; // rbx
  __int64 v17; // r12
  __int64 v18; // rbx
  __int64 v19; // rax
  unsigned int *v20; // rax
  __int64 v21; // rdx
  __int64 v22; // r12
  unsigned __int64 v23; // rbx
  _QWORD *v24; // rdi
  int v25; // r8d
  int v26; // r9d
  __int64 v27; // rax
  unsigned __int64 v28; // rbx
  __int64 v29; // r14
  __int64 v30; // r10
  __int64 v31; // r12
  __int16 v32; // dx
  __int64 v33; // rsi
  __int64 v34; // rsi
  __int64 v35; // rdx
  __int64 v36; // rsi
  __int64 v37; // rsi
  __int64 v38; // rdi
  int v39; // r14d
  __m128i v40; // xmm0
  unsigned __int64 v41; // rdi
  _QWORD *v42; // rbx
  __int64 v43; // rax
  __int64 *v44; // rax
  __int64 v45; // rax
  __int64 v46; // rdx
  __int64 v47; // rbx
  __int64 v48; // rbx
  __int64 v49; // r12
  __int64 v50; // rbx
  __int64 v51; // rax
  __int64 v52; // rax
  __int64 v53; // rax
  __int64 v54; // rdx
  __int64 v55; // rsi
  __int64 v56; // rsi
  __int64 v57; // rsi
  __int64 v58; // rdi
  int v59; // eax
  int v60; // r10d
  __m128i v61; // xmm1
  __int64 v62; // rsi
  unsigned __int8 v63; // al
  const char *v64; // rax
  unsigned __int64 v65; // rbx
  __int64 v66; // rdi
  __int64 v67; // r15
  __int64 v68; // rdx
  unsigned __int64 **v69; // rbx
  unsigned __int64 *v70; // rcx
  unsigned __int64 v71; // r12
  unsigned __int64 *v72; // rsi
  __int64 v73; // rbx
  _QWORD *v74; // rax
  __int64 *v75; // rax
  __int64 v76; // rax
  __int64 v77; // rax
  __int64 v78; // rdx
  unsigned __int8 v79; // al
  __int64 v80; // [rsp+8h] [rbp-1A8h]
  unsigned __int64 v81; // [rsp+18h] [rbp-198h]
  __int64 v82; // [rsp+18h] [rbp-198h]
  __int64 v83; // [rsp+18h] [rbp-198h]
  __int64 v85; // [rsp+28h] [rbp-188h]
  char *v86; // [rsp+30h] [rbp-180h]
  __int64 v87; // [rsp+38h] [rbp-178h]
  char *v88; // [rsp+40h] [rbp-170h]
  unsigned int v91; // [rsp+58h] [rbp-158h]
  unsigned int v92; // [rsp+5Ch] [rbp-154h]
  char *v93; // [rsp+60h] [rbp-150h]
  int v94; // [rsp+60h] [rbp-150h]
  __int64 v95; // [rsp+68h] [rbp-148h]
  char *v96; // [rsp+70h] [rbp-140h]
  __int64 v97; // [rsp+70h] [rbp-140h]
  __int64 v98; // [rsp+70h] [rbp-140h]
  __int64 v99; // [rsp+78h] [rbp-138h]
  __int64 v101; // [rsp+88h] [rbp-128h] BYREF
  __int64 v102; // [rsp+98h] [rbp-118h] BYREF
  const char *v103; // [rsp+A0h] [rbp-110h] BYREF
  __int64 v104; // [rsp+A8h] [rbp-108h]
  unsigned int v105; // [rsp+B0h] [rbp-100h] BYREF
  char v106; // [rsp+B4h] [rbp-FCh]
  __int64 v107; // [rsp+B8h] [rbp-F8h]
  char v108; // [rsp+C0h] [rbp-F0h]
  __int64 v109; // [rsp+D0h] [rbp-E0h] BYREF
  __int64 v110; // [rsp+D8h] [rbp-D8h]
  unsigned __int64 v111; // [rsp+E0h] [rbp-D0h]
  _QWORD *v112; // [rsp+E8h] [rbp-C8h]
  __int64 v113; // [rsp+F0h] [rbp-C0h]
  int v114; // [rsp+F8h] [rbp-B8h]
  __m128i v115[2]; // [rsp+100h] [rbp-B0h] BYREF
  __m128i v116; // [rsp+120h] [rbp-90h] BYREF
  unsigned __int64 v117; // [rsp+130h] [rbp-80h] BYREF
  _QWORD *v118; // [rsp+138h] [rbp-78h]
  __int64 v119; // [rsp+140h] [rbp-70h]
  int v120; // [rsp+148h] [rbp-68h]
  __m128i v121; // [rsp+150h] [rbp-60h]

  v101 = a1;
  v5 = sub_16498A0(a1 & 0xFFFFFFFFFFFFFFF8LL);
  v115[0] = 0u;
  v6 = *(_QWORD *)((a1 & 0xFFFFFFFFFFFFFFF8LL) + 48);
  v112 = (_QWORD *)v5;
  v114 = 0;
  v7 = *(_QWORD *)((a1 & 0xFFFFFFFFFFFFFFF8LL) + 40);
  v109 = 0;
  v110 = v7;
  v113 = 0;
  v111 = (a1 & 0xFFFFFFFFFFFFFFF8LL) + 24;
  v116.m128i_i64[0] = v6;
  if ( v6 )
  {
    sub_1623A60((__int64)&v116, v6, 2);
    v109 = v116.m128i_i64[0];
    if ( v116.m128i_i64[0] )
      sub_1623210((__int64)&v116, (unsigned __int8 *)v116.m128i_i64[0], (__int64)&v109);
  }
  v96 = *a3;
  v99 = *((unsigned int *)a3 + 2);
  v81 = sub_1389B50(&v101);
  v8 = v101 & 0xFFFFFFFFFFFFFFF8LL;
  v88 = (char *)((v101 & 0xFFFFFFFFFFFFFFF8LL) - 24LL * (*(_DWORD *)((v101 & 0xFFFFFFFFFFFFFFF8LL) + 20) & 0xFFFFFFF));
  v85 = 0xAAAAAAAAAAAAAAABLL * ((__int64)(v81 - (_QWORD)v88) >> 3);
  v9 = *(_BYTE *)((v101 & 0xFFFFFFFFFFFFFFF8LL) + 16);
  if ( v9 <= 0x17u )
  {
    v8 = 0;
  }
  else if ( v9 == 78 )
  {
    v8 |= 4u;
  }
  else if ( v9 != 29 )
  {
    v8 = 0;
  }
  v93 = 0;
  v86 = (char *)sub_1A95E00(v8);
  v10 = v101;
  v87 = v11;
  v95 = 0;
  v12 = v101 & 0xFFFFFFFFFFFFFFF8LL;
  if ( (v101 & 4) == 0 )
  {
    if ( *(char *)(v12 + 23) >= 0 )
      goto LABEL_53;
    v45 = sub_1648A40(v101 & 0xFFFFFFFFFFFFFFF8LL);
    v47 = v45 + v46;
    if ( *(char *)(v12 + 23) < 0 )
      v47 -= sub_1648A40(v12);
    v48 = v47 >> 4;
    if ( (_DWORD)v48 )
    {
      v49 = 0;
      v50 = 16LL * (unsigned int)v48;
      while ( 1 )
      {
        v51 = 0;
        if ( *(char *)(v12 + 23) < 0 )
          v51 = sub_1648A40(v12);
        v20 = (unsigned int *)(v49 + v51);
        if ( *(_DWORD *)(*(_QWORD *)v20 + 8LL) == 2 )
          goto LABEL_18;
        v49 += 16;
        if ( v50 == v49 )
          goto LABEL_52;
      }
    }
    goto LABEL_52;
  }
  if ( *(char *)(v12 + 23) < 0 )
  {
    v13 = sub_1648A40(v101 & 0xFFFFFFFFFFFFFFF8LL);
    v15 = v13 + v14;
    if ( *(char *)(v12 + 23) < 0 )
      v15 -= sub_1648A40(v12);
    v16 = v15 >> 4;
    if ( (_DWORD)v16 )
    {
      v17 = 0;
      v18 = 16LL * (unsigned int)v16;
      while ( 1 )
      {
        v19 = 0;
        if ( *(char *)(v12 + 23) < 0 )
          v19 = sub_1648A40(v12);
        v20 = (unsigned int *)(v17 + v19);
        if ( *(_DWORD *)(*(_QWORD *)v20 + 8LL) == 2 )
          break;
        v17 += 16;
        if ( v17 == v18 )
          goto LABEL_52;
      }
LABEL_18:
      v91 = 1;
      v21 = 24LL * v20[2];
      v93 = (char *)(v21 - 24LL * (*(_DWORD *)(v12 + 20) & 0xFFFFFFF) + v12);
      v95 = 0xAAAAAAAAAAAAAAABLL * ((24LL * v20[3] - v21) >> 3);
      v10 = v101;
      goto LABEL_19;
    }
LABEL_52:
    v10 = v101;
  }
LABEL_53:
  v91 = 0;
LABEL_19:
  sub_1642E70((__int64)&v105, *(_QWORD *)((v10 & 0xFFFFFFFFFFFFFFF8LL) + 56));
  v92 = 0;
  if ( v106 )
    v92 = v105;
  v22 = 2882400000LL;
  if ( v108 )
    v22 = v107;
  v23 = v101 & 0xFFFFFFFFFFFFFFF8LL;
  v24 = (_QWORD *)((v101 & 0xFFFFFFFFFFFFFFF8LL) + 56);
  if ( (v101 & 4) != 0 )
  {
    if ( !sub_15602A0(v24, -1, "deopt-lowering", 0xEu) )
    {
      v27 = *(_QWORD *)(v23 - 24);
      if ( *(_BYTE *)(v27 + 16) )
        goto LABEL_27;
      v116.m128i_i64[0] = *(_QWORD *)(v27 + 112);
      if ( !sub_15602A0(&v116, -1, "deopt-lowering", 0xEu) )
        goto LABEL_27;
    }
    v64 = *(const char **)(v23 + 56);
    v65 = v23 - 24;
    v103 = v64;
    if ( !sub_15602A0(&v103, -1, "deopt-lowering", 0xEu) )
      goto LABEL_104;
  }
  else
  {
    if ( !sub_15602A0(v24, -1, "deopt-lowering", 0xEu) )
    {
      v52 = *(_QWORD *)(v23 - 72);
      if ( *(_BYTE *)(v52 + 16) )
        goto LABEL_27;
      v116.m128i_i64[0] = *(_QWORD *)(v52 + 112);
      if ( !sub_15602A0(&v116, -1, "deopt-lowering", 0xEu) )
        goto LABEL_27;
    }
    v103 = *(const char **)(v23 + 56);
    if ( !sub_15602A0(&v103, -1, "deopt-lowering", 0xEu) )
    {
      v65 = v23 - 72;
LABEL_104:
      v66 = *(_QWORD *)v65;
      if ( *(_BYTE *)(*(_QWORD *)v65 + 16LL) )
        v66 = 0;
      v116.m128i_i64[0] = sub_1560340((_QWORD *)(v66 + 112), -1, "deopt-lowering", 0xEu);
      v53 = sub_155D8B0(v116.m128i_i64);
      goto LABEL_74;
    }
  }
  v116.m128i_i64[0] = sub_1560340(&v103, -1, "deopt-lowering", 0xEu);
  v53 = sub_155D8B0(v116.m128i_i64);
LABEL_74:
  if ( v54 == 7 && *(_DWORD *)v53 == 1702259052 && *(_WORD *)(v53 + 4) == 26925 && *(_BYTE *)(v53 + 6) == 110 )
    v91 |= 2u;
LABEL_27:
  v28 = v101 & 0xFFFFFFFFFFFFFFF8LL;
  v29 = (v101 >> 2) & 1;
  if ( ((v101 >> 2) & 1) != 0 )
  {
    v30 = *(_QWORD *)(v28 - 24);
    if ( *(_BYTE *)(v30 + 16) || *(_DWORD *)(v30 + 36) != 75 )
    {
      LOBYTE(v29) = 0;
      goto LABEL_31;
    }
  }
  else
  {
    v30 = *(_QWORD *)(v28 - 72);
    if ( *(_BYTE *)(v30 + 16) || *(_DWORD *)(v30 + 36) != 75 )
      goto LABEL_81;
  }
  v116.m128i_i64[0] = (__int64)&v117;
  v116.m128i_i64[1] = 0x800000000LL;
  if ( (char *)v81 == v88 )
  {
    v73 = 0;
    v72 = &v117;
  }
  else
  {
    v67 = v22;
    v68 = 0;
    v69 = (unsigned __int64 **)(v88 + 24);
    v70 = &v117;
    v71 = **(_QWORD **)v88;
    while ( 1 )
    {
      v70[v68] = v71;
      v68 = (unsigned int)++v116.m128i_i32[2];
      if ( (unsigned __int64 **)v81 == v69 )
        break;
      v71 = **v69;
      if ( v116.m128i_i32[3] <= (unsigned int)v68 )
      {
        v80 = v30;
        sub_16CD150((__int64)&v116, &v117, 0, 8, v25, v26);
        v68 = v116.m128i_u32[2];
        v30 = v80;
      }
      v70 = (unsigned __int64 *)v116.m128i_i64[0];
      v69 += 3;
    }
    v22 = v67;
    v72 = (unsigned __int64 *)v116.m128i_i64[0];
    v73 = (unsigned int)v68;
  }
  v82 = v30;
  v74 = (_QWORD *)sub_15E0530(v30);
  v75 = (__int64 *)sub_1643270(v74);
  v76 = sub_1644EA0(v75, v72, v73, 0);
  v77 = sub_1632190(*(_QWORD *)(v82 + 40), (__int64)"__llvm_deoptimize", 17, v76);
  v30 = v77;
  if ( (unsigned __int64 *)v116.m128i_i64[0] != &v117 )
  {
    v83 = v77;
    _libc_free(v116.m128i_u64[0]);
    v30 = v83;
  }
  v28 = v101 & 0xFFFFFFFFFFFFFFF8LL;
  v29 = (v101 >> 2) & 1;
  if ( ((v101 >> 2) & 1) != 0 )
  {
LABEL_31:
    v116.m128i_i64[0] = (__int64)"safepoint_token";
    LOWORD(v117) = 259;
    v31 = (__int64)sub_15E8C20(&v109, v22, v92, (__int64 *)v30, v91, (int)&v116, v88, v85, v93, v95, v86, v87, v96, v99);
    v32 = *(_WORD *)(v28 + 18) & 3 | *(_WORD *)(v31 + 18) & 0xFFFC;
    *(_WORD *)(v31 + 18) = v32;
    *(_WORD *)(v31 + 18) = v32 & 0x8000 | v32 & 3 | (4 * ((*(_WORD *)(v28 + 18) >> 2) & 0xDFFF));
    *(_QWORD *)(v31 + 56) = sub_1A958E0(*(_QWORD *)(v28 + 56));
    v33 = *(_QWORD *)(v28 + 32);
    if ( v33 == *(_QWORD *)(v28 + 40) + 40LL || !v33 )
      v34 = 0;
    else
      v34 = v33 - 24;
    sub_17050D0(&v109, v34);
    v35 = *(_QWORD *)(v28 + 32);
    if ( v35 == *(_QWORD *)(v28 + 40) + 40LL || !v35 )
      BUG();
    v36 = *(_QWORD *)(v35 + 24);
    v116.m128i_i64[0] = v36;
    if ( v36 )
    {
      sub_1623A60((__int64)&v116, v36, 2);
      v37 = v109;
      if ( !v109 )
        goto LABEL_39;
    }
    else
    {
      v37 = v109;
      if ( !v109 )
        goto LABEL_41;
    }
    sub_161E7C0((__int64)&v109, v37);
LABEL_39:
    v109 = v116.m128i_i64[0];
    if ( v116.m128i_i64[0] )
      sub_1623210((__int64)&v116, (unsigned __int8 *)v116.m128i_i64[0], (__int64)&v109);
    goto LABEL_41;
  }
  LOBYTE(v29) = 1;
LABEL_81:
  v116.m128i_i64[0] = (__int64)"statepoint_token";
  LOWORD(v117) = 259;
  v31 = (__int64)sub_15E8D00(
                   &v109,
                   v22,
                   v92,
                   (__int64 *)v30,
                   *(_QWORD *)(v28 - 48),
                   *(_QWORD *)(v28 - 24),
                   v91,
                   v88,
                   v85,
                   v93,
                   v95,
                   v86,
                   v87,
                   v96,
                   v99,
                   (__int64)&v116);
  *(_WORD *)(v31 + 18) = *(_WORD *)(v31 + 18) & 0x8000
                       | *(_WORD *)(v31 + 18) & 3
                       | (4 * ((*(_WORD *)(v28 + 18) >> 2) & 0xDFFF));
  *(_QWORD *)(v31 + 56) = sub_1A958E0(*(_QWORD *)(v28 + 56));
  v97 = *(_QWORD *)(v28 - 24);
  v55 = sub_157EE30(v97);
  if ( v55 )
    v55 -= 24;
  sub_17050D0(&v109, v55);
  v56 = *(_QWORD *)(v28 + 48);
  v116.m128i_i64[0] = v56;
  if ( v56 )
  {
    sub_1623A60((__int64)&v116, v56, 2);
    v57 = v109;
    if ( !v109 )
      goto LABEL_86;
    goto LABEL_85;
  }
  v57 = v109;
  if ( v109 )
  {
LABEL_85:
    sub_161E7C0((__int64)&v109, v57);
LABEL_86:
    v109 = v116.m128i_i64[0];
    if ( v116.m128i_i64[0] )
      sub_1623210((__int64)&v116, (unsigned __int8 *)v116.m128i_i64[0], (__int64)&v109);
  }
  v98 = sub_157F7B0(v97);
  *(_QWORD *)(a4 + 120) = v98;
  v58 = 0;
  if ( sub_1642D30(v31) )
  {
    v79 = *(_BYTE *)(v31 + 16);
    v58 = 0;
    if ( v79 > 0x17u )
    {
      if ( v79 == 78 )
      {
        v58 = v31 | 4;
      }
      else if ( v79 == 29 )
      {
        v58 = v31 & 0xFFFFFFFFFFFFFFFBLL;
      }
    }
  }
  v59 = sub_1A95F60(v58);
  v60 = v59;
  v116.m128i_i64[0] = v109;
  if ( v109 )
  {
    v94 = v59;
    sub_1623A60((__int64)&v116, v109, 2);
    v60 = v94;
  }
  v61 = _mm_loadu_si128(v115);
  v116.m128i_i64[1] = v110;
  v121 = v61;
  v117 = v111;
  v118 = v112;
  v119 = v113;
  v120 = v114;
  sub_1A980B0(*a3, *((unsigned int *)a3 + 2), v60, *a2, v98, (__int64)&v116);
  if ( v116.m128i_i64[0] )
    sub_161E7C0((__int64)&v116, v116.m128i_i64[0]);
  v62 = sub_157EE30(*(_QWORD *)(v28 - 48));
  if ( v62 )
    v62 -= 24;
  sub_17050D0(&v109, v62);
LABEL_41:
  if ( (_BYTE)v29 )
  {
    v116.m128i_i64[1] = 0;
    LOBYTE(v117) = 1;
    v116.m128i_i64[0] = v101 & 0xFFFFFFFFFFFFFFF8LL;
    sub_1A94C30(a5, &v116);
  }
  else
  {
    v116.m128i_i64[0] = (__int64)"statepoint_token";
    LOWORD(v117) = 259;
    sub_164B780(v31, v116.m128i_i64);
    v41 = v101 & 0xFFFFFFFFFFFFFFF8LL;
    if ( *(_BYTE *)(*(_QWORD *)(v101 & 0xFFFFFFFFFFFFFFF8LL) + 8LL) && *(_QWORD *)(v41 + 8) )
    {
      if ( (*(_BYTE *)(v41 + 23) & 0x20) != 0 )
      {
        v103 = sub_1649960(v41);
        v104 = v78;
        v41 = v101 & 0xFFFFFFFFFFFFFFF8LL;
      }
      else
      {
        v104 = 0;
        v103 = byte_3F871B3;
      }
      LOWORD(v117) = 261;
      v116.m128i_i64[0] = (__int64)&v103;
      v42 = sub_15E8360(&v109, v31, *(_QWORD *)v41, (int)&v116);
      v102 = *(_QWORD *)((v101 & 0xFFFFFFFFFFFFFFF8LL) + 56);
      v43 = sub_1560240(&v102);
      sub_1563030(&v116, v43);
      v44 = (__int64 *)sub_16498A0((__int64)v42);
      v42[7] = sub_1560CD0(v44, 0, &v116);
      sub_1A95860(v118);
      LOBYTE(v117) = 0;
      v116.m128i_i64[1] = (__int64)v42;
      v116.m128i_i64[0] = v101 & 0xFFFFFFFFFFFFFFF8LL;
      sub_1A94C30(a5, &v116);
    }
    else
    {
      v116.m128i_i64[0] = v101 & 0xFFFFFFFFFFFFFFF8LL;
      v116.m128i_i64[1] = 0;
      LOBYTE(v117) = 0;
      sub_1A94C30(a5, &v116);
    }
  }
  *(_QWORD *)(a4 + 112) = v31;
  v38 = 0;
  if ( sub_1642D30(v31) )
  {
    v63 = *(_BYTE *)(v31 + 16);
    if ( v63 > 0x17u )
    {
      if ( v63 == 78 )
      {
        v38 = v31 | 4;
      }
      else if ( v63 == 29 )
      {
        v38 = v31 & 0xFFFFFFFFFFFFFFFBLL;
      }
    }
  }
  v39 = sub_1A95F60(v38);
  v116.m128i_i64[0] = v109;
  if ( v109 )
    sub_1623A60((__int64)&v116, v109, 2);
  v40 = _mm_loadu_si128(v115);
  v116.m128i_i64[1] = v110;
  v121 = v40;
  v117 = v111;
  v118 = v112;
  v119 = v113;
  v120 = v114;
  sub_1A980B0(*a3, *((unsigned int *)a3 + 2), v39, *a2, v31, (__int64)&v116);
  if ( v116.m128i_i64[0] )
    sub_161E7C0((__int64)&v116, v116.m128i_i64[0]);
  if ( v109 )
    sub_161E7C0((__int64)&v109, v109);
}
