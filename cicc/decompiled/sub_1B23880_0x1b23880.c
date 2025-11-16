// Function: sub_1B23880
// Address: 0x1b23880
//
_QWORD *__fastcall sub_1B23880(
        __int64 *a1,
        const __m128i *a2,
        __int64 a3,
        __int64 a4,
        __int64 *a5,
        __int64 a6,
        double a7,
        double a8,
        double a9,
        __int64 a10,
        __int64 a11,
        __int64 **a12)
{
  unsigned __int64 v12; // rax
  __int64 v13; // r13
  __int64 v14; // r15
  unsigned __int64 *v15; // r15
  unsigned __int64 v16; // rcx
  __int64 v17; // rax
  __int64 v18; // r15
  unsigned int v19; // r13d
  __int64 v20; // rdx
  __int64 v21; // rcx
  __int64 v22; // rdi
  _QWORD *v23; // r13
  __int64 v24; // r15
  _QWORD **v25; // rax
  __int64 *v26; // rax
  __int64 v27; // rsi
  __int64 v28; // rbx
  _QWORD *v29; // rdi
  __int64 i; // r15
  __int64 v31; // r13
  __int64 v32; // rax
  unsigned int v33; // edx
  __int64 *v34; // rbx
  __int64 v35; // rbx
  unsigned int v36; // edx
  __int64 *v37; // rax
  __int64 v38; // rax
  __int64 v39; // r12
  __int64 v40; // rbx
  unsigned int v41; // r8d
  unsigned int v42; // esi
  __int64 v43; // rdx
  __int64 v44; // rax
  char v45; // cl
  __int64 v46; // r10
  __int64 v47; // rax
  unsigned int v48; // r9d
  __int64 v49; // rdx
  __int64 v50; // rdi
  unsigned int v51; // esi
  __int64 *v52; // rax
  __int64 v53; // rax
  unsigned int v54; // edi
  __int64 *v55; // rsi
  __int64 v56; // rsi
  int v57; // ecx
  __int64 v58; // rax
  const __m128i *v59; // r13
  __int64 v60; // rax
  __m128i *v61; // rdx
  const __m128i *v62; // rax
  __m128i v63; // xmm0
  __m128i *v64; // rdx
  const __m128i *v65; // rax
  __m128i v66; // xmm1
  __int64 v67; // r15
  int v68; // eax
  __int64 *v69; // rax
  int v70; // r12d
  char *v71; // r8
  __int64 *v72; // rsi
  __int64 v73; // rdi
  unsigned int v74; // edx
  __int64 *v75; // rax
  __int64 v76; // rbx
  unsigned int v77; // edx
  __int64 *v78; // rax
  __int64 v79; // rbx
  __int64 v80; // rax
  __int64 v81; // rax
  __int64 v82; // rcx
  __int64 v83; // r9
  __int64 *v84; // rdx
  __int64 v85; // rax
  __int64 v86; // rbx
  _QWORD *v87; // rax
  _QWORD *v88; // rbx
  __int64 v89; // r13
  _QWORD **v90; // rax
  __int64 *v91; // rax
  __int64 v92; // rsi
  __int64 v93; // rax
  unsigned __int64 *v94; // r15
  __int64 v95; // r12
  unsigned __int64 v96; // rcx
  __int64 v97; // rax
  __int64 v98; // rax
  _QWORD *v99; // rdi
  __int64 v101; // r15
  _QWORD **v102; // rax
  __int64 *v103; // rax
  __int64 v104; // rsi
  __int64 *v105; // r13
  __int64 v106; // rdx
  __int64 v107; // r12
  __int64 v108; // rbx
  _QWORD **v109; // rax
  __int64 *v110; // rax
  __int64 v111; // rsi
  __int64 v112; // r15
  _QWORD **v113; // rax
  __int64 *v114; // rax
  __int64 v115; // rsi
  _QWORD *v116; // [rsp+8h] [rbp-C8h]
  __int64 v117; // [rsp+10h] [rbp-C0h]
  __int64 v118; // [rsp+18h] [rbp-B8h]
  signed __int64 v119; // [rsp+20h] [rbp-B0h]
  unsigned __int64 v120; // [rsp+28h] [rbp-A8h]
  unsigned __int64 v121; // [rsp+38h] [rbp-98h]
  int v122; // [rsp+40h] [rbp-90h]
  int v123; // [rsp+48h] [rbp-88h]
  __int64 v124; // [rsp+48h] [rbp-88h]
  _QWORD *v125; // [rsp+50h] [rbp-80h]
  __int64 v126; // [rsp+50h] [rbp-80h]
  _QWORD *v127; // [rsp+50h] [rbp-80h]
  _QWORD *v128; // [rsp+50h] [rbp-80h]
  _QWORD *v130; // [rsp+58h] [rbp-78h]
  _QWORD *v131; // [rsp+60h] [rbp-70h]
  __int64 v133; // [rsp+68h] [rbp-68h]
  const char *v134; // [rsp+70h] [rbp-60h] BYREF
  __int64 v135; // [rsp+78h] [rbp-58h]
  const char *v136; // [rsp+80h] [rbp-50h] BYREF
  const char *v137; // [rsp+88h] [rbp-48h]
  __int16 v138; // [rsp+90h] [rbp-40h]

  v122 = a4;
  v12 = 0xAAAAAAAAAAAAAAABLL * (((char *)a2 - (char *)a1) >> 3);
  v123 = a3;
  if ( (_DWORD)v12 == 1 )
  {
    if ( *a1 == a3 && a1[1] == a4 )
    {
      if ( a3 && a4 )
      {
        v51 = *(_DWORD *)(a4 + 32);
        v52 = *(__int64 **)(a4 + 24);
        if ( v51 > 0x40 )
          v53 = *v52;
        else
          v53 = (__int64)((_QWORD)v52 << (64 - (unsigned __int8)v51)) >> (64 - (unsigned __int8)v51);
        v54 = *(_DWORD *)(a3 + 32);
        v55 = *(__int64 **)(a3 + 24);
        if ( v54 > 0x40 )
          v56 = *v55;
        else
          v56 = (__int64)((_QWORD)v55 << (64 - (unsigned __int8)v54)) >> (64 - (unsigned __int8)v54);
        v57 = v53 - v56;
      }
      else
      {
        v57 = 0;
      }
      sub_1B23670(a1[2], a10, a6, v57);
      return (_QWORD *)a1[2];
    }
    v13 = *(_QWORD *)(a10 + 56);
    v136 = "LeafBlock";
    v138 = 259;
    v14 = sub_16498A0((__int64)a5);
    v131 = (_QWORD *)sub_22077B0(64);
    if ( v131 )
      sub_157FB60(v131, v14, (__int64)&v136, 0, 0);
    v15 = *(unsigned __int64 **)(a10 + 32);
    sub_15E01D0(v13 + 72, (__int64)v131);
    v16 = *v15;
    v17 = v131[3];
    v131[4] = v15;
    v16 &= 0xFFFFFFFFFFFFFFF8LL;
    v131[3] = v16 | v17 & 7;
    *(_QWORD *)(v16 + 8) = v131 + 3;
    *v15 = *v15 & 7 | (unsigned __int64)(v131 + 3);
    v18 = *a1;
    if ( *a1 == a1[1] )
    {
      v136 = "SwitchLeaf";
      v138 = 259;
      v23 = sub_1648A60(56, 2u);
      if ( v23 )
      {
        v112 = *a1;
        v113 = (_QWORD **)*a5;
        if ( *(_BYTE *)(*a5 + 8) == 16 )
        {
          v128 = v113[4];
          v114 = (__int64 *)sub_1643320(*v113);
          v115 = (__int64)sub_16463B0(v114, (unsigned int)v128);
        }
        else
        {
          v115 = sub_1643320(*v113);
        }
        sub_15FED60((__int64)v23, v115, 51, 32, (__int64)a5, v112, (__int64)&v136, (__int64)v131);
      }
      goto LABEL_13;
    }
    v19 = *(_DWORD *)(v18 + 32);
    v20 = *(_QWORD *)(v18 + 24);
    v21 = v19 - 1;
    if ( v19 <= 0x40 )
    {
      if ( v20 == 1LL << ((unsigned __int8)v19 - 1) )
        goto LABEL_104;
      if ( !v20 )
        goto LABEL_9;
    }
    else
    {
      v22 = v18 + 24;
      if ( (*(_QWORD *)(v20 + 8LL * ((v19 - 1) >> 6)) & (1LL << ((unsigned __int8)v19 - 1))) != 0
        && (unsigned int)sub_16A58A0(v22) == (_DWORD)v21 )
      {
LABEL_104:
        v136 = "SwitchLeaf";
        v138 = 259;
        v23 = sub_1648A60(56, 2u);
        if ( v23 )
        {
          v101 = a1[1];
          v102 = (_QWORD **)*a5;
          if ( *(_BYTE *)(*a5 + 8) == 16 )
          {
            v127 = v102[4];
            v103 = (__int64 *)sub_1643320(*v102);
            v104 = (__int64)sub_16463B0(v103, (unsigned int)v127);
          }
          else
          {
            v104 = sub_1643320(*v102);
          }
          sub_15FED60((__int64)v23, v104, 51, 41, (__int64)a5, v101, (__int64)&v136, (__int64)v131);
        }
LABEL_13:
        v28 = a1[2];
        v29 = sub_1648A60(56, 3u);
        if ( v29 )
          sub_15F8650((__int64)v29, v28, a11, (__int64)v23, (__int64)v131);
        for ( i = *(_QWORD *)(v28 + 48); ; i = *(_QWORD *)(i + 8) )
        {
          if ( !i )
            BUG();
          v31 = i - 24;
          if ( *(_BYTE *)(i - 8) != 77 )
            return v131;
          v32 = a1[1];
          v33 = *(_DWORD *)(v32 + 32);
          v34 = *(__int64 **)(v32 + 24);
          if ( v33 > 0x40 )
            v35 = *v34;
          else
            v35 = (__int64)((_QWORD)v34 << (64 - (unsigned __int8)v33)) >> (64 - (unsigned __int8)v33);
          v36 = *(_DWORD *)(*a1 + 32);
          v37 = *(__int64 **)(*a1 + 24);
          if ( v36 > 0x40 )
            v38 = *v37;
          else
            v38 = (__int64)((_QWORD)v37 << (64 - (unsigned __int8)v36)) >> (64 - (unsigned __int8)v36);
          v39 = 0;
          v40 = v35 - v38;
          if ( v40 )
          {
            do
            {
              v41 = *(_DWORD *)(i - 4) & 0xFFFFFFF;
              if ( v41 )
              {
                v42 = 0;
                v43 = 24LL * *(unsigned int *)(i + 32) + 8;
                while ( 1 )
                {
                  v44 = v31 - 24LL * v41;
                  if ( (*(_BYTE *)(i - 1) & 0x40) != 0 )
                    v44 = *(_QWORD *)(i - 32);
                  if ( a10 == *(_QWORD *)(v44 + v43) )
                    break;
                  ++v42;
                  v43 += 8;
                  if ( v41 == v42 )
                    goto LABEL_108;
                }
              }
              else
              {
LABEL_108:
                v42 = -1;
              }
              ++v39;
              sub_15F5350(i - 24, v42, 1);
            }
            while ( v40 != v39 );
          }
          v45 = *(_BYTE *)(i - 1) & 0x40;
          v46 = 24LL * *(unsigned int *)(i + 32);
          v47 = v46 + 8;
          v48 = *(_DWORD *)(i - 4) & 0xFFFFFFF;
          if ( v48 )
          {
            while ( 1 )
            {
              v49 = v31 - 24LL * v48;
              if ( v45 )
                v49 = *(_QWORD *)(i - 32);
              if ( a10 == *(_QWORD *)(v49 + v47) )
                break;
              v47 += 8;
              if ( v47 == v46 + 8LL * (v48 - 1) + 16 )
                goto LABEL_109;
            }
            if ( v45 )
            {
LABEL_37:
              v50 = *(_QWORD *)(i - 32);
              goto LABEL_38;
            }
          }
          else
          {
LABEL_109:
            v47 = v46 + 0x800000000LL;
            if ( v45 )
              goto LABEL_37;
          }
          v50 = v31 - 24LL * v48;
LABEL_38:
          *(_QWORD *)(v50 + v47) = v131;
        }
      }
      if ( v19 == (unsigned int)sub_16A57B0(v22) )
      {
LABEL_9:
        v136 = "SwitchLeaf";
        v138 = 259;
        v23 = sub_1648A60(56, 2u);
        if ( v23 )
        {
          v24 = a1[1];
          v25 = (_QWORD **)*a5;
          if ( *(_BYTE *)(*a5 + 8) == 16 )
          {
            v125 = v25[4];
            v26 = (__int64 *)sub_1643320(*v25);
            v27 = (__int64)sub_16463B0(v26, (unsigned int)v125);
          }
          else
          {
            v27 = sub_1643320(*v25);
          }
          sub_15FED60((__int64)v23, v27, 51, 37, (__int64)a5, v24, (__int64)&v136, (__int64)v131);
        }
        goto LABEL_13;
      }
    }
    v105 = (__int64 *)sub_15A2B90((__int64 *)v18, 0, 0, v21, a7, a8, a9);
    v134 = sub_1649960((__int64)a5);
    v136 = (const char *)&v134;
    v137 = ".off";
    v135 = v106;
    v138 = 773;
    v107 = sub_15FB4C0(11, a5, (__int64)v105, (__int64)&v136, (__int64)v131);
    v108 = sub_15A2B30(v105, a1[1], 0, 0, a7, a8, a9);
    v138 = 259;
    v136 = "SwitchLeaf";
    v23 = sub_1648A60(56, 2u);
    if ( v23 )
    {
      v109 = *(_QWORD ***)v107;
      if ( *(_BYTE *)(*(_QWORD *)v107 + 8LL) == 16 )
      {
        v130 = v109[4];
        v110 = (__int64 *)sub_1643320(*v109);
        v111 = (__int64)sub_16463B0(v110, (unsigned int)v130);
      }
      else
      {
        v111 = sub_1643320(*v109);
      }
      sub_15FED60((__int64)v23, v111, 51, 37, v107, v108, (__int64)&v136, (__int64)v131);
    }
    goto LABEL_13;
  }
  v58 = 24LL * ((unsigned int)v12 >> 1);
  v117 = v58;
  v59 = (const __m128i *)&a1[(unsigned __int64)v58 / 8];
  if ( v58 )
  {
    v60 = sub_22077B0(v58);
    v126 = v60;
    if ( a1 == (__int64 *)v59 )
    {
      v121 = v60;
    }
    else
    {
      v61 = (__m128i *)v60;
      v62 = (const __m128i *)a1;
      do
      {
        if ( v61 )
        {
          v63 = _mm_loadu_si128(v62);
          v61[1].m128i_i64[0] = v62[1].m128i_i64[0];
          *v61 = v63;
        }
        v62 = (const __m128i *)((char *)v62 + 24);
        v61 = (__m128i *)((char *)v61 + 24);
      }
      while ( v59 != v62 );
      v121 = v126 + 8 * ((unsigned __int64)((char *)v59 - (char *)a1 - 24) >> 3) + 24;
    }
  }
  else
  {
    v126 = 0;
    v121 = 0;
  }
  v119 = (char *)a2 - (char *)v59;
  if ( (unsigned __int64)((char *)a2 - (char *)v59) > 0x7FFFFFFFFFFFFFF8LL )
    sub_4262D8((__int64)"cannot create std::vector larger than max_size()");
  v133 = 0;
  if ( v119 )
    v133 = sub_22077B0(v119);
  if ( a2 == v59 )
  {
    LODWORD(v120) = v133;
  }
  else
  {
    v64 = (__m128i *)v133;
    v65 = v59;
    do
    {
      if ( v64 )
      {
        v66 = _mm_loadu_si128(v65);
        v64[1].m128i_i64[0] = v65[1].m128i_i64[0];
        *v64 = v66;
      }
      v65 = (const __m128i *)((char *)v65 + 24);
      v64 = (__m128i *)((char *)v64 + 24);
    }
    while ( a2 != v65 );
    v120 = v133 + 8 * ((unsigned __int64)((char *)&a2[-2].m128i_u64[1] - (char *)v59) >> 3) + 24;
  }
  v67 = v59->m128i_i64[0];
  LODWORD(v135) = *(_DWORD *)(v59->m128i_i64[0] + 32);
  if ( (unsigned int)v135 > 0x40 )
    sub_16A4FD0((__int64)&v134, (const void **)(v67 + 24));
  else
    v134 = *(const char **)(v67 + 24);
  sub_16A7800((__int64)&v134, 1u);
  v68 = v135;
  LODWORD(v135) = 0;
  LODWORD(v137) = v68;
  v136 = v134;
  v69 = (__int64 *)sub_16498A0(v67);
  v70 = sub_159C0E0(v69, (__int64)&v136);
  if ( (unsigned int)v137 > 0x40 && v136 )
    j_j___libc_free_0_0(v136);
  if ( (unsigned int)v135 > 0x40 && v134 )
    j_j___libc_free_0_0(v134);
  v71 = (char *)a12[1];
  v72 = *a12;
  if ( v71 != (char *)*a12 )
  {
    v73 = *(_QWORD *)(v121 - 16);
    v74 = *(_DWORD *)(v73 + 32);
    v75 = *(__int64 **)(v73 + 24);
    v76 = v74 <= 0x40 ? (__int64)((_QWORD)v75 << (64 - (unsigned __int8)v74)) >> (64 - (unsigned __int8)v74) : *v75;
    v77 = *(_DWORD *)(v67 + 32);
    v78 = *(__int64 **)(v67 + 24);
    v79 = v76 + 1;
    v80 = v77 > 0x40 ? *v78 : (__int64)((_QWORD)v78 << (64 - (unsigned __int8)v77)) >> (64 - (unsigned __int8)v77);
    v81 = v80 - 1;
    if ( v79 <= v81 )
    {
      v82 = (v71 - (char *)v72) >> 4;
      if ( v71 - (char *)v72 <= 0 )
        goto LABEL_84;
      do
      {
        while ( 1 )
        {
          v83 = v82 >> 1;
          v84 = &v72[2 * (v82 >> 1)];
          if ( v81 <= v84[1] )
            break;
          v72 = v84 + 2;
          v82 = v82 - v83 - 1;
          if ( v82 <= 0 )
            goto LABEL_83;
        }
        v82 >>= 1;
      }
      while ( v83 > 0 );
LABEL_83:
      if ( v71 != (char *)v72 )
      {
LABEL_84:
        if ( v79 < *v72 )
          LODWORD(v73) = v70;
        v70 = v73;
      }
    }
  }
  v85 = *(_QWORD *)(a10 + 56);
  v138 = 259;
  v118 = v85;
  v136 = "NodeBlock";
  v86 = sub_16498A0((__int64)a5);
  v87 = (_QWORD *)sub_22077B0(64);
  v131 = v87;
  if ( v87 )
    sub_157FB60(v87, v86, (__int64)&v136, 0, 0);
  v136 = "Pivot";
  v138 = 259;
  v88 = sub_1648A60(56, 2u);
  if ( v88 )
  {
    v89 = v59->m128i_i64[0];
    v90 = (_QWORD **)*a5;
    if ( *(_BYTE *)(*a5 + 8) == 16 )
    {
      v116 = v90[4];
      v91 = (__int64 *)sub_1643320(*v90);
      v92 = (__int64)sub_16463B0(v91, (unsigned int)v116);
    }
    else
    {
      v92 = sub_1643320(*v90);
    }
    sub_15FEC10((__int64)v88, v92, 51, 40, (__int64)a5, v89, (__int64)&v136, 0);
  }
  v124 = sub_1B23880(v126, v121, v123, v70, (_DWORD)a5, (_DWORD)v131, a10, a11, (__int64)a12);
  v93 = sub_1B23880(v133, v120, v67, v122, (_DWORD)a5, (_DWORD)v131, a10, a11, (__int64)a12);
  v94 = *(unsigned __int64 **)(a10 + 32);
  v95 = v93;
  sub_15E01D0(v118 + 72, (__int64)v131);
  v96 = *v94;
  v97 = v131[3];
  v131[4] = v94;
  v96 &= 0xFFFFFFFFFFFFFFF8LL;
  v131[3] = v96 | v97 & 7;
  *(_QWORD *)(v96 + 8) = v131 + 3;
  *v94 = *v94 & 7 | (unsigned __int64)(v131 + 3);
  sub_157E9D0((__int64)(v131 + 5), (__int64)v88);
  v98 = v131[5];
  v88[4] = v131 + 5;
  v88[3] = v98 & 0xFFFFFFFFFFFFFFF8LL | v88[3] & 7LL;
  *(_QWORD *)((v98 & 0xFFFFFFFFFFFFFFF8LL) + 8) = v88 + 3;
  v131[5] = v131[5] & 7LL | (unsigned __int64)(v88 + 3);
  v99 = sub_1648A60(56, 3u);
  if ( v99 )
    sub_15F8650((__int64)v99, v124, v95, (__int64)v88, (__int64)v131);
  if ( v133 )
    j_j___libc_free_0(v133, v119);
  if ( v126 )
    j_j___libc_free_0(v126, v117);
  return v131;
}
