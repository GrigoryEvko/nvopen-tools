// Function: sub_25C1030
// Address: 0x25c1030
//
void __fastcall sub_25C1030(__m128i *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned int v6; // r14d
  unsigned __int64 *v7; // r15
  __m128i *v8; // r12
  __int64 v9; // rax
  __int64 v10; // r13
  __int64 v11; // r12
  void (__fastcall *v12)(unsigned __int64 *, __int64, __int64); // rax
  void (__fastcall *v13)(unsigned __int64 *, __int64, __int64); // rax
  void (__fastcall *v14)(unsigned __int64 *, __int64, __int64); // rax
  __int64 *v15; // rax
  __int64 v16; // rax
  __int64 v17; // rbx
  __int64 v18; // rax
  unsigned __int64 *v19; // rdx
  __int64 v20; // rax
  __int64 v21; // rdx
  __int64 m128i_i64; // rsi
  unsigned __int64 *v23; // r13
  unsigned __int64 *v24; // r14
  __m128i *v25; // r12
  __m128i v26; // xmm0
  __int64 v27; // rdx
  __int64 v28; // rax
  __m128i v29; // xmm0
  void (__fastcall *v30)(__m128i *, __m128i *, __int64); // rcx
  __m128i v31; // xmm0
  __int64 v32; // rcx
  __int64 v33; // rdx
  __m128i v34; // xmm0
  void (__fastcall *v35)(__m128i *, __m128i *, __int64); // rax
  __m128i v36; // xmm0
  __int64 v37; // rcx
  __m128i v38; // xmm0
  void (__fastcall *v39)(__m128i *, __m128i *, __int64); // rax
  __int64 v40; // rcx
  unsigned __int64 v41; // r14
  unsigned __int64 v42; // r8
  __m128i *v43; // r13
  __int64 v44; // rdx
  unsigned __int64 v45; // r15
  __int64 v46; // r12
  unsigned __int64 v47; // r9
  __int32 v48; // eax
  __m128i *v49; // r12
  void (__fastcall *v50)(__m128i *, unsigned __int64, __int64); // rax
  void (__fastcall *v51)(__m128i *, unsigned __int64, __int64); // rax
  void (__fastcall *v52)(__m128i *, unsigned __int64, __int64); // rax
  __int64 v53; // rdx
  __int64 *v54; // r13
  __int64 v55; // rax
  unsigned __int64 *v56; // r12
  __int64 *v57; // rbx
  __int64 v58; // rcx
  __int64 v59; // r8
  __int64 v60; // r9
  unsigned __int64 *v61; // rbx
  void (__fastcall *v62)(unsigned __int64 *, unsigned __int64 *, __int64); // rax
  void (__fastcall *v63)(unsigned __int64 *, unsigned __int64 *, __int64); // rax
  void (__fastcall *v64)(unsigned __int64 *, unsigned __int64 *, __int64); // rax
  __int32 v65; // r13d
  __m128i *v66; // r12
  unsigned __int64 *v67; // r12
  __m128i *v68; // r14
  __m128i *v69; // r14
  __m128i *v70; // r14
  __m128i *v71; // rax
  __int64 v72; // r14
  __int64 v73; // r15
  __int64 v74; // rcx
  unsigned __int64 *v75; // r12
  __m128i *v76; // rbx
  __int64 v77; // rax
  __int64 v78; // r14
  __m128i *v79; // r14
  __m128i v80; // xmm0
  __int64 v81; // rdx
  __int64 v82; // rax
  __m128i v83; // xmm0
  void (__fastcall *v84)(__m128i *, __m128i *, __int64); // rsi
  __m128i v85; // xmm0
  __int64 v86; // rsi
  __int64 v87; // rdx
  __m128i v88; // xmm0
  void (__fastcall *v89)(__m128i *, __m128i *, __int64); // rax
  __m128i v90; // xmm0
  __int64 v91; // rsi
  __m128i v92; // xmm0
  void (__fastcall *v93)(__m128i *, __m128i *, __int64); // rax
  __int64 v94; // r14
  __int64 v95; // r12
  __m128i *v96; // r14
  __m128i v97; // xmm0
  __int64 v98; // rdx
  __int64 v99; // rax
  __m128i v100; // xmm0
  void (__fastcall *v101)(__m128i *, __m128i *, __int64); // rcx
  __m128i v102; // xmm0
  __int64 v103; // rcx
  __int64 v104; // rdx
  __m128i v105; // xmm0
  void (__fastcall *v106)(__m128i *, __m128i *, __int64); // rax
  __m128i v107; // xmm0
  __int64 v108; // rcx
  __int64 v109; // rdx
  __m128i v110; // xmm0
  void (__fastcall *v111)(__m128i *, __m128i *, __int64); // rax
  __int32 v112; // eax
  __int64 v113; // rax
  __int64 v114; // r12
  void (__fastcall *v115)(__int64, __int64, __int64); // rax
  void (__fastcall *v116)(__int64, __int64, __int64); // rax
  void (__fastcall *v117)(__int64, __int64, __int64); // rax
  __int64 i; // rax
  __int64 v119; // rcx
  __m128i *v120; // rbx
  void (__fastcall *v121)(__m128i *, __m128i *, __int64); // rax
  void (__fastcall *v122)(__m128i *, __m128i *, __int64); // rax
  void (__fastcall *v123)(__m128i *, __m128i *, __int64); // rax
  __m128i *v124; // r14
  unsigned __int64 v125; // r15
  __int32 v126; // r15d
  __int64 *v129; // [rsp+18h] [rbp-3F8h]
  __int64 v130; // [rsp+30h] [rbp-3E0h]
  __m128i *v131; // [rsp+38h] [rbp-3D8h]
  __int64 v132; // [rsp+40h] [rbp-3D0h]
  __int64 *v133; // [rsp+48h] [rbp-3C8h]
  __m128i *v134; // [rsp+50h] [rbp-3C0h]
  __m128i *v135; // [rsp+58h] [rbp-3B8h]
  __m128i v136; // [rsp+60h] [rbp-3B0h] BYREF
  void (__fastcall *v137)(__m128i *, __m128i *, __int64); // [rsp+70h] [rbp-3A0h]
  __int64 v138; // [rsp+78h] [rbp-398h]
  __m128i *v139; // [rsp+80h] [rbp-390h] BYREF
  __int64 v140; // [rsp+88h] [rbp-388h]
  _BYTE v141[416]; // [rsp+90h] [rbp-380h] BYREF
  __m128i v142; // [rsp+230h] [rbp-1E0h] BYREF
  void (__fastcall *v143)(__m128i *, __m128i *, __int64); // [rsp+240h] [rbp-1D0h] BYREF
  __int64 v144; // [rsp+248h] [rbp-1C8h]

  v135 = (__m128i *)&v139;
  v6 = a1->m128i_u32[2];
  v139 = (__m128i *)v141;
  v140 = 0x400000000LL;
  if ( v6 )
  {
    v7 = (unsigned __int64 *)v141;
    v8 = a1;
    v9 = v6;
    if ( v6 > 4 )
    {
      v7 = (unsigned __int64 *)sub_C8D7D0((__int64)v135, (__int64)v141, v6, 0x68u, (unsigned __int64 *)&v142, a6);
      sub_25BD6E0((__int64)v135, (__m128i *)v7);
      v65 = v142.m128i_i32[0];
      a1 = v139;
      if ( v139 != (__m128i *)v141 )
        _libc_free((unsigned __int64)v139);
      v139 = (__m128i *)v7;
      v9 = v8->m128i_u32[2];
      HIDWORD(v140) = v65;
    }
    v10 = v8->m128i_i64[0];
    v11 = v8->m128i_i64[0] + 104 * v9;
    if ( v10 != v11 )
    {
      do
      {
        if ( v7 )
        {
          v7[2] = 0;
          v12 = *(void (__fastcall **)(unsigned __int64 *, __int64, __int64))(v10 + 16);
          if ( v12 )
          {
            a1 = (__m128i *)v7;
            v12(v7, v10, 2);
            v7[3] = *(_QWORD *)(v10 + 24);
            v7[2] = *(_QWORD *)(v10 + 16);
          }
          v7[6] = 0;
          v13 = *(void (__fastcall **)(unsigned __int64 *, __int64, __int64))(v10 + 48);
          if ( v13 )
          {
            a1 = (__m128i *)(v7 + 4);
            v13(v7 + 4, v10 + 32, 2);
            v7[7] = *(_QWORD *)(v10 + 56);
            v7[6] = *(_QWORD *)(v10 + 48);
          }
          v7[10] = 0;
          v14 = *(void (__fastcall **)(unsigned __int64 *, __int64, __int64))(v10 + 80);
          if ( v14 )
          {
            a1 = (__m128i *)(v7 + 8);
            v14(v7 + 8, v10 + 64, 2);
            v7[11] = *(_QWORD *)(v10 + 88);
            v7[10] = *(_QWORD *)(v10 + 80);
          }
          *((_DWORD *)v7 + 24) = *(_DWORD *)(v10 + 96);
          *((_BYTE *)v7 + 100) = *(_BYTE *)(v10 + 100);
        }
        v10 += 104;
        v7 += 13;
      }
      while ( v11 != v10 );
      v7 = (unsigned __int64 *)v139;
    }
    LODWORD(v140) = v6;
    v15 = *(__int64 **)(a2 + 32);
    v129 = &v15[*(unsigned int *)(a2 + 40)];
    if ( v15 == v129 )
    {
      v61 = &v7[13 * v6];
      goto LABEL_72;
    }
    v133 = *(__int64 **)(a2 + 32);
    v16 = v6;
    while ( 1 )
    {
      v17 = *v133;
      if ( !(_DWORD)v16 )
        goto LABEL_4;
      v136.m128i_i64[0] = *v133;
      v18 = 13 * v16;
      v19 = &v7[v18];
      v20 = 0x4EC4EC4EC4EC4EC5LL * ((v18 * 8) >> 3);
      v134 = (__m128i *)v19;
      v21 = v20 >> 2;
      if ( v20 >> 2 )
      {
        m128i_i64 = v17;
        v23 = &v7[52 * v21];
        while ( 1 )
        {
          if ( !v7[2] )
            goto LABEL_229;
          a1 = (__m128i *)v7;
          if ( !((unsigned __int8 (__fastcall *)(unsigned __int64 *, __int64))v7[3])(v7, m128i_i64) )
          {
            a1 = (__m128i *)v136.m128i_i64[0];
            if ( sub_B2FC80(v136.m128i_i64[0]) )
              goto LABEL_27;
            if ( *((_BYTE *)v7 + 100) )
            {
              v66 = (__m128i *)v136.m128i_i64[0];
              a1 = (__m128i *)v136.m128i_i64[0];
              if ( sub_B2FC80(v136.m128i_i64[0]) )
                goto LABEL_27;
              a1 = v66;
              if ( (unsigned __int8)sub_B2FC00(v66) )
                goto LABEL_27;
            }
          }
          m128i_i64 = v136.m128i_i64[0];
          v67 = v7 + 13;
          if ( !v7[15] )
            goto LABEL_229;
          a1 = (__m128i *)(v7 + 13);
          if ( !((unsigned __int8 (__fastcall *)(unsigned __int64 *, __int64))v7[16])(v7 + 13, v136.m128i_i64[0]) )
          {
            a1 = (__m128i *)v136.m128i_i64[0];
            if ( sub_B2FC80(v136.m128i_i64[0])
              || *((_BYTE *)v7 + 204)
              && ((v68 = (__m128i *)v136.m128i_i64[0], a1 = (__m128i *)v136.m128i_i64[0], sub_B2FC80(v136.m128i_i64[0]))
               || (a1 = v68, (unsigned __int8)sub_B2FC00(v68))) )
            {
LABEL_103:
              v7 = v67;
              goto LABEL_27;
            }
          }
          m128i_i64 = v136.m128i_i64[0];
          v67 = v7 + 26;
          if ( !v7[28] )
            goto LABEL_229;
          a1 = (__m128i *)(v7 + 26);
          if ( !((unsigned __int8 (__fastcall *)(unsigned __int64 *, __int64))v7[29])(v7 + 26, v136.m128i_i64[0]) )
          {
            a1 = (__m128i *)v136.m128i_i64[0];
            if ( sub_B2FC80(v136.m128i_i64[0]) )
              goto LABEL_103;
            if ( *((_BYTE *)v7 + 308) )
            {
              v69 = (__m128i *)v136.m128i_i64[0];
              a1 = (__m128i *)v136.m128i_i64[0];
              if ( sub_B2FC80(v136.m128i_i64[0]) )
                goto LABEL_103;
              a1 = v69;
              if ( (unsigned __int8)sub_B2FC00(v69) )
              {
                v7 += 26;
                goto LABEL_27;
              }
            }
          }
          m128i_i64 = v136.m128i_i64[0];
          v67 = v7 + 39;
          if ( !v7[41] )
            goto LABEL_229;
          a1 = (__m128i *)(v7 + 39);
          if ( ((unsigned __int8 (__fastcall *)(unsigned __int64 *, __int64))v7[42])(v7 + 39, v136.m128i_i64[0]) )
            goto LABEL_99;
          a1 = (__m128i *)v136.m128i_i64[0];
          if ( sub_B2FC80(v136.m128i_i64[0]) )
            goto LABEL_103;
          if ( *((_BYTE *)v7 + 412) )
          {
            v70 = (__m128i *)v136.m128i_i64[0];
            a1 = (__m128i *)v136.m128i_i64[0];
            if ( sub_B2FC80(v136.m128i_i64[0]) )
              goto LABEL_103;
            a1 = v70;
            if ( (unsigned __int8)sub_B2FC00(v70) )
              goto LABEL_103;
            v7 += 52;
            if ( v23 == v7 )
            {
LABEL_110:
              v21 = 0x4EC4EC4EC4EC4EC5LL;
              v20 = 0x4EC4EC4EC4EC4EC5LL * (((char *)v134 - (char *)v7) >> 3);
              break;
            }
          }
          else
          {
LABEL_99:
            v7 += 52;
            if ( v23 == v7 )
              goto LABEL_110;
          }
          m128i_i64 = v136.m128i_i64[0];
        }
      }
      if ( v20 == 2 )
        goto LABEL_224;
      if ( v20 == 3 )
        break;
      if ( v20 != 1 )
        goto LABEL_114;
LABEL_226:
      m128i_i64 = (__int64)v7;
      a1 = &v136;
      if ( !(unsigned __int8)sub_25BCCE0(v136.m128i_i64, (__int64)v7, v21) )
      {
LABEL_114:
        v7 = (unsigned __int64 *)v134;
        goto LABEL_41;
      }
LABEL_27:
      v24 = (unsigned __int64 *)v134;
      if ( v134 != (__m128i *)v7 )
      {
        v25 = (__m128i *)(v7 + 13);
        if ( v134 != (__m128i *)(v7 + 13) )
        {
          do
          {
            if ( !v25[1].m128i_i64[0] )
              goto LABEL_229;
            m128i_i64 = v17;
            a1 = v25;
            if ( ((unsigned __int8 (__fastcall *)(__m128i *, __int64))v25[1].m128i_i64[1])(v25, v17)
              || (a1 = (__m128i *)v17, !sub_B2FC80(v17))
              && (!v25[6].m128i_i8[4]
               || (a1 = (__m128i *)v17, !sub_B2FC80(v17))
               && (a1 = (__m128i *)v17, !(unsigned __int8)sub_B2FC00((_BYTE *)v17))) )
            {
              v26 = _mm_loadu_si128(v25);
              *v25 = _mm_loadu_si128(&v142);
              v142 = v26;
              v27 = v25[1].m128i_i64[0];
              v28 = v25[1].m128i_i64[1];
              v25[1].m128i_i64[0] = 0;
              v25[1].m128i_i64[1] = v144;
              v29 = _mm_loadu_si128(&v142);
              v142 = _mm_loadu_si128((const __m128i *)v7);
              v30 = (void (__fastcall *)(__m128i *, __m128i *, __int64))v7[2];
              *(__m128i *)v7 = v29;
              v143 = v30;
              v7[2] = v27;
              v144 = v7[3];
              v7[3] = v28;
              if ( v143 )
              {
                m128i_i64 = (__int64)&v142;
                a1 = &v142;
                v143(&v142, &v142, 3);
              }
              v31 = _mm_loadu_si128(v25 + 2);
              v25[2] = _mm_loadu_si128(&v142);
              v142 = v31;
              v32 = v25[3].m128i_i64[0];
              v33 = v25[3].m128i_i64[1];
              v25[3].m128i_i64[0] = 0;
              v25[3].m128i_i64[1] = v144;
              v34 = _mm_loadu_si128(&v142);
              v142 = _mm_loadu_si128((const __m128i *)v7 + 2);
              v35 = (void (__fastcall *)(__m128i *, __m128i *, __int64))v7[6];
              *((__m128i *)v7 + 2) = v34;
              v143 = v35;
              v7[6] = v32;
              v144 = v7[7];
              v7[7] = v33;
              if ( v35 )
              {
                m128i_i64 = (__int64)&v142;
                a1 = &v142;
                v35(&v142, &v142, 3);
              }
              v36 = _mm_loadu_si128(v25 + 4);
              v25[4] = _mm_loadu_si128(&v142);
              v142 = v36;
              v37 = v25[5].m128i_i64[0];
              v21 = v25[5].m128i_i64[1];
              v25[5].m128i_i64[0] = 0;
              v25[5].m128i_i64[1] = v144;
              v38 = _mm_loadu_si128(&v142);
              v142 = _mm_loadu_si128((const __m128i *)v7 + 4);
              v39 = (void (__fastcall *)(__m128i *, __m128i *, __int64))v7[10];
              *((__m128i *)v7 + 4) = v38;
              v143 = v39;
              v7[10] = v37;
              v144 = v7[11];
              v7[11] = v21;
              if ( v39 )
              {
                m128i_i64 = (__int64)&v142;
                a1 = &v142;
                v39(&v142, &v142, 3);
              }
              v7 += 13;
              *((_DWORD *)v7 - 2) = v25[6].m128i_i32[0];
              *((_BYTE *)v7 - 4) = v25[6].m128i_i8[4];
            }
            v25 = (__m128i *)((char *)v25 + 104);
          }
          while ( v24 != (unsigned __int64 *)v25 );
        }
      }
LABEL_41:
      a1 = v135;
      m128i_i64 = (__int64)v7;
      sub_25BDD80(v135->m128i_i64, (__m128i *)v7, v134);
      v41 = (unsigned __int64)v139;
      v142.m128i_i64[0] = (__int64)&v143;
      v42 = (unsigned __int64)v139;
      v142.m128i_i64[1] = 0x400000000LL;
      v21 = 3LL * (unsigned int)v140;
      v43 = (__m128i *)((char *)v139 + 104 * (unsigned int)v140);
      v134 = &v136;
      if ( v139 == v43 )
        goto LABEL_60;
      do
      {
        while ( 1 )
        {
          if ( !*(_QWORD *)(v41 + 16) )
            goto LABEL_229;
          m128i_i64 = v17;
          a1 = (__m128i *)v41;
          if ( !(*(unsigned __int8 (__fastcall **)(unsigned __int64, __int64, __int64, __int64, unsigned __int64))(v41 + 24))(
                  v41,
                  v17,
                  v21,
                  v40,
                  v42) )
            break;
          v41 += 104LL;
          if ( v43 == (__m128i *)v41 )
            goto LABEL_56;
        }
        v44 = v142.m128i_u32[2];
        v45 = v41;
        v46 = v142.m128i_i64[0];
        v47 = v142.m128i_u32[2] + 1LL;
        v48 = v142.m128i_i32[2];
        if ( v47 > v142.m128i_u32[3] )
        {
          if ( v142.m128i_i64[0] > v41 || v142.m128i_i64[0] + 104 * (unsigned __int64)v142.m128i_u32[2] <= v41 )
          {
            m128i_i64 = sub_C8D7D0(
                          (__int64)&v142,
                          (__int64)&v143,
                          v142.m128i_u32[2] + 1LL,
                          0x68u,
                          (unsigned __int64 *)v134,
                          v47);
            v46 = m128i_i64;
            sub_25BD6E0((__int64)&v142, (__m128i *)m128i_i64);
            v126 = v136.m128i_i32[0];
            a1 = (__m128i *)v142.m128i_i64[0];
            if ( (void (__fastcall **)(__m128i *, __m128i *, __int64))v142.m128i_i64[0] != &v143 )
              _libc_free(v142.m128i_u64[0]);
            v44 = v142.m128i_u32[2];
            v142.m128i_i32[3] = v126;
            v45 = v41;
            v142.m128i_i64[0] = m128i_i64;
            v48 = v142.m128i_i32[2];
          }
          else
          {
            v125 = v41 - v142.m128i_i64[0];
            m128i_i64 = sub_C8D7D0(
                          (__int64)&v142,
                          (__int64)&v143,
                          v142.m128i_u32[2] + 1LL,
                          0x68u,
                          (unsigned __int64 *)v134,
                          v47);
            v46 = m128i_i64;
            sub_25BD6E0((__int64)&v142, (__m128i *)m128i_i64);
            a1 = (__m128i *)v142.m128i_i64[0];
            if ( (void (__fastcall **)(__m128i *, __m128i *, __int64))v142.m128i_i64[0] == &v143 )
            {
              v142.m128i_i64[0] = m128i_i64;
              v142.m128i_i32[3] = v136.m128i_i32[0];
            }
            else
            {
              v132 = v136.m128i_i64[0];
              _libc_free(v142.m128i_u64[0]);
              v142.m128i_i64[0] = m128i_i64;
              v142.m128i_i32[3] = v132;
            }
            v44 = v142.m128i_u32[2];
            v45 = m128i_i64 + v125;
            v48 = v142.m128i_i32[2];
          }
        }
        v40 = 3 * v44;
        v21 = 13 * v44;
        v49 = (__m128i *)(v46 + 8 * v21);
        if ( v49 )
        {
          v49[1].m128i_i64[0] = 0;
          v50 = *(void (__fastcall **)(__m128i *, unsigned __int64, __int64))(v45 + 16);
          if ( v50 )
          {
            m128i_i64 = v45;
            a1 = v49;
            v50(v49, v45, 2);
            v49[1].m128i_i64[1] = *(_QWORD *)(v45 + 24);
            v49[1].m128i_i64[0] = *(_QWORD *)(v45 + 16);
          }
          v49[3].m128i_i64[0] = 0;
          v51 = *(void (__fastcall **)(__m128i *, unsigned __int64, __int64))(v45 + 48);
          if ( v51 )
          {
            m128i_i64 = v45 + 32;
            a1 = v49 + 2;
            v51(v49 + 2, v45 + 32, 2);
            v49[3].m128i_i64[1] = *(_QWORD *)(v45 + 56);
            v49[3].m128i_i64[0] = *(_QWORD *)(v45 + 48);
          }
          v49[5].m128i_i64[0] = 0;
          v52 = *(void (__fastcall **)(__m128i *, unsigned __int64, __int64))(v45 + 80);
          if ( v52 )
          {
            m128i_i64 = v45 + 64;
            a1 = v49 + 4;
            v52(v49 + 4, v45 + 64, 2);
            v49[5].m128i_i64[1] = *(_QWORD *)(v45 + 88);
            v49[5].m128i_i64[0] = *(_QWORD *)(v45 + 80);
          }
          v49[6].m128i_i32[0] = *(_DWORD *)(v45 + 96);
          v49[6].m128i_i8[4] = *(_BYTE *)(v45 + 100);
          v48 = v142.m128i_i32[2];
        }
        v41 += 104LL;
        v142.m128i_i32[2] = v48 + 1;
      }
      while ( v43 != (__m128i *)v41 );
LABEL_56:
      v53 = v142.m128i_u32[2];
      if ( !v142.m128i_i32[2] )
      {
        a1 = (__m128i *)v142.m128i_i64[0];
        if ( (void (__fastcall **)(__m128i *, __m128i *, __int64))v142.m128i_i64[0] != &v143 )
          _libc_free(v142.m128i_u64[0]);
        goto LABEL_59;
      }
      v71 = *(__m128i **)(v17 + 80);
      a1 = (__m128i *)(v17 + 72);
      v131 = (__m128i *)(v17 + 72);
      v132 = (__int64)v71;
      if ( (__m128i *)(v17 + 72) == v71 )
      {
        v134 = 0;
      }
      else
      {
        do
        {
          if ( !v71 )
LABEL_234:
            BUG();
          m128i_i64 = v71[2].m128i_i64[0];
          if ( (unsigned __int64 *)m128i_i64 != &v71[1].m128i_u64[1] )
            break;
          v71 = (__m128i *)v71->m128i_i64[1];
        }
        while ( a1 != v71 );
        v132 = (__int64)v71;
        v134 = (__m128i *)m128i_i64;
      }
      v72 = v142.m128i_i64[0];
      if ( v131 != (__m128i *)v132 )
      {
        v73 = v142.m128i_i64[0];
        while ( 1 )
        {
          v74 = 0x4EC4EC4EC4EC4EC5LL;
          v75 = &v134[-2].m128i_u64[1];
          if ( !v134 )
            v75 = 0;
          v76 = (__m128i *)(v73 + 104 * v53);
          v77 = 0x4EC4EC4EC4EC4EC5LL * ((104 * v53) >> 3);
          v21 = v77 >> 2;
          if ( !(v77 >> 2) )
            break;
          v78 = v73 + 416 * v21;
          while ( 1 )
          {
            if ( !*(_QWORD *)(v73 + 48) )
              goto LABEL_229;
            a1 = (__m128i *)(v73 + 32);
            m128i_i64 = (__int64)v75;
            if ( (*(unsigned __int8 (__fastcall **)(__int64, unsigned __int64 *, __int64, __int64))(v73 + 56))(
                   v73 + 32,
                   v75,
                   v21,
                   v74) )
            {
              goto LABEL_131;
            }
            if ( !*(_QWORD *)(v73 + 152) )
              goto LABEL_229;
            a1 = (__m128i *)(v73 + 136);
            m128i_i64 = (__int64)v75;
            if ( (*(unsigned __int8 (__fastcall **)(__int64, unsigned __int64 *))(v73 + 160))(v73 + 136, v75) )
            {
              v73 += 104;
              a1 = v135;
              m128i_i64 = v73;
              sub_25BE790((__int64)v135, v73);
              goto LABEL_132;
            }
            if ( !*(_QWORD *)(v73 + 256) )
              goto LABEL_229;
            a1 = (__m128i *)(v73 + 240);
            m128i_i64 = (__int64)v75;
            if ( (*(unsigned __int8 (__fastcall **)(__int64, unsigned __int64 *))(v73 + 264))(v73 + 240, v75) )
              break;
            if ( !*(_QWORD *)(v73 + 360) )
              goto LABEL_229;
            a1 = (__m128i *)(v73 + 344);
            m128i_i64 = (__int64)v75;
            if ( (*(unsigned __int8 (__fastcall **)(__int64, unsigned __int64 *))(v73 + 368))(v73 + 344, v75) )
            {
              v73 += 312;
              a1 = v135;
              m128i_i64 = v73;
              sub_25BE790((__int64)v135, v73);
              goto LABEL_132;
            }
            v73 += 416;
            if ( v78 == v73 )
            {
              v74 = 0x4EC4EC4EC4EC4EC5LL;
              v77 = 0x4EC4EC4EC4EC4EC5LL * (((__int64)v76->m128i_i64 - v73) >> 3);
              goto LABEL_157;
            }
          }
          v73 += 208;
          a1 = v135;
          m128i_i64 = v73;
          sub_25BE790((__int64)v135, v73);
LABEL_132:
          if ( v76 != (__m128i *)v73 )
          {
            v79 = (__m128i *)(v73 + 104);
            if ( v76 != (__m128i *)(v73 + 104) )
            {
              while ( v79[3].m128i_i64[0] )
              {
                a1 = v79 + 2;
                if ( ((unsigned __int8 (__fastcall *)(__int64 *, unsigned __int64 *))v79[3].m128i_i64[1])(
                       v79[2].m128i_i64,
                       v75) )
                {
                  a1 = v135;
                  m128i_i64 = (__int64)v79;
                  sub_25BE790((__int64)v135, (__int64)v79);
                }
                else
                {
                  v80 = _mm_loadu_si128(v79);
                  *v79 = _mm_loadu_si128(&v136);
                  v136 = v80;
                  v81 = v79[1].m128i_i64[0];
                  v82 = v79[1].m128i_i64[1];
                  v79[1].m128i_i64[0] = 0;
                  v79[1].m128i_i64[1] = v138;
                  v83 = _mm_loadu_si128(&v136);
                  v136 = _mm_loadu_si128((const __m128i *)v73);
                  v84 = *(void (__fastcall **)(__m128i *, __m128i *, __int64))(v73 + 16);
                  *(__m128i *)v73 = v83;
                  v137 = v84;
                  *(_QWORD *)(v73 + 16) = v81;
                  v138 = *(_QWORD *)(v73 + 24);
                  *(_QWORD *)(v73 + 24) = v82;
                  if ( v137 )
                  {
                    a1 = &v136;
                    v137(&v136, &v136, 3);
                  }
                  v85 = _mm_loadu_si128(v79 + 2);
                  v79[2] = _mm_loadu_si128(&v136);
                  v136 = v85;
                  v86 = v79[3].m128i_i64[0];
                  v87 = v79[3].m128i_i64[1];
                  v79[3].m128i_i64[0] = 0;
                  v79[3].m128i_i64[1] = v138;
                  v88 = _mm_loadu_si128(&v136);
                  v136 = _mm_loadu_si128((const __m128i *)(v73 + 32));
                  v89 = *(void (__fastcall **)(__m128i *, __m128i *, __int64))(v73 + 48);
                  *(__m128i *)(v73 + 32) = v88;
                  v137 = v89;
                  *(_QWORD *)(v73 + 48) = v86;
                  v138 = *(_QWORD *)(v73 + 56);
                  *(_QWORD *)(v73 + 56) = v87;
                  if ( v89 )
                  {
                    a1 = &v136;
                    v89(&v136, &v136, 3);
                  }
                  v90 = _mm_loadu_si128(v79 + 4);
                  v79[4] = _mm_loadu_si128(&v136);
                  v136 = v90;
                  v91 = v79[5].m128i_i64[0];
                  v21 = v79[5].m128i_i64[1];
                  v79[5].m128i_i64[0] = 0;
                  v79[5].m128i_i64[1] = v138;
                  v92 = _mm_loadu_si128(&v136);
                  v136 = _mm_loadu_si128((const __m128i *)(v73 + 64));
                  v93 = *(void (__fastcall **)(__m128i *, __m128i *, __int64))(v73 + 80);
                  *(__m128i *)(v73 + 64) = v92;
                  v137 = v93;
                  *(_QWORD *)(v73 + 80) = v91;
                  m128i_i64 = *(_QWORD *)(v73 + 88);
                  v138 = m128i_i64;
                  *(_QWORD *)(v73 + 88) = v21;
                  if ( v93 )
                  {
                    m128i_i64 = (__int64)&v136;
                    a1 = &v136;
                    v93(&v136, &v136, 3);
                  }
                  v73 += 104;
                  *(_DWORD *)(v73 - 8) = v79[6].m128i_i32[0];
                  *(_BYTE *)(v73 - 4) = v79[6].m128i_i8[4];
                }
                v79 = (__m128i *)((char *)v79 + 104);
                if ( v76 == v79 )
                  goto LABEL_161;
              }
LABEL_229:
              sub_4263D6(a1, m128i_i64, v21);
            }
          }
LABEL_161:
          m128i_i64 = 3LL * v142.m128i_u32[2];
          v94 = v142.m128i_i64[0] + 104LL * v142.m128i_u32[2];
          v130 = v94 - (_QWORD)v76;
          v95 = 0x4EC4EC4EC4EC4EC5LL * ((v94 - (__int64)v76) >> 3);
          if ( v94 - (__int64)v76 <= 0 )
          {
            v114 = v73;
            v73 = v142.m128i_i64[0];
          }
          else
          {
            v96 = (__m128i *)v73;
            do
            {
              v97 = _mm_loadu_si128(v76);
              *v76 = _mm_loadu_si128(&v136);
              v136 = v97;
              v98 = v76[1].m128i_i64[0];
              v99 = v76[1].m128i_i64[1];
              v76[1].m128i_i64[0] = 0;
              v76[1].m128i_i64[1] = v138;
              v100 = _mm_loadu_si128(&v136);
              v136 = _mm_loadu_si128(v96);
              v101 = (void (__fastcall *)(__m128i *, __m128i *, __int64))v96[1].m128i_i64[0];
              *v96 = v100;
              v137 = v101;
              v96[1].m128i_i64[0] = v98;
              v138 = v96[1].m128i_i64[1];
              v96[1].m128i_i64[1] = v99;
              if ( v137 )
              {
                m128i_i64 = (__int64)&v136;
                a1 = &v136;
                v137(&v136, &v136, 3);
              }
              v102 = _mm_loadu_si128(v76 + 2);
              v76[2] = _mm_loadu_si128(&v136);
              v136 = v102;
              v103 = v76[3].m128i_i64[0];
              v104 = v76[3].m128i_i64[1];
              v76[3].m128i_i64[0] = 0;
              v76[3].m128i_i64[1] = v138;
              v105 = _mm_loadu_si128(&v136);
              v136 = _mm_loadu_si128(v96 + 2);
              v106 = (void (__fastcall *)(__m128i *, __m128i *, __int64))v96[3].m128i_i64[0];
              v96[2] = v105;
              v137 = v106;
              v96[3].m128i_i64[0] = v103;
              v138 = v96[3].m128i_i64[1];
              v96[3].m128i_i64[1] = v104;
              if ( v106 )
              {
                m128i_i64 = (__int64)&v136;
                a1 = &v136;
                v106(&v136, &v136, 3);
              }
              v107 = _mm_loadu_si128(v76 + 4);
              v76[4] = _mm_loadu_si128(&v136);
              v136 = v107;
              v108 = v76[5].m128i_i64[0];
              v109 = v76[5].m128i_i64[1];
              v76[5].m128i_i64[0] = 0;
              v76[5].m128i_i64[1] = v138;
              v110 = _mm_loadu_si128(&v136);
              v136 = _mm_loadu_si128(v96 + 4);
              v111 = (void (__fastcall *)(__m128i *, __m128i *, __int64))v96[5].m128i_i64[0];
              v96[4] = v110;
              v137 = v111;
              v96[5].m128i_i64[0] = v108;
              v138 = v96[5].m128i_i64[1];
              v96[5].m128i_i64[1] = v109;
              if ( v111 )
              {
                m128i_i64 = (__int64)&v136;
                a1 = &v136;
                v111(&v136, &v136, 3);
              }
              v112 = v76[6].m128i_i32[0];
              v96 = (__m128i *)((char *)v96 + 104);
              v76 = (__m128i *)((char *)v76 + 104);
              v96[-1].m128i_i32[2] = v112;
              v96[-1].m128i_i8[12] = v76[-1].m128i_i8[12];
              --v95;
            }
            while ( v95 );
            v113 = v130;
            if ( v130 <= 0 )
              v113 = 104;
            v114 = v73 + v113;
            v73 = v142.m128i_i64[0];
            v94 = v142.m128i_i64[0] + 104LL * v142.m128i_u32[2];
          }
          if ( v114 != v94 )
          {
            do
            {
              v115 = *(void (__fastcall **)(__int64, __int64, __int64))(v94 - 24);
              v94 -= 104;
              if ( v115 )
              {
                a1 = (__m128i *)(v94 + 64);
                m128i_i64 = v94 + 64;
                v115(v94 + 64, v94 + 64, 3);
              }
              v116 = *(void (__fastcall **)(__int64, __int64, __int64))(v94 + 48);
              if ( v116 )
              {
                a1 = (__m128i *)(v94 + 32);
                m128i_i64 = v94 + 32;
                v116(v94 + 32, v94 + 32, 3);
              }
              v117 = *(void (__fastcall **)(__int64, __int64, __int64))(v94 + 16);
              if ( v117 )
              {
                m128i_i64 = v94;
                a1 = (__m128i *)v94;
                v117(v94, v94, 3);
              }
            }
            while ( v94 != v114 );
            v73 = v142.m128i_i64[0];
          }
          v142.m128i_i32[2] = -991146299 * ((v114 - v73) >> 3);
          v53 = v142.m128i_u32[2];
          if ( !v142.m128i_i32[2] )
          {
            v124 = (__m128i *)v73;
            goto LABEL_201;
          }
          a1 = v131;
          m128i_i64 = v134->m128i_i64[1];
          for ( i = v132; ; m128i_i64 = *(_QWORD *)(i + 32) )
          {
            v119 = i - 24;
            if ( !i )
              v119 = 0;
            if ( m128i_i64 != v119 + 48 )
              break;
            i = *(_QWORD *)(i + 8);
            if ( v131 == (__m128i *)i )
              goto LABEL_191;
            if ( !i )
              goto LABEL_234;
          }
          v132 = i;
          v134 = (__m128i *)m128i_i64;
          if ( (__m128i *)i == v131 )
          {
LABEL_191:
            v72 = v73;
            goto LABEL_192;
          }
        }
LABEL_157:
        if ( v77 == 2 )
          goto LABEL_214;
        if ( v77 != 3 )
        {
          if ( v77 != 1 )
            goto LABEL_160;
          goto LABEL_217;
        }
        if ( !*(_QWORD *)(v73 + 48) )
          goto LABEL_229;
        a1 = (__m128i *)(v73 + 32);
        m128i_i64 = (__int64)v75;
        if ( !(*(unsigned __int8 (__fastcall **)(__int64, unsigned __int64 *, __int64, __int64))(v73 + 56))(
                v73 + 32,
                v75,
                v21,
                0x4EC4EC4EC4EC4EC5LL) )
        {
          v73 += 104;
LABEL_214:
          if ( !*(_QWORD *)(v73 + 48) )
            goto LABEL_229;
          a1 = (__m128i *)(v73 + 32);
          m128i_i64 = (__int64)v75;
          if ( !(*(unsigned __int8 (__fastcall **)(__int64, unsigned __int64 *, __int64, __int64))(v73 + 56))(
                  v73 + 32,
                  v75,
                  v21,
                  v74) )
          {
            v73 += 104;
LABEL_217:
            if ( !*(_QWORD *)(v73 + 48) )
              goto LABEL_229;
            a1 = (__m128i *)(v73 + 32);
            if ( !(*(unsigned __int8 (__fastcall **)(__int64, unsigned __int64 *, __int64, __int64))(v73 + 56))(
                    v73 + 32,
                    v75,
                    v21,
                    v74) )
            {
LABEL_160:
              v73 = (__int64)v76;
              goto LABEL_161;
            }
          }
        }
LABEL_131:
        a1 = v135;
        m128i_i64 = v73;
        sub_25BE790((__int64)v135, v73);
        goto LABEL_132;
      }
LABEL_192:
      v120 = (__m128i *)(v72 + 104LL * (unsigned int)v53);
      do
      {
        v121 = (void (__fastcall *)(__m128i *, __m128i *, __int64))v120[-2].m128i_i64[1];
        v120 = (__m128i *)((char *)v120 - 104);
        if ( v121 )
        {
          a1 = v120 + 4;
          m128i_i64 = (__int64)v120[4].m128i_i64;
          v121(v120 + 4, v120 + 4, 3);
        }
        v122 = (void (__fastcall *)(__m128i *, __m128i *, __int64))v120[3].m128i_i64[0];
        if ( v122 )
        {
          a1 = v120 + 2;
          m128i_i64 = (__int64)v120[2].m128i_i64;
          v122(v120 + 2, v120 + 2, 3);
        }
        v123 = (void (__fastcall *)(__m128i *, __m128i *, __int64))v120[1].m128i_i64[0];
        if ( v123 )
        {
          m128i_i64 = (__int64)v120;
          a1 = v120;
          v123(v120, v120, 3);
        }
      }
      while ( v120 != (__m128i *)v72 );
      v124 = (__m128i *)v142.m128i_i64[0];
LABEL_201:
      if ( v124 != (__m128i *)&v143 )
      {
        a1 = v124;
        _libc_free((unsigned __int64)v124);
        v42 = (unsigned __int64)v139;
        goto LABEL_60;
      }
LABEL_59:
      v42 = (unsigned __int64)v139;
LABEL_60:
      ++v133;
      v16 = (unsigned int)v140;
      v7 = (unsigned __int64 *)v42;
      if ( v129 == v133 )
      {
        if ( !(_DWORD)v140 )
          goto LABEL_4;
        v54 = *(__int64 **)(a2 + 32);
        v134 = (__m128i *)&v54[*(unsigned int *)(a2 + 40)];
        if ( v134 == (__m128i *)v54 )
        {
          v61 = (unsigned __int64 *)(v42 + 104LL * (unsigned int)v140);
        }
        else
        {
          v135 = (__m128i *)v54;
          v55 = (unsigned int)v140;
          do
          {
            v21 = 13 * v55;
            v56 = &v7[13 * v55];
            v57 = (__int64 *)v135->m128i_i64[0];
            if ( v56 == v7 )
            {
              v61 = v7;
            }
            else
            {
              do
              {
                if ( !v7[2] )
                  goto LABEL_229;
                m128i_i64 = (__int64)v57;
                a1 = (__m128i *)v7;
                if ( !((unsigned __int8 (__fastcall *)(unsigned __int64 *, __int64 *))v7[3])(v7, v57) )
                {
                  m128i_i64 = a3;
                  a1 = &v142;
                  sub_25C0F40((__int64)&v142, a3, v57, v58, v59, v60);
                  if ( !v7[10] )
                    goto LABEL_229;
                  a1 = (__m128i *)(v7 + 8);
                  m128i_i64 = (__int64)v57;
                  ((void (__fastcall *)(unsigned __int64 *, __int64 *))v7[11])(v7 + 8, v57);
                }
                v7 += 13;
              }
              while ( v7 != v56 );
              v55 = (unsigned int)v140;
              v7 = (unsigned __int64 *)v139;
              v61 = (unsigned __int64 *)v139 + 13 * (unsigned int)v140;
            }
            v135 = (__m128i *)((char *)v135 + 8);
          }
          while ( v134 != v135 );
        }
LABEL_72:
        if ( v61 != v7 )
        {
          do
          {
            v62 = (void (__fastcall *)(unsigned __int64 *, unsigned __int64 *, __int64))*(v61 - 3);
            v61 -= 13;
            if ( v62 )
              v62(v61 + 8, v61 + 8, 3);
            v63 = (void (__fastcall *)(unsigned __int64 *, unsigned __int64 *, __int64))v61[6];
            if ( v63 )
              v63(v61 + 4, v61 + 4, 3);
            v64 = (void (__fastcall *)(unsigned __int64 *, unsigned __int64 *, __int64))v61[2];
            if ( v64 )
              v64(v61, v61, 3);
          }
          while ( v61 != v7 );
          v7 = (unsigned __int64 *)v139;
        }
        goto LABEL_4;
      }
    }
    m128i_i64 = (__int64)v7;
    a1 = &v136;
    if ( (unsigned __int8)sub_25BCCE0(v136.m128i_i64, (__int64)v7, v21) )
      goto LABEL_27;
    v7 += 13;
LABEL_224:
    m128i_i64 = (__int64)v7;
    a1 = &v136;
    if ( (unsigned __int8)sub_25BCCE0(v136.m128i_i64, (__int64)v7, v21) )
      goto LABEL_27;
    v7 += 13;
    goto LABEL_226;
  }
  if ( !*(_DWORD *)(a2 + 40) )
    return;
  v7 = (unsigned __int64 *)v141;
LABEL_4:
  if ( v7 != (unsigned __int64 *)v141 )
    _libc_free((unsigned __int64)v7);
}
