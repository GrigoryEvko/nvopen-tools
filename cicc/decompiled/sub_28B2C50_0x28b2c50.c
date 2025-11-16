// Function: sub_28B2C50
// Address: 0x28b2c50
//
__int64 __fastcall sub_28B2C50(_QWORD *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v7; // r13
  bool v10; // al
  unsigned int v11; // r10d
  __int64 v12; // rax
  _QWORD *v14; // rax
  __int64 v15; // rcx
  __int64 *v16; // rax
  __int64 *v17; // rax
  __int64 v18; // rdx
  unsigned int v19; // eax
  __int64 v20; // rbx
  unsigned __int8 *v21; // r14
  __m128i v22; // xmm0
  __m128i v23; // xmm1
  __m128i v24; // xmm2
  unsigned __int8 *v25; // r9
  char v26; // al
  _QWORD *v27; // rax
  __m128i *v28; // r9
  unsigned __int64 v29; // rcx
  unsigned __int8 v30; // si
  unsigned __int64 v31; // rax
  __int64 v32; // rsi
  __m128i v33; // rax
  unsigned __int8 *v34; // rax
  __int64 v35; // rdx
  __int64 v36; // rcx
  __int64 v37; // r8
  __int64 v38; // r9
  char v39; // r10
  _BYTE *v40; // r15
  __int64 v41; // rdx
  __int64 v42; // rcx
  __int64 v43; // r8
  __int64 v44; // r9
  _QWORD *v45; // rdi
  __m128i v46; // xmm3
  __int64 *v47; // rax
  __m128i v48; // xmm4
  __m128i v49; // xmm5
  __int64 *v50; // rax
  unsigned __int8 *v51; // r9
  __int64 v52; // r9
  __int64 v53; // rax
  __int64 v54; // rdx
  __int64 v55; // rax
  __int64 v56; // r14
  unsigned __int64 v57; // rdx
  unsigned __int64 v58; // rax
  char v59; // si
  unsigned __int16 v60; // dx
  unsigned int v61; // eax
  __int64 v62; // rax
  __int64 v63; // r14
  __int64 v64; // rax
  __int64 v65; // r11
  int v66; // ecx
  __int64 v67; // rsi
  int v68; // ecx
  unsigned int v69; // edx
  __int64 *v70; // rax
  __int64 v71; // r8
  __int64 *v72; // rax
  __int64 v73; // rdx
  __int64 v74; // rcx
  __int64 v75; // r8
  __int64 v76; // r9
  __int64 v77; // rdx
  __int64 v78; // rcx
  __int64 v79; // r8
  __int64 v80; // r9
  __m128i v81; // rax
  char v82; // al
  char v83; // r14
  __int64 v84; // rax
  __int64 v85; // rdx
  __int64 v86; // rcx
  __int64 v87; // r8
  __int64 v88; // r9
  __int64 v89; // rdx
  __int64 v90; // rcx
  __int64 v91; // r8
  __int64 v92; // r9
  int v93; // eax
  int v94; // edi
  unsigned __int8 *v95; // [rsp+8h] [rbp-648h]
  __int64 v96; // [rsp+10h] [rbp-640h]
  __int64 v97; // [rsp+18h] [rbp-638h]
  __int64 v98; // [rsp+18h] [rbp-638h]
  __int64 v99; // [rsp+20h] [rbp-630h]
  unsigned __int8 *v100; // [rsp+28h] [rbp-628h]
  __int64 v101; // [rsp+28h] [rbp-628h]
  unsigned __int8 *v102; // [rsp+30h] [rbp-620h]
  char v103; // [rsp+30h] [rbp-620h]
  __int64 v104; // [rsp+30h] [rbp-620h]
  unsigned __int8 *v105; // [rsp+38h] [rbp-618h]
  char v106; // [rsp+38h] [rbp-618h]
  unsigned __int8 v107; // [rsp+38h] [rbp-618h]
  char v108; // [rsp+38h] [rbp-618h]
  char v109; // [rsp+38h] [rbp-618h]
  unsigned __int8 v110; // [rsp+40h] [rbp-610h]
  _QWORD *v111; // [rsp+48h] [rbp-608h] BYREF
  __int64 v112; // [rsp+50h] [rbp-600h] BYREF
  __int64 v113; // [rsp+58h] [rbp-5F8h]
  __m128i v114; // [rsp+60h] [rbp-5F0h] BYREF
  __m128i v115; // [rsp+70h] [rbp-5E0h] BYREF
  __m128i v116; // [rsp+80h] [rbp-5D0h] BYREF
  __m128i v117[3]; // [rsp+90h] [rbp-5C0h] BYREF
  char v118; // [rsp+C0h] [rbp-590h]
  __m128i v119; // [rsp+D0h] [rbp-580h] BYREF
  __m128i v120; // [rsp+E0h] [rbp-570h] BYREF
  __m128i v121; // [rsp+F0h] [rbp-560h]
  char v122; // [rsp+100h] [rbp-550h]
  _QWORD *v123; // [rsp+118h] [rbp-538h]
  void *v124; // [rsp+150h] [rbp-500h]
  _QWORD v125[2]; // [rsp+228h] [rbp-428h] BYREF
  char v126; // [rsp+238h] [rbp-418h]
  _BYTE *v127; // [rsp+240h] [rbp-410h]
  __int64 v128; // [rsp+248h] [rbp-408h]
  _BYTE v129[128]; // [rsp+250h] [rbp-400h] BYREF
  __int16 v130; // [rsp+2D0h] [rbp-380h]
  _QWORD v131[2]; // [rsp+2D8h] [rbp-378h] BYREF
  __int64 v132; // [rsp+2E8h] [rbp-368h]
  __int64 v133; // [rsp+2F0h] [rbp-360h] BYREF
  unsigned int v134; // [rsp+2F8h] [rbp-358h]
  _QWORD *v135; // [rsp+370h] [rbp-2E0h] BYREF
  _QWORD v136[2]; // [rsp+378h] [rbp-2D8h] BYREF
  __int64 v137; // [rsp+388h] [rbp-2C8h]
  __int64 v138; // [rsp+390h] [rbp-2C0h] BYREF
  unsigned int v139; // [rsp+398h] [rbp-2B8h]
  _QWORD v140[2]; // [rsp+4D0h] [rbp-180h] BYREF
  char v141; // [rsp+4E0h] [rbp-170h]
  _BYTE *v142; // [rsp+4E8h] [rbp-168h]
  __int64 v143; // [rsp+4F0h] [rbp-160h]
  _BYTE v144[128]; // [rsp+4F8h] [rbp-158h] BYREF
  __int16 v145; // [rsp+578h] [rbp-D8h]
  void *v146; // [rsp+580h] [rbp-D0h]
  __int64 v147; // [rsp+588h] [rbp-C8h]
  __int64 v148; // [rsp+590h] [rbp-C0h]
  __int64 v149; // [rsp+598h] [rbp-B8h] BYREF
  unsigned int v150; // [rsp+5A0h] [rbp-B0h]
  char v151; // [rsp+618h] [rbp-38h] BYREF

  v7 = (__int64)a1;
  v111 = (_QWORD *)a3;
  v10 = sub_B46500((unsigned __int8 *)a3);
  v11 = 0;
  if ( !v10 )
  {
    v11 = *(_BYTE *)(a3 + 2) & 1;
    if ( (*(_BYTE *)(a3 + 2) & 1) != 0 )
      return 0;
    v12 = *(_QWORD *)(a3 + 16);
    if ( v12 )
    {
      if ( !*(_QWORD *)(v12 + 8) && *(_QWORD *)(a3 + 40) == *(_QWORD *)(a2 + 40) )
      {
        v14 = (_QWORD *)a1[1];
        v15 = a1[7];
        v136[1] = 0;
        v137 = 1;
        v135 = v14;
        v136[0] = v14;
        v16 = &v138;
        do
        {
          *v16 = -4;
          v16 += 5;
          *(v16 - 4) = -3;
          *(v16 - 3) = -4;
          *(v16 - 2) = -3;
        }
        while ( v16 != v140 );
        v140[0] = v15;
        v142 = v144;
        v143 = 0x400000000LL;
        v140[1] = 0;
        v141 = 0;
        v145 = 256;
        v147 = 0;
        v148 = 1;
        v146 = &unk_49DDBE8;
        v17 = &v149;
        do
        {
          *v17 = -4096;
          v17 += 2;
        }
        while ( v17 != (__int64 *)&v151 );
        v99 = *(_QWORD *)(a3 + 8);
        if ( (unsigned int)*(unsigned __int8 *)(v99 + 8) - 15 <= 1 )
        {
          if ( (_BYTE)qword_5004468
            || (v18 = *(_QWORD *)(*a1 + 48LL), (v18 & 0x4000000000LL) == 0)
            && (v19 = *(unsigned __int8 *)(*(_QWORD *)*a1 + 89LL), (v19 & 0x30) != 0)
            && (v18 & 0x8000000000LL) == 0
            && v19 >> 6 )
          {
            sub_D665A0(&v114, a3);
            if ( a2 + 24 == v111[4] )
            {
LABEL_43:
              v25 = (unsigned __int8 *)a2;
            }
            else
            {
              v96 = a5;
              v20 = v111[4];
              v97 = a4;
              while ( 1 )
              {
                v21 = 0;
                if ( v20 )
                  v21 = (unsigned __int8 *)(v20 - 24);
                v22 = _mm_loadu_si128(&v114);
                v23 = _mm_loadu_si128(&v115);
                v24 = _mm_loadu_si128(&v116);
                v122 = 1;
                v119 = v22;
                v120 = v23;
                v121 = v24;
                if ( (sub_CF63E0(v135, v21, &v119, (__int64)v136) & 2) != 0 )
                  break;
                v20 = *(_QWORD *)(v20 + 8);
                if ( a2 + 24 == v20 )
                {
                  v7 = (__int64)a1;
                  a4 = v97;
                  a5 = v96;
                  goto LABEL_43;
                }
              }
              v25 = v21;
              v7 = (__int64)a1;
              a4 = v97;
              a5 = v96;
              if ( (unsigned __int8 *)a2 != v25 )
              {
                v105 = v25;
                v26 = sub_28B18B0((__int64)a1, a2, v25, (__int64)v111);
                v25 = v105;
                if ( !v26 )
                  goto LABEL_26;
              }
            }
            v45 = *(_QWORD **)(v7 + 8);
            v46 = _mm_loadu_si128(&v114);
            v118 = 1;
            v47 = &v120.m128i_i64[1];
            v48 = _mm_loadu_si128(&v115);
            v49 = _mm_loadu_si128(&v116);
            v119 = (__m128i)(unsigned __int64)v45;
            v120.m128i_i64[0] = 1;
            v117[0] = v46;
            v117[1] = v48;
            v117[2] = v49;
            do
            {
              *v47 = -4;
              v47 += 5;
              *(v47 - 4) = -3;
              *(v47 - 3) = -4;
              *(v47 - 2) = -3;
            }
            while ( v47 != v125 );
            v125[1] = 0;
            v127 = v129;
            v128 = 0x400000000LL;
            v125[0] = v131;
            v126 = 0;
            v130 = 256;
            v131[1] = 0;
            v132 = 1;
            v131[0] = &unk_49DDBE8;
            v50 = &v133;
            do
            {
              *v50 = -4096;
              v50 += 2;
            }
            while ( v50 != (__int64 *)&v135 );
            v95 = v25;
            v109 = sub_CF63E0(v45, (unsigned __int8 *)a2, v117, (__int64)&v119);
            v51 = v95;
            v131[0] = &unk_49DDBE8;
            if ( (v132 & 1) == 0 )
            {
              sub_C7D6A0(v133, 16LL * v134, 8);
              v51 = v95;
            }
            v98 = (__int64)v51;
            nullsub_184();
            v52 = v98;
            if ( v127 != v129 )
            {
              _libc_free((unsigned __int64)v127);
              v52 = v98;
            }
            if ( (v120.m128i_i8[0] & 1) == 0 )
            {
              v101 = v52;
              sub_C7D6A0(v120.m128i_i64[1], 40LL * v121.m128i_u32[0], 8);
              v52 = v101;
            }
            sub_23D0AB0((__int64)&v119, v52, 0, 0, 0);
            v53 = sub_9C6480(a4, v99);
            v113 = v54;
            v112 = v53;
            v55 = sub_BCB2E0(v123);
            v56 = sub_B33F60((__int64)&v119, v55, v112, v113);
            _BitScanReverse64(&v57, 1LL << (*((_WORD *)v111 + 1) >> 1));
            _BitScanReverse64(&v58, 1LL << (*(_WORD *)(a2 + 2) >> 1));
            LOBYTE(v60) = 63 - (v57 ^ 0x3F);
            v59 = 63 - (v58 ^ 0x3F);
            v61 = 256;
            HIBYTE(v60) = 1;
            LOBYTE(v61) = v59;
            if ( (v109 & 2) != 0 )
              v62 = sub_B343C0((__int64)&v119, 0xF1u, *(_QWORD *)(a2 - 32), v61, *(v111 - 4), v60, v56, 0, 0, 0, 0, 0);
            else
              v62 = sub_B343C0((__int64)&v119, 0xEEu, *(_QWORD *)(a2 - 32), v61, *(v111 - 4), v60, v56, 0, 0, 0, 0, 0);
            v63 = v62;
            v117[0].m128i_i32[0] = 38;
            sub_B47C00(v62, a2, v117[0].m128i_i32, 1);
            v64 = *(_QWORD *)(v7 + 40);
            v65 = 0;
            v66 = *(_DWORD *)(v64 + 56);
            v67 = *(_QWORD *)(v64 + 40);
            if ( v66 )
            {
              v68 = v66 - 1;
              v69 = v68 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
              v70 = (__int64 *)(v67 + 16LL * v69);
              v71 = *v70;
              if ( a2 == *v70 )
              {
LABEL_58:
                v65 = v70[1];
              }
              else
              {
                v93 = 1;
                while ( v71 != -4096 )
                {
                  v94 = v93 + 1;
                  v69 = v68 & (v93 + v69);
                  v70 = (__int64 *)(v67 + 16LL * v69);
                  v71 = *v70;
                  if ( a2 == *v70 )
                    goto LABEL_58;
                  v93 = v94;
                }
              }
            }
            v72 = (__int64 *)sub_D69570(*(_QWORD **)(v7 + 48), v63, 0, v65);
            sub_D75120(*(__int64 **)(v7 + 48), v72, 1);
            sub_28AAD10(v7, (_QWORD *)a2, v73, v74, v75, v76);
            sub_28AAD10(v7, v111, v77, v78, v79, v80);
            *(_QWORD *)a5 = v63 + 24;
            *(_WORD *)(a5 + 8) = 0;
            nullsub_61();
            v124 = &unk_49DA100;
            nullsub_63();
            if ( (__m128i *)v119.m128i_i64[0] != &v120 )
              _libc_free(v119.m128i_u64[0]);
            v39 = 1;
            goto LABEL_34;
          }
        }
LABEL_26:
        v120.m128i_i64[0] = 0;
        v27 = (_QWORD *)sub_22077B0(0x18u);
        if ( v27 )
        {
          *v27 = v7;
          v27[1] = &v111;
          v27[2] = &v135;
        }
        v119.m128i_i64[0] = (__int64)v27;
        v28 = (__m128i *)&v112;
        v120.m128i_i64[1] = (__int64)sub_28AA9B0;
        v120.m128i_i64[0] = (__int64)sub_28A9760;
        _BitScanReverse64(&v29, 1LL << (*((_WORD *)v111 + 1) >> 1));
        v30 = 63 - (v29 ^ 0x3F);
        LOWORD(v29) = *(_WORD *)(a2 + 2);
        v114.m128i_i8[0] = v30;
        _BitScanReverse64(&v31, 1LL << ((unsigned __int16)v29 >> 1));
        LOBYTE(v112) = 63 - (v31 ^ 0x3F);
        if ( (unsigned __int8)v112 > v30 )
          v28 = &v114;
        v32 = *(_QWORD *)(*(_QWORD *)(a2 - 64) + 8LL);
        v100 = (unsigned __int8 *)v28;
        v33.m128i_i64[0] = (unsigned __int64)(sub_9208B0(a4, v32) + 7) >> 3;
        v117[0] = v33;
        v102 = sub_BD3990((unsigned __int8 *)*(v111 - 4), v32);
        v34 = sub_BD3990(*(unsigned __int8 **)(a2 - 32), v32);
        v39 = sub_28AADB0(
                (_QWORD *)v7,
                (__int64)v111,
                a2,
                (__int64)v34,
                (__int64)v102,
                *v100,
                v117[0].m128i_i8[0],
                v117[0].m128i_i8[8],
                &v135,
                (__int64)&v119);
        if ( v120.m128i_i64[0] )
        {
          v103 = v39;
          ((void (__fastcall *)(__m128i *, __m128i *, __int64))v120.m128i_i64[0])(&v119, &v119, 3);
          v39 = v103;
        }
        if ( v39 )
        {
          v108 = v39;
          sub_28AAD10(v7, (_QWORD *)a2, v35, v36, v37, v38);
          sub_28AAD10(v7, v111, v41, v42, v43, v44);
          v39 = v108;
        }
        else
        {
          v40 = *(_BYTE **)(a2 - 32);
          if ( *v40 == 60 && *(_BYTE *)*(v111 - 4) == 60 )
          {
            v104 = *(v111 - 4);
            v81.m128i_i64[0] = sub_9C6480(a4, v99);
            v119 = v81;
            v82 = sub_28AC0B0(
                    v7,
                    (__int64)v111,
                    a2,
                    (__int64)v40,
                    v104,
                    (__int64)&v135,
                    v81.m128i_u64[0],
                    v81.m128i_i8[8]);
            v39 = 0;
            v83 = v82;
            if ( v82 )
            {
              v84 = sub_B46B10(a2, 0);
              *(_WORD *)(a5 + 8) = 0;
              *(_QWORD *)a5 = v84 + 24;
              sub_28AAD10(v7, (_QWORD *)a2, v85, v86, v87, v88);
              sub_28AAD10(v7, v111, v89, v90, v91, v92);
              v39 = v83;
            }
          }
        }
LABEL_34:
        v146 = &unk_49DDBE8;
        if ( (v148 & 1) == 0 )
        {
          v106 = v39;
          sub_C7D6A0(v149, 16LL * v150, 8);
          v39 = v106;
        }
        v107 = v39;
        nullsub_184();
        v11 = v107;
        if ( v142 != v144 )
        {
          _libc_free((unsigned __int64)v142);
          v11 = v107;
        }
        if ( (v137 & 1) == 0 )
        {
          v110 = v11;
          sub_C7D6A0(v138, 40LL * v139, 8);
          return v110;
        }
      }
    }
  }
  return v11;
}
