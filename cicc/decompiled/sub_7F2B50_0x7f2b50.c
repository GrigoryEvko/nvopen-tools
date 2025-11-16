// Function: sub_7F2B50
// Address: 0x7f2b50
//
__int64 *__fastcall sub_7F2B50(__int64 *a1, const __m128i *a2)
{
  __int64 v2; // r14
  __m128i *v3; // r15
  __m128i *v4; // r12
  __m128i *v5; // r13
  __int64 v6; // rdx
  __int64 v7; // rcx
  __int64 v8; // r8
  __int64 v9; // r9
  __m128i *v10; // rax
  __int64 v11; // rsi
  __int64 v12; // rdx
  __int64 v13; // rcx
  __int64 v14; // r8
  __int64 v15; // r9
  __m128i *v16; // r15
  __int64 v17; // r8
  const __m128i *v18; // rax
  __int8 v19; // al
  __int64 v20; // rdx
  __int64 v21; // rcx
  __int64 v22; // r8
  __int64 v23; // r9
  __int64 *result; // rax
  const __m128i *v25; // rax
  __int64 v26; // rdx
  __int64 v27; // rcx
  __int64 v28; // rax
  __int64 v29; // rax
  bool v30; // cf
  bool v31; // zf
  const char *v32; // rdi
  __int64 v33; // rax
  __int64 v34; // rdx
  __int64 v35; // rcx
  __int64 v36; // r8
  __int64 v37; // r9
  __int64 v38; // xmm0_8
  __m128i v39; // xmm1
  __m128i v40; // xmm2
  __m128i v41; // xmm3
  __int64 v42; // rdx
  __int64 v43; // rcx
  __int64 v44; // r8
  __int64 v45; // rax
  __int64 v46; // r13
  __m128i v47; // xmm5
  __m128i v48; // xmm6
  __m128i v49; // xmm7
  __int64 v50; // rdx
  __int64 v51; // rcx
  __int64 v52; // r8
  __int64 v53; // rax
  __int64 v54; // rcx
  __int64 v55; // r8
  __int64 v56; // rbx
  __int64 v57; // rax
  __int64 v58; // r9
  __int64 v59; // r12
  __int64 v60; // rdx
  __int64 v61; // rdx
  __int64 v62; // rcx
  __int64 v63; // r8
  __int64 v64; // r9
  __int64 v65; // rdx
  __int64 v66; // rcx
  __int64 v67; // r8
  __int64 v68; // r9
  __int64 v69; // rdx
  __int64 v70; // rax
  __int64 v71; // rdx
  __int64 v72; // rcx
  __int64 v73; // r8
  __int64 v74; // r9
  __int64 v75; // r15
  __int64 v76; // rax
  _BYTE *v77; // rax
  __int64 v78; // rdx
  __int64 v79; // rdi
  _BYTE *v80; // r9
  __int64 v81; // rax
  __int64 v82; // r13
  __int64 v83; // rax
  __int64 v84; // rax
  _BYTE *v85; // r13
  _BYTE *v86; // rax
  __int64 v87; // rdx
  const __m128i *v88; // rbx
  unsigned __int64 v89; // r14
  __int64 v90; // r15
  __int64 v91; // rdi
  unsigned __int64 v92; // rcx
  unsigned __int64 v93; // rdx
  unsigned __int64 v94; // rcx
  __int64 *v95; // r13
  __int64 v96; // rax
  _BYTE *v97; // r13
  __int64 v98; // rax
  _BYTE *v99; // rax
  _BYTE *v100; // rax
  __int64 v101; // rcx
  __int64 v102; // r8
  __int64 v103; // r9
  __int64 v104; // rax
  __int64 k; // rax
  __int64 v106; // rax
  _BYTE *v107; // rax
  _QWORD *v108; // rbx
  _QWORD *v109; // rax
  __int64 v110; // rsi
  __int64 v111; // rax
  _QWORD *i; // r15
  __int64 v113; // rax
  int j; // edx
  __int64 v115; // rax
  const __m128i *v116; // [rsp+0h] [rbp-190h]
  __int64 v117; // [rsp+0h] [rbp-190h]
  __int64 v118; // [rsp+8h] [rbp-188h]
  __int64 v119; // [rsp+8h] [rbp-188h]
  __int64 v120; // [rsp+10h] [rbp-180h]
  _QWORD *v121; // [rsp+18h] [rbp-178h]
  _QWORD *v122; // [rsp+20h] [rbp-170h]
  _QWORD *v123; // [rsp+28h] [rbp-168h]
  __int64 v124; // [rsp+28h] [rbp-168h]
  _BYTE *v125; // [rsp+28h] [rbp-168h]
  __m128i *v126; // [rsp+28h] [rbp-168h]
  __int64 v127; // [rsp+28h] [rbp-168h]
  unsigned int v128; // [rsp+30h] [rbp-160h]
  __int64 v129; // [rsp+30h] [rbp-160h]
  __int64 v130; // [rsp+30h] [rbp-160h]
  const __m128i *v131; // [rsp+38h] [rbp-158h]
  unsigned int v132; // [rsp+44h] [rbp-14Ch] BYREF
  __int64 v133; // [rsp+48h] [rbp-148h] BYREF
  __int64 v134; // [rsp+50h] [rbp-140h] BYREF
  __int64 v135; // [rsp+58h] [rbp-138h] BYREF
  __int64 v136; // [rsp+60h] [rbp-130h] BYREF
  __int64 v137; // [rsp+68h] [rbp-128h] BYREF
  int v138[8]; // [rsp+70h] [rbp-120h] BYREF
  __m128i v139; // [rsp+90h] [rbp-100h] BYREF
  __m128i v140; // [rsp+A0h] [rbp-F0h]
  __m128i v141; // [rsp+B0h] [rbp-E0h]
  __m128i v142; // [rsp+C0h] [rbp-D0h]
  unsigned int v143; // [rsp+E0h] [rbp-B0h]

  v2 = (__int64)a1;
  v3 = (__m128i *)a1[9];
  v128 = (unsigned int)a2;
  v4 = (__m128i *)v3[1].m128i_i64[0];
  v5 = (__m128i *)v4[1].m128i_i64[0];
  v131 = 0;
  if ( !(unsigned int)sub_8D2600(*a1) )
  {
    v131 = v4;
    if ( v4[1].m128i_i8[8] != 8 )
    {
      v10 = 0;
      if ( v5[1].m128i_i8[8] == 8 )
        v10 = v5;
      v131 = v10;
    }
  }
  if ( (*((_BYTE *)a1 + 25) & 1) == 0 )
  {
    if ( (v4[1].m128i_i8[9] & 1) != 0 )
    {
      a2 = (const __m128i *)sub_731370((__int64)v4, (__int64)a2, v6, v7, v8, v9);
      sub_730620((__int64)v4, a2);
      if ( (v5[1].m128i_i8[9] & 1) == 0 )
        goto LABEL_9;
    }
    else if ( (v5[1].m128i_i8[9] & 1) == 0 )
    {
      goto LABEL_9;
    }
    v25 = (const __m128i *)sub_731370((__int64)v5, (__int64)a2, v6, v7, v8, v9);
    sub_730620((__int64)v5, v25);
  }
LABEL_9:
  if ( dword_4F077C4 == 2 )
    sub_7F2A70(v3, 0);
  else
    sub_7D98E0((__int64)v3, 0);
  v11 = unk_4F06964;
  if ( sub_7E6F30((__int64)v3, unk_4F06964, &v132) )
  {
    v11 = v132;
    if ( v132 )
    {
      v16 = v4;
      v17 = (__int64)v5;
    }
    else
    {
      v16 = v5;
      v17 = (__int64)v4;
    }
    v123 = (_QWORD *)v17;
    if ( !(unsigned int)sub_731D60(v17, v132, v12, v13, v17, v15) )
    {
      v14 = (__int64)v123;
      if ( dword_4F077C4 != 2 )
      {
        if ( v16 != v131 )
          goto LABEL_17;
LABEL_84:
        v11 = (__int64)a1;
        sub_7E0CD0(v131, (__int64)a1);
        v131 = 0;
LABEL_36:
        if ( dword_4F077C4 == 2 )
        {
          sub_7EE560(v16, (__m128i *)v128);
          goto LABEL_18;
        }
LABEL_17:
        sub_7D9DD0(v16, v11, v12, v13, v14, v15);
LABEL_18:
        v18 = (const __m128i *)sub_73E130(v16, *a1);
        sub_730620((__int64)a1, v18);
        goto LABEL_25;
      }
      sub_76C7C0((__int64)&v139);
      v11 = (__int64)&v139;
      v139.m128i_i64[0] = (__int64)sub_7E0550;
      sub_76CDC0(v123, (__int64)&v139, v26, v27, (__int64)v123);
      v13 = v143;
      if ( !v143 )
      {
        if ( v16 != v131 )
          goto LABEL_36;
        goto LABEL_84;
      }
    }
  }
  else
  {
    while ( v3[1].m128i_i8[8] == 1 )
    {
      v19 = v3[3].m128i_i8[8];
      if ( v19 != 91 )
      {
        if ( v19 != 105 )
          break;
        if ( (v3[1].m128i_i8[11] & 2) == 0 )
          break;
        v28 = v3[4].m128i_i64[1];
        if ( *(_BYTE *)(v28 + 24) != 20 )
          break;
        v29 = *(_QWORD *)(v28 + 56);
        if ( !v29 )
          break;
        v12 = *(unsigned __int8 *)(v29 + 89);
        if ( (v12 & 0x40) != 0 )
          break;
        v12 &= 8u;
        v11 = (_DWORD)v12 ? *(_QWORD *)(v29 + 24) : *(_QWORD *)(v29 + 8);
        v30 = 0;
        v31 = v11 == 0;
        if ( !v11 )
          break;
        v13 = 28;
        v32 = "__cudaPushCallConfiguration";
        do
        {
          if ( !v13 )
            break;
          v30 = *(_BYTE *)v11 < *v32;
          v31 = *(_BYTE *)v11++ == *v32++;
          --v13;
        }
        while ( v31 );
        if ( (!v30 && !v31) != v30 )
          break;
        v12 = (unsigned int)dword_4D04530;
        if ( dword_4D04530 )
        {
          if ( !qword_4F04C50 )
            break;
          v33 = *(_QWORD *)(qword_4F04C50 + 32LL);
          if ( !v33 )
            break;
          if ( (*(_BYTE *)(v33 + 193) & 2) != 0 && (*(_BYTE *)(v33 + 198) & 0x10) == 0 )
          {
            if ( dword_4F077C4 == 2 )
              sub_7EE560(v4, (__m128i *)v128);
            else
              sub_7D9DD0(v4, v11, (unsigned int)dword_4D04530, v13, v14, v15);
            sub_730620(v2, v4);
            goto LABEL_25;
          }
        }
        else
        {
          if ( !qword_4F04C50 )
            break;
          v33 = *(_QWORD *)(qword_4F04C50 + 32LL);
          if ( !v33 )
            break;
        }
        if ( (*(_BYTE *)(v33 + 198) & 0x10) == 0 )
          break;
        if ( v5 )
        {
          if ( v5[1].m128i_i8[8] == 1 && v5[3].m128i_i8[8] == 105 )
          {
            v111 = v5[4].m128i_i64[1];
            if ( v111 )
            {
              for ( i = *(_QWORD **)(v111 + 16); i; i = (_QWORD *)i[2] )
              {
                v113 = *i;
                for ( j = *(unsigned __int8 *)(*i + 140LL); (_BYTE)j == 12; j = *(unsigned __int8 *)(v113 + 140) )
                  v113 = *(_QWORD *)(v113 + 160);
                v12 = (unsigned int)(j - 9);
                if ( (unsigned __int8)v12 <= 2u )
                {
                  v115 = *(_QWORD *)(*(_QWORD *)v113 + 96LL);
                  if ( (*(_BYTE *)(v115 + 177) & 0x40) == 0 )
                  {
                    v11 = (__int64)dword_4F07508;
                    v127 = v115;
                    sub_6851C0(0xDC3u, dword_4F07508);
                    v115 = v127;
                  }
                  if ( *(_QWORD *)(v115 + 24) )
                  {
                    v11 = (__int64)dword_4F07508;
                    sub_6851C0(0xDC4u, dword_4F07508);
                  }
                }
              }
            }
          }
        }
        if ( dword_4F077C4 == 2 )
        {
          v11 = v128;
          sub_7EE560(v4, (__m128i *)v128);
        }
        else
        {
          sub_7D9DD0(v4, v11, v12, v13, v14, v15);
        }
        if ( dword_4F077C4 == 2 )
          sub_7EE560(v5, (__m128i *)v128);
        else
          sub_7D9DD0(v5, v11, v34, v35, v36, v37);
        v133 = 0;
        v38 = _mm_loadu_si128(xmmword_4F06660).m128i_u64[0];
        v39 = _mm_loadu_si128(&xmmword_4F06660[1]);
        v134 = 0;
        v40 = _mm_loadu_si128(&xmmword_4F06660[2]);
        v41 = _mm_loadu_si128(&xmmword_4F06660[3]);
        v135 = 0;
        v139.m128i_i64[0] = v38;
        v136 = 0;
        v137 = 0;
        v139.m128i_i64[1] = *(_QWORD *)&dword_4F077C8;
        v140 = v39;
        v141 = v40;
        v142 = v41;
        sub_878540("cudaGetParameterBufferV2", 0x18u);
        v45 = sub_7D5DD0(&v139, 0, v42, v43, v44);
        if ( !v45 )
          goto LABEL_96;
        v46 = *(_QWORD *)(v45 + 88);
        v47 = _mm_loadu_si128(&xmmword_4F06660[1]);
        v48 = _mm_loadu_si128(&xmmword_4F06660[2]);
        v49 = _mm_loadu_si128(&xmmword_4F06660[3]);
        v139.m128i_i64[0] = _mm_loadu_si128(xmmword_4F06660).m128i_u64[0];
        v140 = v47;
        v139.m128i_i64[1] = *(_QWORD *)&dword_4F077C8;
        v141 = v48;
        v142 = v49;
        sub_878540("cudaLaunchDeviceV2", 0x12u);
        v53 = sub_7D5DD0(&v139, 0, v50, v51, v52);
        if ( v53 )
        {
          v121 = 0;
          v56 = *(_QWORD *)(v53 + 88);
          v120 = *(_QWORD *)(*(_QWORD *)(v2 + 72) + 16LL);
          v57 = *(_QWORD *)(v120 + 16);
          if ( *(_BYTE *)(v57 + 24) == 1 && *(_BYTE *)(v57 + 56) == 91 )
          {
            v121 = *(_QWORD **)(v120 + 16);
            v57 = *(_QWORD *)(*(_QWORD *)(v57 + 72) + 16LL);
          }
          v122 = 0;
          v58 = *(_QWORD *)(v57 + 72);
          v59 = *(_QWORD *)(v58 + 16);
          *(_QWORD *)(v58 + 16) = 0;
          v60 = *(_QWORD *)(v2 + 72);
          if ( *(_BYTE *)(v60 + 24) == 1 && *(_BYTE *)(v60 + 56) == 91 )
          {
            v122 = *(_QWORD **)(v2 + 72);
            v60 = *(_QWORD *)(*(_QWORD *)(v60 + 72) + 16LL);
          }
          v124 = v58;
          v129 = v60;
          v133 = *(_QWORD *)(*(_QWORD *)(v60 + 72) + 16LL);
          v134 = *(_QWORD *)(v133 + 16);
          v135 = *(_QWORD *)(v134 + 16);
          v136 = *(_QWORD *)(v135 + 16);
          sub_7E9060(&v133, &v137, v60, v54, v55, v58);
          sub_7E9060(&v134, &v137, v61, v62, v63, v64);
          sub_7E9060(&v135, &v137, v65, v66, v67, v68);
          v69 = v133;
          *(_QWORD *)(*(_QWORD *)(v129 + 72) + 16LL) = v133;
          v70 = v134;
          *(_QWORD *)(v69 + 16) = v134;
          v71 = v135;
          *(_QWORD *)(v70 + 16) = v135;
          sub_7E9060(&v136, &v137, v71, v72, v73, v74);
          *(_QWORD *)(v135 + 16) = 0;
          *(_QWORD *)(v136 + 16) = 0;
          v75 = *(_QWORD *)(*****(_QWORD *****)(*(_QWORD *)(v46 + 152) + 168LL) + 8LL);
          v118 = *(_QWORD *)(***(_QWORD ***)(*(_QWORD *)(v56 + 152) + 168LL) + 8LL);
          v76 = sub_7E1C10();
          v77 = sub_73E110(v124, v76);
          v78 = v134;
          v79 = v135;
          v80 = v77;
          v81 = v133;
          v125 = v80;
          *((_QWORD *)v80 + 2) = v133;
          *(_QWORD *)(v81 + 16) = v78;
          v130 = *(_QWORD *)(*((_QWORD *)v80 + 2) + 16LL);
          *(_QWORD *)(v130 + 16) = sub_73E110(v79, v75);
          *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(*((_QWORD *)v125 + 2) + 16LL) + 16LL) + 16LL) = 0;
          v82 = sub_7F88E0(v46, v125);
          v83 = sub_7E1C30();
          v126 = (__m128i *)sub_73E110(v82, v83);
          v116 = (const __m128i *)sub_7E88C0(v126);
          v84 = sub_7E1C10();
          v85 = sub_73E110((__int64)v116, v84);
          v86 = sub_73E110(v136, v118);
          *((_QWORD *)v85 + 2) = v86;
          *((_QWORD *)v86 + 2) = 0;
          v119 = sub_7F88E0(v56, v85);
          sub_7E1780(v119, (__int64)v138);
          if ( v59 )
          {
            v88 = v116;
            v117 = v2;
            v89 = 0;
            do
            {
              v90 = v59;
              v59 = *(_QWORD *)(v59 + 16);
              v91 = *(_QWORD *)v90;
              *(_QWORD *)(v90 + 16) = 0;
              if ( *(char *)(v91 + 142) >= 0 && *(_BYTE *)(v91 + 140) == 12 )
                v92 = (unsigned int)sub_8D4AB0(v91, v138, v87);
              else
                v92 = *(unsigned int *)(v91 + 136);
              v93 = v89 % v92;
              v94 = v89 + v92 - v89 % v92;
              if ( v93 )
                v89 = v94;
              v95 = (__int64 *)sub_730FF0(v88);
              if ( v89 )
              {
                v109 = sub_73A830(v89, byte_4F06A51[0]);
                v110 = *v95;
                v95[2] = (__int64)v109;
                v95 = (__int64 *)sub_73DBF0(0x32u, v110, (__int64)v95);
              }
              v96 = sub_7E1C10();
              v97 = sub_73E110((__int64)v95, v96);
              v98 = sub_72D2E0(*(_QWORD **)v90);
              v99 = sub_73E110((__int64)v97, v98);
              v100 = sub_73DCD0(v99);
              v104 = sub_698020(v100, 73, v90, v101, v102, v103);
              sub_7E25D0(v104, v138);
              for ( k = *(_QWORD *)v90; *(_BYTE *)(k + 140) == 12; k = *(_QWORD *)(k + 160) )
                ;
              v89 += *(_QWORD *)(k + 128);
            }
            while ( v59 );
            v2 = v117;
          }
          *(_BYTE *)(v119 + 25) |= 4u;
          v106 = sub_72CBE0();
          v107 = sub_73E110(v119, v106);
          v107[25] |= 4u;
          v108 = v107;
          if ( v137 )
            v126 = (__m128i *)sub_73DF90(v137, v126->m128i_i64);
          if ( v122 )
          {
            *v122 = v126->m128i_i64[0];
            *(_QWORD *)(v122[9] + 16LL) = v126;
          }
          else
          {
            *(_QWORD *)(v2 + 72) = v126;
          }
          if ( v121 )
          {
            *v121 = *v108;
            *(_QWORD *)(v121[9] + 16LL) = v108;
            *(_QWORD *)(*(_QWORD *)(v2 + 72) + 16LL) = v121;
          }
          else
          {
            *(_QWORD *)(*(_QWORD *)(v2 + 72) + 16LL) = v108;
          }
          *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(v2 + 72) + 16LL) + 16LL) = v120;
          *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(*(_QWORD *)(v2 + 72) + 16LL) + 16LL) + 16LL) = 0;
        }
        else
        {
LABEL_96:
          sub_6851C0(0xDC2u, dword_4F07508);
        }
        goto LABEL_25;
      }
      v3 = *(__m128i **)(v3[4].m128i_i64[1] + 16);
    }
  }
  if ( dword_4F077C4 == 2 )
  {
    v11 = v128;
    sub_7EE560(v4, (__m128i *)v128);
    if ( dword_4F077C4 != 2 )
      goto LABEL_24;
  }
  else
  {
    sub_7D9DD0(v4, v11, v12, v13, v14, v15);
    if ( dword_4F077C4 != 2 )
    {
LABEL_24:
      sub_7D9DD0(v5, v11, v20, v21, v22, v23);
      goto LABEL_25;
    }
  }
  sub_7EE560(v5, (__m128i *)v128);
LABEL_25:
  result = (__int64 *)v131;
  if ( v131 )
    return sub_7E0CD0(v131, v2);
  return result;
}
