// Function: sub_6980A0
// Address: 0x6980a0
//
__int64 __fastcall sub_6980A0(const __m128i *a1, _DWORD *a2, unsigned int a3, int a4, int a5, int a6)
{
  unsigned int v7; // edi
  __int64 v8; // rax
  __int64 result; // rax
  int v13; // eax
  _QWORD *v14; // rdx
  __int64 v15; // rax
  __int64 v16; // rax
  __int64 v17; // rax
  char j; // dl
  __int64 v19; // rcx
  char i; // al
  __m128i v21; // xmm1
  __m128i v22; // xmm2
  __m128i v23; // xmm3
  __m128i v24; // xmm4
  __m128i v25; // xmm5
  __m128i v26; // xmm6
  __m128i v27; // xmm7
  __m128i v28; // xmm0
  __int8 v29; // al
  __int64 v30; // r12
  __int64 v31; // rax
  _QWORD *v32; // rbx
  __int64 v33; // rax
  __int64 v34; // r12
  _QWORD *v35; // rax
  __int64 v36; // rcx
  __int64 v37; // r8
  __int64 v38; // r9
  __int64 v39; // rax
  __int64 v40; // r12
  __int64 v41; // rbx
  __int64 v42; // rax
  const char *v43; // rdx
  __m128i v44; // xmm2
  __m128i v45; // xmm3
  __m128i v46; // xmm4
  __m128i v47; // xmm5
  __m128i v48; // xmm6
  __m128i v49; // xmm7
  __m128i v50; // xmm1
  __m128i v51; // xmm2
  __m128i v52; // xmm4
  __m128i v53; // xmm5
  __m128i v54; // xmm6
  __m128i v55; // xmm7
  __m128i v56; // xmm3
  __m128i v57; // xmm1
  __m128i v58; // xmm2
  __m128i v59; // xmm4
  __m128i v60; // xmm5
  __m128i v61; // xmm6
  __m128i v62; // xmm7
  __m128i v63; // xmm3
  __m128i v64; // xmm5
  __m128i v65; // xmm6
  __m128i v66; // xmm7
  __m128i v67; // xmm1
  __m128i v68; // xmm2
  __m128i v69; // xmm4
  __m128i v70; // xmm5
  __m128i v71; // xmm6
  __m128i v72; // xmm0
  __m128i v73; // xmm1
  __m128i v74; // xmm2
  __m128i v75; // xmm7
  __m128i v76; // xmm4
  __m128i v77; // xmm5
  __m128i v78; // xmm6
  __m128i v79; // xmm3
  __m128i v80; // xmm0
  __m128i v81; // xmm7
  __m128i v82; // xmm1
  __m128i v83; // xmm2
  __m128i v84; // xmm5
  __m128i v85; // xmm6
  __m128i v86; // xmm0
  __m128i v87; // xmm4
  __m128i v88; // xmm3
  __m128i v89; // xmm5
  __m128i v90; // xmm6
  __m128i v91; // xmm4
  __m128i v92; // xmm5
  __m128i v93; // xmm6
  __m128i v94; // xmm4
  __m128i v95; // xmm5
  __int64 v96; // [rsp+10h] [rbp-A40h]
  _QWORD *v97; // [rsp+18h] [rbp-A38h]
  __int64 v98; // [rsp+20h] [rbp-A30h]
  _BYTE v100[4]; // [rsp+38h] [rbp-A18h] BYREF
  _BYTE v101[4]; // [rsp+3Ch] [rbp-A14h] BYREF
  __m128i v102[4]; // [rsp+40h] [rbp-A10h] BYREF
  _BYTE v103[16]; // [rsp+80h] [rbp-9D0h] BYREF
  char v104; // [rsp+90h] [rbp-9C0h]
  _BYTE v105[352]; // [rsp+1E0h] [rbp-870h] BYREF
  _QWORD v106[2]; // [rsp+340h] [rbp-710h] BYREF
  char v107; // [rsp+350h] [rbp-700h]
  _BYTE v108[352]; // [rsp+4A0h] [rbp-5B0h] BYREF
  __m128i v109; // [rsp+600h] [rbp-450h] BYREF
  __m128i v110; // [rsp+610h] [rbp-440h] BYREF
  __m128i v111; // [rsp+620h] [rbp-430h] BYREF
  __m128i v112; // [rsp+630h] [rbp-420h] BYREF
  __m128i v113; // [rsp+640h] [rbp-410h] BYREF
  __m128i v114; // [rsp+650h] [rbp-400h] BYREF
  __m128i v115; // [rsp+660h] [rbp-3F0h] BYREF
  __m128i v116; // [rsp+670h] [rbp-3E0h] BYREF
  __m128i v117; // [rsp+680h] [rbp-3D0h] BYREF
  __m128i v118; // [rsp+690h] [rbp-3C0h] BYREF
  __m128i v119; // [rsp+6A0h] [rbp-3B0h] BYREF
  __m128i v120; // [rsp+6B0h] [rbp-3A0h] BYREF
  __m128i v121; // [rsp+6C0h] [rbp-390h] BYREF
  __m128i v122; // [rsp+6D0h] [rbp-380h] BYREF
  __m128i v123; // [rsp+6E0h] [rbp-370h] BYREF
  __m128i v124; // [rsp+6F0h] [rbp-360h] BYREF
  __m128i v125; // [rsp+700h] [rbp-350h] BYREF
  __m128i v126; // [rsp+710h] [rbp-340h] BYREF
  __m128i v127; // [rsp+720h] [rbp-330h] BYREF
  __m128i v128; // [rsp+730h] [rbp-320h] BYREF
  __m128i v129; // [rsp+740h] [rbp-310h] BYREF
  __m128i v130; // [rsp+750h] [rbp-300h] BYREF
  _OWORD v131[9]; // [rsp+760h] [rbp-2F0h] BYREF
  __m128i v132; // [rsp+7F0h] [rbp-260h]
  __m128i v133; // [rsp+800h] [rbp-250h]
  __m128i v134; // [rsp+810h] [rbp-240h]
  __m128i v135; // [rsp+820h] [rbp-230h]
  __m128i v136; // [rsp+830h] [rbp-220h]
  __m128i v137; // [rsp+840h] [rbp-210h]
  __m128i v138; // [rsp+850h] [rbp-200h]
  __m128i v139; // [rsp+860h] [rbp-1F0h]
  __m128i v140; // [rsp+870h] [rbp-1E0h]
  __m128i v141; // [rsp+880h] [rbp-1D0h]
  __m128i v142; // [rsp+890h] [rbp-1C0h]
  __m128i v143; // [rsp+8A0h] [rbp-1B0h]
  __m128i v144; // [rsp+8B0h] [rbp-1A0h]
  _OWORD v145[9]; // [rsp+8C0h] [rbp-190h] BYREF
  __m128i v146; // [rsp+950h] [rbp-100h]
  __m128i v147; // [rsp+960h] [rbp-F0h]
  __m128i v148; // [rsp+970h] [rbp-E0h]
  __m128i v149; // [rsp+980h] [rbp-D0h]
  __m128i v150; // [rsp+990h] [rbp-C0h]
  __m128i v151; // [rsp+9A0h] [rbp-B0h]
  __m128i v152; // [rsp+9B0h] [rbp-A0h]
  __m128i v153; // [rsp+9C0h] [rbp-90h]
  __m128i v154; // [rsp+9D0h] [rbp-80h]
  __m128i v155; // [rsp+9E0h] [rbp-70h]
  __m128i v156; // [rsp+9F0h] [rbp-60h]
  __m128i v157; // [rsp+A00h] [rbp-50h]
  __m128i v158; // [rsp+A10h] [rbp-40h]

  v7 = 2679;
  if ( !unk_4F04C50 )
    goto LABEL_5;
  v8 = *(_QWORD *)(unk_4F04C50 + 32LL);
  if ( v8 && (*(_BYTE *)(v8 + 198) & 0x10) != 0 )
  {
    v7 = 3708;
LABEL_5:
    sub_6851C0(v7, a2);
    return sub_6E6260(a1);
  }
  v7 = 2737;
  if ( (*(_BYTE *)(qword_4D03C50 + 17LL) & 2) == 0 )
    goto LABEL_5;
  v13 = sub_86FC80(2737);
  v14 = &qword_4D03C50;
  v7 = 2680;
  if ( v13 )
    goto LABEL_5;
  v7 = 2980;
  if ( (*(_BYTE *)(qword_4D03C50 + 19LL) & 4) != 0 )
    goto LABEL_5;
  if ( dword_4F04C44 != -1
    || (v14 = qword_4F04C68, v15 = qword_4F04C68[0] + 776LL * dword_4F04C64, (*(_BYTE *)(v15 + 6) & 6) != 0)
    || *(_BYTE *)(v15 + 4) == 12 )
  {
    if ( (unsigned int)sub_82ED00(a1, a2, v14) )
      goto LABEL_42;
  }
  v98 = sub_71DF80(*(_QWORD *)(unk_4F04C50 + 32LL));
  if ( (*(_BYTE *)(v98 + 120) & 1) != 0 )
    return sub_6E6260(a1);
  if ( dword_4F04C44 != -1
    || (v16 = qword_4F04C68[0] + 776LL * dword_4F04C64, (*(_BYTE *)(v16 + 6) & 6) != 0)
    || *(_BYTE *)(v16 + 4) == 12 )
  {
    if ( (unsigned int)sub_8DBE70(*(_QWORD *)(*(_QWORD *)(v98 + 16) + 120LL)) )
    {
LABEL_42:
      sub_6F40C0(a1);
      return sub_7032B0(117 - ((unsigned int)(a4 == 0) - 1), a1, a1, a2, a3);
    }
  }
  if ( a4 )
  {
    v97 = (_QWORD *)sub_726700(28);
    v42 = sub_6F6F40(a1, 0);
    v97[7] = sub_73B8B0(v42, 0);
LABEL_24:
    sub_84EC30(47, 1, 0, 0, 1, (_DWORD)a1, 0, (__int64)a2, a3, 0, 0, (__int64)a1, 0, 0, (__int64)v101);
    if ( a1[1].m128i_i8[1] != 2 || (unsigned int)sub_8D2600(a1->m128i_i64[0]) )
    {
      if ( (unsigned int)sub_6ED0A0(a1) )
        sub_6ED030(a1);
    }
    else if ( (unsigned int)sub_8D3A70(a1->m128i_i64[0]) )
    {
      sub_6F9770(a1, 0);
    }
    else
    {
      sub_844770(a1, 1);
    }
    v19 = a1->m128i_i64[0];
    for ( i = *(_BYTE *)(a1->m128i_i64[0] + 140); i == 12; i = *(_BYTE *)(v19 + 140) )
      v19 = *(_QWORD *)(v19 + 160);
    if ( (unsigned __int8)(i - 9) > 2u )
    {
      v43 = "co_yield";
      if ( !a4 )
        v43 = "co_await";
      sub_686470(0xBA3u, a2, (__int64)v43, v19);
      return sub_6E6260(a1);
    }
    else
    {
      v21 = _mm_loadu_si128(a1 + 1);
      v22 = _mm_loadu_si128(a1 + 2);
      v23 = _mm_loadu_si128(a1 + 3);
      v24 = _mm_loadu_si128(a1 + 4);
      v25 = _mm_loadu_si128(a1 + 5);
      v109 = _mm_loadu_si128(a1);
      v26 = _mm_loadu_si128(a1 + 6);
      v27 = _mm_loadu_si128(a1 + 7);
      v110 = v21;
      v28 = _mm_loadu_si128(a1 + 8);
      v29 = a1[1].m128i_i8[0];
      v111 = v22;
      v112 = v23;
      v113 = v24;
      v114 = v25;
      v115 = v26;
      v116 = v27;
      v117 = v28;
      if ( v29 == 2 )
      {
        v52 = _mm_loadu_si128(a1 + 10);
        v53 = _mm_loadu_si128(a1 + 11);
        v54 = _mm_loadu_si128(a1 + 12);
        v55 = _mm_loadu_si128(a1 + 13);
        v118 = _mm_loadu_si128(a1 + 9);
        v56 = _mm_loadu_si128(a1 + 14);
        v57 = _mm_loadu_si128(a1 + 19);
        v119 = v52;
        v58 = _mm_loadu_si128(a1 + 20);
        v59 = _mm_loadu_si128(a1 + 15);
        v120 = v53;
        v60 = _mm_loadu_si128(a1 + 16);
        v121 = v54;
        v61 = _mm_loadu_si128(a1 + 17);
        v122 = v55;
        v62 = _mm_loadu_si128(a1 + 18);
        v123 = v56;
        v63 = _mm_loadu_si128(a1 + 21);
        v124 = v59;
        v125 = v60;
        v126 = v61;
        v127 = v62;
        v128 = v57;
        v129 = v58;
        v130 = v63;
      }
      else if ( v29 == 5 || v29 == 1 )
      {
        v118.m128i_i64[0] = a1[9].m128i_i64[0];
      }
      sub_7029D0(a1, "await_ready", 0, 0, &v109, v103);
      if ( v104 == 2 )
      {
        v44 = _mm_loadu_si128(&v110);
        v45 = _mm_loadu_si128(&v111);
        v46 = _mm_loadu_si128(&v112);
        v47 = _mm_loadu_si128(&v113);
        v48 = _mm_loadu_si128(&v114);
        v131[0] = _mm_loadu_si128(&v109);
        v49 = _mm_loadu_si128(&v115);
        v50 = _mm_loadu_si128(&v116);
        v131[1] = v44;
        v51 = _mm_loadu_si128(&v117);
        v131[2] = v45;
        v131[3] = v46;
        v131[4] = v47;
        v131[5] = v48;
        v131[6] = v49;
        v131[7] = v50;
        v131[8] = v51;
        switch ( v110.m128i_i8[0] )
        {
          case 2:
            v72 = _mm_loadu_si128(&v119);
            v73 = _mm_loadu_si128(&v121);
            v74 = _mm_loadu_si128(&v122);
            v132 = _mm_loadu_si128(&v118);
            v75 = _mm_loadu_si128(&v120);
            v76 = _mm_loadu_si128(&v123);
            v77 = _mm_loadu_si128(&v124);
            v78 = _mm_loadu_si128(&v125);
            v133 = v72;
            v79 = _mm_loadu_si128(&v127);
            v80 = _mm_loadu_si128(&v126);
            v134 = v75;
            v135 = v73;
            v81 = _mm_loadu_si128(&v128);
            v82 = _mm_loadu_si128(&v129);
            v136 = v74;
            v83 = _mm_loadu_si128(&v130);
            v137 = v76;
            v138 = v77;
            v139 = v78;
            v140 = v80;
            v141 = v79;
            v142 = v81;
            v143 = v82;
            v144 = v83;
            break;
          case 5:
            v132.m128i_i64[0] = v118.m128i_i64[0];
            break;
          case 1:
            v132.m128i_i64[0] = v118.m128i_i64[0];
            break;
        }
      }
      else
      {
        sub_6FF9F0(&v109, v131, 1, v100, 1);
      }
      sub_7029D0(v131, "await_resume", 0, 0, &v109, v106);
      if ( v104 == 2 && v107 == 2 )
      {
        v64 = _mm_loadu_si128(&v110);
        v65 = _mm_loadu_si128(&v111);
        v66 = _mm_loadu_si128(&v112);
        v67 = _mm_loadu_si128(&v113);
        v68 = _mm_loadu_si128(&v114);
        v145[0] = _mm_loadu_si128(&v109);
        v145[1] = v64;
        v69 = _mm_loadu_si128(&v115);
        v70 = _mm_loadu_si128(&v116);
        v145[2] = v65;
        v71 = _mm_loadu_si128(&v117);
        v145[3] = v66;
        v145[4] = v67;
        v145[5] = v68;
        v145[6] = v69;
        v145[7] = v70;
        v145[8] = v71;
        switch ( v110.m128i_i8[0] )
        {
          case 2:
            v84 = _mm_loadu_si128(&v119);
            v85 = _mm_loadu_si128(&v120);
            v86 = _mm_loadu_si128(&v127);
            v146 = _mm_loadu_si128(&v118);
            v87 = _mm_loadu_si128(&v121);
            v88 = _mm_loadu_si128(&v128);
            v147 = v84;
            v89 = _mm_loadu_si128(&v122);
            v148 = v85;
            v90 = _mm_loadu_si128(&v123);
            v149 = v87;
            v91 = _mm_loadu_si128(&v124);
            v150 = v89;
            v92 = _mm_loadu_si128(&v125);
            v151 = v90;
            v93 = _mm_loadu_si128(&v126);
            v152 = v91;
            v94 = _mm_loadu_si128(&v129);
            v153 = v92;
            v95 = _mm_loadu_si128(&v130);
            v154 = v93;
            v155 = v86;
            v156 = v88;
            v157 = v94;
            v158 = v95;
            break;
          case 5:
            v146.m128i_i64[0] = v118.m128i_i64[0];
            break;
          case 1:
            v146.m128i_i64[0] = v118.m128i_i64[0];
            break;
        }
      }
      else
      {
        sub_6FF9F0(&v109, v145, 1, v100, 1);
      }
      sub_6F8E70(*(_QWORD *)(v98 + 8), a2, a2, v108, 0);
      v30 = sub_6E3060(v108);
      sub_7029D0(v145, "await_suspend", 0, v30, &v109, v105);
      sub_6E1990(v30);
      *v97 = v106[0];
      v31 = sub_6F6F40(v103, 0);
      v97[8] = v31;
      if ( a6 )
      {
        v32 = (_QWORD *)sub_6F6F40(v106, 0);
        v33 = sub_72C390();
        v34 = sub_73A7F0(v33);
        v35 = (_QWORD *)sub_731250(*(_QWORD *)(v98 + 24));
        v39 = sub_698020(v35, 73, v34, v36, v37, v38);
        *(_QWORD *)(v39 + 16) = v32;
        v40 = v97[8];
        *(_QWORD *)(v40 + 16) = sub_73DBF0(91, *v32, v39);
      }
      else
      {
        *(_QWORD *)(v31 + 16) = sub_6F6F40(v106, 0);
      }
      v41 = *(_QWORD *)(v97[8] + 16LL);
      *(_QWORD *)(v41 + 16) = sub_6F6F40(v105, 0);
      return sub_6E70E0(v97, a1);
    }
  }
  v97 = (_QWORD *)sub_726700(29);
  v17 = sub_6F6F40(a1, 0);
  v97[7] = sub_73B8B0(v17, 0);
  if ( a5 || !sub_694FD0(*(_QWORD *)(*(_QWORD *)(v98 + 16) + 120LL), "await_transform", v102) )
    goto LABEL_24;
  sub_6F8E70(*(_QWORD *)(v98 + 16), a2, a2, v108, 0);
  v96 = sub_6E3060(a1);
  sub_7029D0(v108, "await_transform", 0, v96, v108, a1);
  result = sub_6E1990(v96);
  if ( a1[1].m128i_i8[0] )
  {
    result = a1->m128i_i64[0];
    for ( j = *(_BYTE *)(a1->m128i_i64[0] + 140); j == 12; j = *(_BYTE *)(result + 140) )
      result = *(_QWORD *)(result + 160);
    if ( j )
      goto LABEL_24;
  }
  return result;
}
