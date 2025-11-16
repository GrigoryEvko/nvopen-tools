// Function: sub_1BADF80
// Address: 0x1badf80
//
__int64 __fastcall sub_1BADF80(
        __int64 a1,
        __m128i a2,
        __m128i a3,
        double a4,
        double a5,
        double a6,
        double a7,
        double a8,
        __m128 a9)
{
  __int64 v9; // r12
  __int64 v10; // rdi
  _QWORD *v11; // r13
  __int64 v12; // rax
  __int64 v13; // r14
  unsigned __int64 v14; // rax
  _QWORD *v15; // r15
  unsigned __int64 v16; // rax
  _QWORD *v17; // r15
  unsigned __int64 v18; // rax
  __int64 v19; // r13
  __int64 v20; // r15
  _BYTE *v21; // rsi
  __int64 *v22; // r15
  double v23; // xmm4_8
  double v24; // xmm5_8
  double v25; // xmm4_8
  double v26; // xmm5_8
  double v27; // xmm4_8
  double v28; // xmm5_8
  __int64 v29; // r8
  __int64 v30; // rax
  __int64 v31; // r14
  __int64 v32; // rax
  unsigned int v33; // ebx
  unsigned __int64 v34; // r15
  __int64 v35; // rax
  __int64 v36; // r9
  __int64 v37; // rbx
  __int64 v38; // rsi
  __int64 v39; // r15
  unsigned int v40; // esi
  __int64 v41; // rdx
  __int64 v42; // r8
  __int64 v43; // rdi
  unsigned int v44; // eax
  __int64 *v45; // r15
  __int64 v46; // rcx
  __int64 v47; // rax
  __int64 v48; // rax
  unsigned int v49; // eax
  __int64 v50; // rax
  __int64 v51; // rax
  _QWORD *v52; // rax
  __int64 v53; // rsi
  __int64 v54; // rdx
  __int64 v55; // r8
  __int64 v56; // r9
  __int64 v57; // rsi
  unsigned int v58; // edi
  __int64 v59; // rax
  __int64 v60; // rcx
  __int64 v61; // rcx
  __int64 v62; // rax
  __int64 *v63; // r13
  __int64 *v64; // r12
  __int64 v65; // rdi
  __int64 v66; // r14
  __int64 v67; // r15
  int v68; // eax
  __int64 v69; // rax
  int v70; // edx
  __int64 v71; // rdx
  _QWORD *v72; // rax
  __int64 v73; // rdi
  unsigned __int64 v74; // rdx
  __int64 v75; // rdx
  __int64 v76; // rdx
  __int64 v77; // rax
  __int64 v78; // rsi
  __int64 *v79; // r15
  __int64 v80; // rdx
  unsigned __int64 v81; // rax
  __int64 v82; // rax
  unsigned __int64 v83; // rax
  __int64 v84; // r15
  _QWORD *v85; // rax
  _QWORD *v86; // r14
  unsigned __int64 v87; // rax
  double v88; // xmm4_8
  double v89; // xmm5_8
  __int64 v90; // rsi
  __int64 v91; // rax
  __int64 v92; // rdi
  __int64 v93; // rsi
  __int64 v94; // r8
  __int64 v96; // rsi
  unsigned __int8 *v97; // rsi
  int v98; // ecx
  _BYTE *v99; // r9
  size_t v100; // rdx
  int v101; // r11d
  int v102; // eax
  __int64 v103; // rax
  _BYTE *v104; // rsi
  __int64 v105; // [rsp+0h] [rbp-190h]
  __int64 v106; // [rsp+8h] [rbp-188h]
  __int64 v107; // [rsp+10h] [rbp-180h]
  __int64 v108; // [rsp+18h] [rbp-178h]
  __int64 v109; // [rsp+20h] [rbp-170h]
  __int64 *v110; // [rsp+28h] [rbp-168h]
  __int64 v111; // [rsp+30h] [rbp-160h]
  __int64 v112; // [rsp+38h] [rbp-158h]
  __int64 v113; // [rsp+48h] [rbp-148h]
  __int64 v114; // [rsp+50h] [rbp-140h]
  __int64 v115; // [rsp+58h] [rbp-138h]
  _QWORD *v116; // [rsp+60h] [rbp-130h]
  __int64 v117; // [rsp+70h] [rbp-120h]
  unsigned __int64 v118; // [rsp+70h] [rbp-120h]
  __int64 v119; // [rsp+70h] [rbp-120h]
  __int64 v120; // [rsp+70h] [rbp-120h]
  __int64 v121; // [rsp+70h] [rbp-120h]
  __int64 v122; // [rsp+88h] [rbp-108h] BYREF
  __int64 v123[2]; // [rsp+90h] [rbp-100h] BYREF
  char v124; // [rsp+A0h] [rbp-F0h]
  char v125; // [rsp+A1h] [rbp-EFh]
  __m128i v126; // [rsp+B0h] [rbp-E0h] BYREF
  __int64 v127; // [rsp+C0h] [rbp-D0h]
  __int64 v128; // [rsp+C8h] [rbp-C8h]
  __int64 v129; // [rsp+D0h] [rbp-C0h]
  int v130; // [rsp+D8h] [rbp-B8h]
  __int64 v131; // [rsp+E0h] [rbp-B0h]
  __int64 v132; // [rsp+E8h] [rbp-A8h]
  unsigned __int64 v133[2]; // [rsp+100h] [rbp-90h] BYREF
  __int64 v134; // [rsp+110h] [rbp-80h]
  int v135; // [rsp+118h] [rbp-78h]
  __int64 v136; // [rsp+120h] [rbp-70h]
  __int64 v137; // [rsp+128h] [rbp-68h]
  __m128i v138; // [rsp+130h] [rbp-60h] BYREF
  _BYTE v139[80]; // [rsp+140h] [rbp-50h] BYREF

  v9 = a1;
  v10 = *(_QWORD *)(a1 + 8);
  v107 = **(_QWORD **)(v10 + 32);
  v11 = (_QWORD *)sub_13FC520(v10);
  v109 = sub_13FA090(*(_QWORD *)(v9 + 8));
  v12 = *(_QWORD *)(v9 + 448);
  *(_QWORD *)(v9 + 272) = *(_QWORD *)(v12 + 64);
  v13 = *(_QWORD *)(v12 + 368);
  v133[0] = (unsigned __int64)"vector.body";
  LOWORD(v134) = 259;
  v14 = sub_157EBA0((__int64)v11);
  v15 = (_QWORD *)sub_157FBF0(v11, (__int64 *)(v14 + 24), (__int64)v133);
  LOWORD(v134) = 259;
  v133[0] = (unsigned __int64)"middle.block";
  v108 = (__int64)v15;
  v16 = sub_157EBA0((__int64)v15);
  v17 = (_QWORD *)sub_157FBF0(v15, (__int64 *)(v16 + 24), (__int64)v133);
  LOWORD(v134) = 259;
  v133[0] = (unsigned __int64)"scalar.ph";
  v112 = (__int64)v17;
  v18 = sub_157EBA0((__int64)v17);
  v19 = sub_157FBF0(v17, (__int64 *)(v18 + 24), (__int64)v133);
  v110 = sub_194ACF0(*(_QWORD *)(v9 + 24));
  v20 = **(_QWORD **)(v9 + 8);
  if ( v20 )
  {
    v133[0] = (unsigned __int64)v110;
    *v110 = v20;
    v21 = *(_BYTE **)(v20 + 16);
    if ( v21 == *(_BYTE **)(v20 + 24) )
    {
      sub_13FD960(v20 + 8, v21, v133);
    }
    else
    {
      if ( v21 )
      {
        *(_QWORD *)v21 = v133[0];
        v21 = *(_BYTE **)(v20 + 16);
      }
      *(_QWORD *)(v20 + 16) = v21 + 8;
    }
    sub_1400330(v20, v19, *(_QWORD *)(v9 + 24));
    sub_1400330(v20, v112, *(_QWORD *)(v9 + 24));
  }
  else
  {
    v103 = *(_QWORD *)(v9 + 24);
    v133[0] = (unsigned __int64)v110;
    v104 = *(_BYTE **)(v103 + 40);
    if ( v104 == *(_BYTE **)(v103 + 48) )
    {
      sub_13FD960(v103 + 32, v104, v133);
    }
    else
    {
      if ( v104 )
      {
        *(_QWORD *)v104 = v110;
        v104 = *(_BYTE **)(v103 + 40);
      }
      *(_QWORD *)(v103 + 40) = v104 + 8;
    }
  }
  sub_1400330((__int64)v110, v108, *(_QWORD *)(v9 + 24));
  v106 = sub_1B91F20((_QWORD *)v9, (__int64)v110, a2, a3);
  v22 = (__int64 *)sub_15A0680(v13, 0, 0);
  sub_1BADBC0(v9, v110, v19, a2, a3, a4, a5, v23, v24, a8, a9);
  sub_1BAD7F0(v9, v110, v19, (__m128)a2, *(double *)a3.m128i_i64, a4, a5, v25, v26, a8, a9);
  sub_1BAD600(v9, v110, v19, (__m128)a2, *(double *)a3.m128i_i64, a4, a5, v27, v28, a8, a9);
  v116 = (_QWORD *)sub_1B97190(v9, (__int64)v110, a2, a3, a4);
  sub_15A0680(v13, (unsigned int)(*(_DWORD *)(v9 + 92) * *(_DWORD *)(v9 + 88)), 0);
  sub_1B8F7D0(*(_QWORD *)(v9 + 272));
  *(_QWORD *)(v9 + 264) = sub_1B919A0(
                            v9,
                            (__int64)v110,
                            v22,
                            (__int64)v116,
                            v29,
                            *(double *)a2.m128i_i64,
                            *(double *)a3.m128i_i64,
                            a4);
  v30 = *(_QWORD *)(v9 + 448);
  v31 = *(_QWORD *)(v30 + 136);
  v111 = *(_QWORD *)(v30 + 144);
  if ( v111 != v31 )
  {
    v105 = v9 + 472;
    do
    {
      v32 = *(_QWORD *)v31;
      v133[0] = 6;
      v133[1] = 0;
      v122 = v32;
      v134 = *(_QWORD *)(v31 + 24);
      if ( v134 != -8 && v134 != 0 && v134 != -16 )
        sub_1649AC0(v133, *(_QWORD *)(v31 + 8) & 0xFFFFFFFFFFFFFFF8LL);
      v135 = *(_DWORD *)(v31 + 32);
      v136 = *(_QWORD *)(v31 + 40);
      v137 = *(_QWORD *)(v31 + 48);
      v138.m128i_i64[0] = (__int64)v139;
      v138.m128i_i64[1] = 0x200000000LL;
      v33 = *(_DWORD *)(v31 + 64);
      if ( v33 && &v138 != (__m128i *)(v31 + 56) )
      {
        v99 = v139;
        v100 = 8LL * v33;
        if ( v33 <= 2
          || (sub_16CD150((__int64)&v138, v139, v33, 8, v33, (int)v139),
              v99 = (_BYTE *)v138.m128i_i64[0],
              (v100 = 8LL * *(unsigned int *)(v31 + 64)) != 0) )
        {
          memcpy(v99, *(const void **)(v31 + 56), v100);
        }
        v138.m128i_i32[2] = v33;
      }
      v34 = sub_157EBA0(v19);
      LOWORD(v127) = 259;
      v126.m128i_i64[0] = (__int64)"bc.resume.val";
      v117 = *(_QWORD *)v122;
      v35 = sub_1648B60(64);
      v37 = v35;
      if ( v35 )
      {
        sub_15F1EA0(v35, v117, 53, 0, 0, v34);
        *(_DWORD *)(v37 + 56) = 3;
        sub_164B780(v37, v126.m128i_i64);
        sub_1648880(v37, *(_DWORD *)(v37 + 56), 1);
      }
      v38 = *(_QWORD *)(v122 + 48);
      v126.m128i_i64[0] = v38;
      if ( v38 )
      {
        v39 = v37 + 48;
        sub_1623A60((__int64)&v126, v38, 2);
        if ( (__m128i *)(v37 + 48) == &v126 )
        {
          if ( v126.m128i_i64[0] )
            sub_161E7C0((__int64)&v126, v126.m128i_i64[0]);
LABEL_19:
          v40 = *(_DWORD *)(v9 + 496);
          if ( !v40 )
            goto LABEL_78;
          goto LABEL_20;
        }
      }
      else
      {
        v39 = v37 + 48;
        if ( (__m128i *)(v37 + 48) == &v126 )
          goto LABEL_19;
      }
      v96 = *(_QWORD *)(v37 + 48);
      if ( v96 )
        sub_161E7C0(v39, v96);
      v97 = (unsigned __int8 *)v126.m128i_i64[0];
      *(_QWORD *)(v37 + 48) = v126.m128i_i64[0];
      if ( !v97 )
        goto LABEL_19;
      sub_1623210((__int64)&v126, v97, v39);
      v40 = *(_DWORD *)(v9 + 496);
      if ( !v40 )
      {
LABEL_78:
        ++*(_QWORD *)(v9 + 472);
        goto LABEL_79;
      }
LABEL_20:
      v41 = v122;
      v42 = v40 - 1;
      v43 = *(_QWORD *)(v9 + 480);
      v44 = v42 & (((unsigned int)v122 >> 9) ^ ((unsigned int)v122 >> 4));
      v45 = (__int64 *)(v43 + 16LL * v44);
      v46 = *v45;
      if ( v122 != *v45 )
      {
        v101 = 1;
        v36 = 0;
        while ( v46 != -8 )
        {
          if ( v46 == -16 && !v36 )
            v36 = (__int64)v45;
          v44 = v42 & (v101 + v44);
          v45 = (__int64 *)(v43 + 16LL * v44);
          v46 = *v45;
          if ( v122 == *v45 )
            goto LABEL_21;
          ++v101;
        }
        v102 = *(_DWORD *)(v9 + 488);
        if ( v36 )
          v45 = (__int64 *)v36;
        ++*(_QWORD *)(v9 + 472);
        v98 = v102 + 1;
        if ( 4 * (v102 + 1) >= 3 * v40 )
        {
LABEL_79:
          v40 *= 2;
        }
        else if ( v40 - *(_DWORD *)(v9 + 492) - v98 > v40 >> 3 )
        {
          goto LABEL_96;
        }
        sub_1BA3880(v105, v40);
        sub_1BA0D30(v105, &v122, &v126);
        v45 = (__int64 *)v126.m128i_i64[0];
        v41 = v122;
        v98 = *(_DWORD *)(v9 + 488) + 1;
LABEL_96:
        *(_DWORD *)(v9 + 488) = v98;
        if ( *v45 != -8 )
          --*(_DWORD *)(v9 + 492);
        *v45 = v41;
        v46 = v122;
        v45[1] = 0;
      }
LABEL_21:
      if ( *(_QWORD *)(v9 + 272) == v46 )
      {
        v45[1] = (__int64)v116;
        v53 = (__int64)v116;
      }
      else
      {
        v47 = sub_13FC520((__int64)v110);
        v118 = sub_157EBA0(v47);
        v48 = sub_16498A0(v118);
        v126 = 0u;
        v128 = v48;
        v127 = 0;
        v129 = 0;
        v130 = 0;
        v131 = 0;
        v132 = 0;
        sub_17050D0(v126.m128i_i64, v118);
        v119 = sub_1456040(v136);
        v49 = sub_15FBEB0(v116, 1, v119, 1);
        v125 = 1;
        v123[0] = (__int64)"cast.crd";
        v124 = 3;
        v120 = sub_12AA3B0(v126.m128i_i64, v49, (__int64)v116, v119, (__int64)v123);
        v50 = sub_157EB90(**(_QWORD **)(*(_QWORD *)(v9 + 8) + 32LL));
        v51 = sub_1632FA0(v50);
        v52 = sub_1B19340(
                (__int64)v133,
                (__int64)&v126,
                v120,
                *(_QWORD **)(*(_QWORD *)(v9 + 16) + 112LL),
                v51,
                a2,
                a3,
                a4);
        v45[1] = (__int64)v52;
        v125 = 1;
        v123[0] = (__int64)"ind.end";
        v124 = 3;
        sub_164B780((__int64)v52, v123);
        if ( v126.m128i_i64[0] )
          sub_161E7C0((__int64)&v126, v126.m128i_i64[0]);
        v53 = v45[1];
      }
      sub_1704F80(v37, v53, v112, v46, v42, v36);
      v57 = v122;
      v121 = 0x17FFFFFFE8LL;
      v58 = *(_DWORD *)(v122 + 20) & 0xFFFFFFF;
      if ( v58 )
      {
        v55 = *(_BYTE *)(v122 + 23) & 0x40;
        v56 = v122 - 24LL * v58;
        v54 = 24LL * *(unsigned int *)(v122 + 56) + 8;
        v59 = 0;
        do
        {
          v60 = v122 - 24LL * v58;
          if ( (_BYTE)v55 )
            v60 = *(_QWORD *)(v122 - 8);
          if ( v19 == *(_QWORD *)(v60 + v54) )
          {
            v121 = 24 * v59;
            goto LABEL_32;
          }
          ++v59;
          v54 += 8;
        }
        while ( v58 != (_DWORD)v59 );
        v121 = 0x17FFFFFFE8LL;
      }
LABEL_32:
      v61 = *(_QWORD *)(v9 + 216);
      v62 = *(unsigned int *)(v9 + 224);
      if ( v61 != v61 + 8 * v62 )
      {
        v115 = v19;
        v63 = *(__int64 **)(v9 + 216);
        v113 = v9;
        v64 = (__int64 *)(v61 + 8 * v62);
        v114 = v31;
        do
        {
          v66 = *v63;
          v67 = v134;
          v68 = *(_DWORD *)(v37 + 20) & 0xFFFFFFF;
          if ( v68 == *(_DWORD *)(v37 + 56) )
          {
            sub_15F55D0(v37, v57, v54, v61, v55, v56);
            v68 = *(_DWORD *)(v37 + 20) & 0xFFFFFFF;
          }
          v69 = (v68 + 1) & 0xFFFFFFF;
          v70 = v69 | *(_DWORD *)(v37 + 20) & 0xF0000000;
          *(_DWORD *)(v37 + 20) = v70;
          if ( (v70 & 0x40000000) != 0 )
            v71 = *(_QWORD *)(v37 - 8);
          else
            v71 = v37 - 24 * v69;
          v72 = (_QWORD *)(v71 + 24LL * (unsigned int)(v69 - 1));
          if ( *v72 )
          {
            v73 = v72[1];
            v74 = v72[2] & 0xFFFFFFFFFFFFFFFCLL;
            *(_QWORD *)v74 = v73;
            if ( v73 )
            {
              v55 = *(_QWORD *)(v73 + 16) & 3LL;
              *(_QWORD *)(v73 + 16) = v55 | v74;
            }
          }
          *v72 = v67;
          if ( v67 )
          {
            v75 = *(_QWORD *)(v67 + 8);
            v55 = v67 + 8;
            v72[1] = v75;
            if ( v75 )
            {
              v56 = (__int64)(v72 + 1);
              *(_QWORD *)(v75 + 16) = (unsigned __int64)(v72 + 1) | *(_QWORD *)(v75 + 16) & 3LL;
            }
            v72[2] = v55 | v72[2] & 3LL;
            *(_QWORD *)(v67 + 8) = v72;
          }
          v76 = *(_DWORD *)(v37 + 20) & 0xFFFFFFF;
          v77 = (unsigned int)(v76 - 1);
          if ( (*(_BYTE *)(v37 + 23) & 0x40) != 0 )
            v65 = *(_QWORD *)(v37 - 8);
          else
            v65 = v37 - 24 * v76;
          ++v63;
          v54 = 3LL * *(unsigned int *)(v37 + 56);
          *(_QWORD *)(v65 + 8 * v77 + 24LL * *(unsigned int *)(v37 + 56) + 8) = v66;
        }
        while ( v64 != v63 );
        v19 = v115;
        v31 = v114;
        v9 = v113;
        v57 = v122;
      }
      if ( (*(_BYTE *)(v57 + 23) & 0x40) != 0 )
        v78 = *(_QWORD *)(v57 - 8);
      else
        v78 = v57 - 24LL * (*(_DWORD *)(v57 + 20) & 0xFFFFFFF);
      v79 = (__int64 *)(v78 + v121);
      if ( *(_QWORD *)(v78 + v121) )
      {
        v80 = v79[1];
        v81 = v79[2] & 0xFFFFFFFFFFFFFFFCLL;
        *(_QWORD *)v81 = v80;
        if ( v80 )
          *(_QWORD *)(v80 + 16) = *(_QWORD *)(v80 + 16) & 3LL | v81;
      }
      *v79 = v37;
      if ( v37 )
      {
        v82 = *(_QWORD *)(v37 + 8);
        v79[1] = v82;
        if ( v82 )
          *(_QWORD *)(v82 + 16) = (unsigned __int64)(v79 + 1) | *(_QWORD *)(v82 + 16) & 3LL;
        v79[2] = (v37 + 8) | v79[2] & 3;
        *(_QWORD *)(v37 + 8) = v79;
      }
      if ( (_BYTE *)v138.m128i_i64[0] != v139 )
        _libc_free(v138.m128i_u64[0]);
      if ( v134 != -8 && v134 != 0 && v134 != -16 )
        sub_1649B30(v133);
      v31 += 88;
    }
    while ( v111 != v31 );
  }
  v83 = sub_157EBA0(v112);
  LOWORD(v134) = 259;
  v133[0] = (unsigned __int64)"cmp.n";
  v84 = sub_15FEEB0(51, 32, v106, (__int64)v116, (__int64)v133, v83);
  v85 = sub_1648A60(56, 3u);
  v86 = v85;
  if ( v85 )
    sub_15F83E0((__int64)v85, v109, v19, v84, 0);
  v87 = sub_157EBA0(v112);
  sub_1AA6530(v87, v86, (__m128)a2, *(double *)a3.m128i_i64, a4, a5, v88, v89, a8, a9);
  v90 = sub_157EE30(v108);
  if ( v90 )
    v90 -= 24;
  sub_17050D0((__int64 *)(v9 + 96), v90);
  v91 = sub_13FC520((__int64)v110);
  *(_QWORD *)(v9 + 176) = v19;
  v92 = *(_QWORD *)(v9 + 8);
  *(_QWORD *)(v9 + 168) = v91;
  *(_QWORD *)(v9 + 184) = v112;
  *(_QWORD *)(v9 + 192) = v109;
  *(_QWORD *)(v9 + 200) = v108;
  *(_QWORD *)(v9 + 208) = v107;
  v93 = sub_13FD000(v92);
  if ( v93 )
    sub_13FCC30((__int64)v110, v93);
  sub_1BF1BF0(v133, v110, 1, *(_QWORD *)(v9 + 72), v94);
  v138.m128i_i32[2] = 1;
  v126 = _mm_loadu_si128(&v138);
  sub_1BF1E00(v133, &v126, 1);
  return *(_QWORD *)(v9 + 168);
}
