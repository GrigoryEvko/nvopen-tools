// Function: sub_1E4CC00
// Address: 0x1e4cc00
//
__int64 __fastcall sub_1E4CC00(
        _QWORD *a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int64 a7,
        __int64 a8,
        int a9,
        unsigned int a10,
        char a11)
{
  __int64 v12; // rdi
  __int64 result; // rax
  __int64 v14; // rsi
  int v15; // r8d
  __int64 v16; // r9
  char v17; // di
  unsigned int v18; // r10d
  unsigned int v19; // r11d
  unsigned int i; // eax
  unsigned __int64 v21; // rax
  __int64 v22; // rax
  unsigned __int64 v23; // rax
  __int64 v24; // rsi
  __int64 v25; // rax
  __int64 v26; // rcx
  __int64 v27; // rdx
  _DWORD *v28; // rax
  _DWORD *v29; // rbx
  unsigned int v30; // eax
  int v31; // esi
  bool v32; // al
  int v33; // ecx
  bool v34; // zf
  int v35; // eax
  int v36; // ebx
  bool v37; // r12
  bool v38; // al
  unsigned int v39; // ecx
  char v40; // al
  __int64 v41; // r12
  __int64 v42; // rax
  __int64 v43; // r13
  __int64 v44; // r12
  __int64 v45; // rcx
  __int64 v46; // rsi
  unsigned int v47; // r13d
  int v48; // r9d
  int v49; // eax
  __int64 v50; // rdi
  int v51; // edx
  int v52; // r8d
  unsigned int v53; // eax
  int v54; // esi
  _DWORD *v55; // rax
  __int32 v56; // eax
  __int64 v57; // rsi
  __int64 v58; // rax
  __int64 v59; // r13
  __int64 v60; // rsi
  __int64 v61; // rax
  __int64 v62; // rcx
  __int64 v63; // rdx
  int v64; // ecx
  int v65; // r12d
  unsigned int v66; // esi
  __int64 v67; // r8
  unsigned int v68; // ecx
  __int64 *v69; // rdx
  __int64 v70; // rdi
  __int64 v71; // rdi
  __int64 v72; // r13
  __int64 v73; // rbx
  int v74; // ecx
  __int64 v75; // rsi
  __int64 v76; // rax
  unsigned int v77; // eax
  int v78; // ebx
  int v79; // eax
  __int64 v80; // r13
  unsigned __int64 v81; // rax
  int v82; // eax
  __int64 v83; // rdi
  int v84; // esi
  int v85; // r12d
  __int64 v86; // rax
  __int64 v87; // rsi
  __int64 v88; // rdi
  unsigned __int64 v89; // rax
  int v90; // eax
  unsigned int v91; // eax
  _DWORD *v92; // rax
  char v93; // al
  __int64 v94; // rdi
  unsigned int v95; // r13d
  int v96; // r9d
  __int64 v97; // r12
  _DWORD *v98; // rax
  __int64 v99; // rbx
  __int64 v100; // r13
  __int64 v101; // r12
  _DWORD *v102; // rax
  int v103; // r9d
  int v104; // r13d
  __int64 *v105; // r10
  int v106; // edi
  __int64 v107; // rdi
  unsigned int v108; // [rsp+14h] [rbp-15Ch]
  __int64 v109; // [rsp+20h] [rbp-150h]
  __int64 v110; // [rsp+28h] [rbp-148h]
  __int64 v111; // [rsp+30h] [rbp-140h]
  __int64 v112; // [rsp+38h] [rbp-138h]
  char v113; // [rsp+43h] [rbp-12Dh]
  unsigned int v114; // [rsp+44h] [rbp-12Ch]
  __int64 v115; // [rsp+48h] [rbp-128h]
  int v116; // [rsp+50h] [rbp-120h]
  unsigned int v117; // [rsp+54h] [rbp-11Ch]
  __int64 v119; // [rsp+60h] [rbp-110h]
  unsigned int v122; // [rsp+78h] [rbp-F8h]
  int v123; // [rsp+80h] [rbp-F0h]
  unsigned int v124; // [rsp+84h] [rbp-ECh]
  unsigned int v125; // [rsp+90h] [rbp-E0h]
  int v126; // [rsp+94h] [rbp-DCh]
  unsigned int v127; // [rsp+A0h] [rbp-D0h]
  unsigned int v128; // [rsp+A4h] [rbp-CCh]
  __int64 *v129; // [rsp+A8h] [rbp-C8h]
  __int64 v130; // [rsp+A8h] [rbp-C8h]
  __int64 v131; // [rsp+A8h] [rbp-C8h]
  int v132; // [rsp+A8h] [rbp-C8h]
  __int64 v133; // [rsp+B0h] [rbp-C0h]
  unsigned int v134; // [rsp+B0h] [rbp-C0h]
  __int64 v135; // [rsp+C0h] [rbp-B0h]
  int v137; // [rsp+D0h] [rbp-A0h]
  unsigned int v138; // [rsp+D4h] [rbp-9Ch]
  __int32 v139; // [rsp+D8h] [rbp-98h]
  __int32 v141; // [rsp+E8h] [rbp-88h]
  int v142; // [rsp+E8h] [rbp-88h]
  unsigned int v143; // [rsp+F4h] [rbp-7Ch] BYREF
  unsigned int v144; // [rsp+F8h] [rbp-78h] BYREF
  unsigned int v145; // [rsp+FCh] [rbp-74h] BYREF
  __int64 v146; // [rsp+100h] [rbp-70h] BYREF
  __int64 v147; // [rsp+108h] [rbp-68h] BYREF
  __m128i v148; // [rsp+110h] [rbp-60h] BYREF
  __int64 v149; // [rsp+120h] [rbp-50h]
  __int64 v150; // [rsp+128h] [rbp-48h]
  __int64 v151; // [rsp+130h] [rbp-40h]

  if ( a9 == a10 )
  {
    v138 = a9 - 1;
    v127 = a9;
  }
  else
  {
    v138 = 2 * a9 - a10;
    v127 = a10 - 1;
  }
  v12 = a1[115];
  v146 = *(_QWORD *)(v12 + 32);
  v110 = sub_1DD5D10(v12);
  result = v146;
  if ( v146 != v110 )
  {
    v109 = a7 + 32LL * (unsigned int)a9;
    v108 = v138 + 2;
    v112 = a7 + 32LL * a10;
    v119 = a6 + 88;
    do
    {
      v14 = *(_QWORD *)(result + 32);
      v15 = *(_DWORD *)(result + 40);
      v144 = 0;
      v16 = a1[115];
      v143 = *(_DWORD *)(v14 + 8);
      if ( v15 == 1 )
      {
        v139 = 0;
        v122 = 0;
      }
      else
      {
        v17 = 0;
        v18 = 0;
        v19 = 0;
        for ( i = 1; i != v15; i += 2 )
        {
          if ( v16 == *(_QWORD *)(v14 + 40LL * (i + 1) + 24) )
            v17 = 1;
          else
            v19 = *(_DWORD *)(v14 + 40LL * i + 8);
          if ( v16 == *(_QWORD *)(v14 + 40LL * (i + 1) + 24) )
            v18 = *(_DWORD *)(v14 + 40LL * i + 8);
        }
        v139 = v18;
        v122 = v19;
        if ( v17 )
          v144 = v18;
        else
          v139 = 0;
      }
      v145 = 0;
      if ( (unsigned __int8)sub_1932870(v109, (int *)&v144, &v148) )
        v139 = sub_1E49390(v109, (int *)&v144)[1];
      v21 = sub_1E45EB0((__int64)a1, v146);
      v126 = sub_1E404B0(a6, v21);
      v22 = sub_1E69D00(a1[5], v144);
      v23 = sub_1E45EB0((__int64)a1, v22);
      v24 = a6 + 88;
      v123 = sub_1E404B0(a6, v23);
      v25 = *(_QWORD *)(a6 + 96);
      LODWORD(v147) = v143;
      if ( !v25 )
        goto LABEL_24;
      do
      {
        while ( 1 )
        {
          v26 = *(_QWORD *)(v25 + 16);
          v27 = *(_QWORD *)(v25 + 24);
          if ( v143 <= *(_DWORD *)(v25 + 32) )
            break;
          v25 = *(_QWORD *)(v25 + 24);
          if ( !v27 )
            goto LABEL_22;
        }
        v24 = v25;
        v25 = *(_QWORD *)(v25 + 16);
      }
      while ( v26 );
LABEL_22:
      if ( v24 == v119 || v143 < *(_DWORD *)(v24 + 32) )
      {
LABEL_24:
        v148.m128i_i64[0] = (__int64)&v147;
        v24 = sub_1E48710((_QWORD *)(a6 + 80), v24, (unsigned int **)&v148);
      }
      v113 = 0;
      v125 = *(_DWORD *)(v24 + 36);
      if ( a10 <= (*(_DWORD *)(a6 + 132) - *(_DWORD *)(a6 + 128)) / *(_DWORD *)(a6 + 136) )
      {
        if ( v125 )
          goto LABEL_31;
      }
      else
      {
        if ( *(_DWORD *)(v24 + 36) )
          goto LABEL_31;
        v125 = 1;
        if ( *(_BYTE *)(v24 + 40) )
          goto LABEL_31;
      }
      v28 = sub_1E49390(a7 + 32LL * v127, (int *)&v144);
      sub_1E46110((__int64)a1, a2, a6, a8, a10, 0, v146, v143, v122, v28[1]);
      if ( (unsigned __int8)sub_1932870(v112, (int *)&v144, &v148) )
      {
        v29 = sub_1E49390(v112, (int *)&v144);
        sub_1E49390(v112, (int *)&v143)[1] = v29[1];
      }
      v125 = 0;
      v113 = a11;
LABEL_31:
      v30 = v138 + 2;
      if ( v123 >= (int)v138 && a9 != a10 )
      {
        v30 = v108 - v123;
        if ( (int)(v108 - v123) <= 0 )
          v30 = 1;
      }
      if ( v30 > v125 )
        v30 = v125;
      v128 = v30;
      if ( v123 == -1 )
      {
        v31 = v126;
        v32 = a9 != a10 && v126 >= -1;
        if ( v32 )
          goto LABEL_159;
      }
      else
      {
        v31 = v123;
        if ( v126 >= v123 && a9 != a10 )
        {
LABEL_159:
          v33 = v31 == 0 && v128 == 1;
          v32 = v123 != -1;
          goto LABEL_42;
        }
        v32 = a9 == a10 && v126 > v123;
        if ( v32 )
        {
          v33 = v126 - v123;
          goto LABEL_42;
        }
        v32 = 1;
      }
      v33 = 0;
LABEL_42:
      if ( v128 )
      {
        v34 = !v32 || v126 <= v123;
        v35 = 0;
        if ( !v34 )
          v35 = v126 - v123;
        v117 = v123 + v35;
        v114 = v127 - v35;
        v111 = a7 + 32LL * (v127 - v35);
        v36 = 0;
        v124 = v138 - v33;
        v135 = a7 + 32LL * a10;
        v116 = v33 + v31;
        while ( 1 )
        {
          if ( v138 >= v36 && v126 < a9 && v138 >= v36 + v116 )
          {
            if ( (unsigned __int8)sub_1932870(a7 + 32LL * v124, (int *)&v144, &v148) )
            {
              v92 = sub_1E49390(a7 + 32LL * v124, (int *)&v144);
              v57 = (unsigned int)v92[1];
              v137 = v36 + 1;
              v145 = v92[1];
            }
            else
            {
              v71 = a1[5];
              v145 = v144;
              v142 = 1;
              v72 = sub_1E69D00(v71, v144);
              v137 = v36 + 1;
              v134 = v36 + 1;
              if ( v72 )
              {
                v132 = v36;
                v73 = v72;
                while ( 1 )
                {
                  if ( **(_WORD **)(v73 + 16) != 45 && **(_WORD **)(v73 + 16)
                    || (v80 = a1[115], v80 != *(_QWORD *)(v73 + 24)) )
                  {
LABEL_116:
                    v36 = v132;
                    goto LABEL_117;
                  }
                  v81 = sub_1E45EB0((__int64)a1, v73);
                  v82 = sub_1E404B0(a6, v81);
                  v83 = *(_QWORD *)(v73 + 32);
                  v84 = *(_DWORD *)(v73 + 40);
                  v85 = v82;
                  if ( v82 + v142 <= (int)v124 )
                  {
                    v87 = (unsigned int)sub_1E40FE0(v83, v84, v80);
                  }
                  else if ( v84 == 1 )
                  {
LABEL_164:
                    v87 = 0;
                  }
                  else
                  {
                    v86 = 1;
                    while ( v80 == *(_QWORD *)(v83 + 40LL * (unsigned int)(v86 + 1) + 24) )
                    {
                      v86 = (unsigned int)(v86 + 2);
                      if ( (_DWORD)v86 == v84 )
                        goto LABEL_164;
                    }
                    v87 = *(unsigned int *)(v83 + 40 * v86 + 8);
                  }
                  v88 = a1[5];
                  v145 = v87;
                  v73 = sub_1E69D00(v88, v87);
                  v89 = sub_1E45EB0((__int64)a1, v73);
                  v90 = sub_1E404B0(a6, v89);
                  if ( v90 != -1 )
                  {
                    v91 = v138 - (v85 - v90);
                    if ( v91 >= v134 )
                    {
                      v97 = a7 + 32LL * (v91 - v134);
                      if ( (unsigned __int8)sub_1932870(v97, (int *)&v145, &v148) )
                        break;
                    }
                  }
                  ++v142;
                  ++v134;
                  if ( !v73 )
                    goto LABEL_116;
                }
                v36 = v132;
                v98 = sub_1E49390(v97, (int *)&v145);
                v57 = (unsigned int)v98[1];
                v145 = v98[1];
              }
              else
              {
LABEL_117:
                v57 = v145;
              }
            }
          }
          else
          {
            v145 = v122;
            v57 = v122;
            v137 = v36 + 1;
          }
          v58 = sub_1E69D00(a1[5], v57);
          if ( v58 && (**(_WORD **)(v58 + 16) == 45 || !**(_WORD **)(v58 + 16)) && a5 == *(_QWORD *)(v58 + 24) )
          {
            v74 = *(_DWORD *)(v58 + 40);
            v75 = *(_QWORD *)(v58 + 32);
            if ( v74 == 1 )
            {
LABEL_155:
              v77 = 0;
            }
            else
            {
              v76 = 1;
              while ( a5 == *(_QWORD *)(v75 + 40LL * (unsigned int)(v76 + 1) + 24) )
              {
                v76 = (unsigned int)(v76 + 2);
                if ( v74 == (_DWORD)v76 )
                  goto LABEL_155;
              }
              v77 = *(_DWORD *)(v75 + 40 * v76 + 8);
            }
            v145 = v77;
          }
          v59 = sub_1E69D00(a1[5], v144);
          v133 = a7 + 32LL * (a10 - v36);
          if ( v59 )
          {
            v37 = **(_WORD **)(v59 + 16) == 45 || **(_WORD **)(v59 + 16) == 0;
            if ( a9 == a10 )
              goto LABEL_53;
            v38 = v127 == a9;
            if ( v36 )
            {
LABEL_81:
              if ( v38 )
              {
                v130 = a7 + 32LL * (v127 + 1 - v36);
                if ( (unsigned __int8)sub_1932870(v130, (int *)&v143, &v148) )
                  goto LABEL_83;
              }
              goto LABEL_50;
            }
          }
          else
          {
            v37 = 0;
            if ( a9 == a10 )
              goto LABEL_55;
            v38 = v127 == a9;
            if ( v36 )
              goto LABEL_81;
          }
          if ( v38 && v123 | v126 && (unsigned __int8)sub_1932870(v111, (int *)&v144, &v148) )
          {
            v139 = sub_1E49390(v111, (int *)&v144)[1];
LABEL_53:
            if ( !v37 )
              goto LABEL_55;
            goto LABEL_54;
          }
LABEL_50:
          v39 = v127 - v36;
          if ( v117 > v138 + 1 )
          {
            v115 = a7 + 32LL * (v114 - v36);
            v40 = sub_1932870(v115, (int *)&v144, &v148);
            v39 = v127 - v36;
            if ( v40 )
            {
              v139 = sub_1E49390(v115, (int *)&v144)[1];
              goto LABEL_53;
            }
          }
          v130 = a7 + 32LL * v39;
          if ( !(unsigned __int8)sub_1932870(v130, (int *)&v143, &v148) )
            goto LABEL_53;
          if ( !v37 || v127 != a9 )
          {
LABEL_83:
            v139 = sub_1E49390(v130, (int *)&v143)[1];
            goto LABEL_53;
          }
LABEL_54:
          if ( v126 <= (int)(v138 - v36) )
          {
            v60 = a6 + 88;
            v61 = *(_QWORD *)(a6 + 96);
            LODWORD(v147) = v144;
            if ( !v61 )
              goto LABEL_99;
            do
            {
              while ( 1 )
              {
                v62 = *(_QWORD *)(v61 + 16);
                v63 = *(_QWORD *)(v61 + 24);
                if ( v144 <= *(_DWORD *)(v61 + 32) )
                  break;
                v61 = *(_QWORD *)(v61 + 24);
                if ( !v63 )
                  goto LABEL_97;
              }
              v60 = v61;
              v61 = *(_QWORD *)(v61 + 16);
            }
            while ( v62 );
LABEL_97:
            if ( v60 == v119 || v144 < *(_DWORD *)(v60 + 32) )
            {
LABEL_99:
              v148.m128i_i64[0] = (__int64)&v147;
              v60 = sub_1E48710((_QWORD *)(a6 + 80), v60, (unsigned int **)&v148);
            }
            v64 = v126 - v123;
            v65 = *(_DWORD *)(v60 + 36) - (*(_BYTE *)(v60 + 40) == 0) - (v126 - v123);
            if ( v65 > v36 && (v93 = sub_1932870(v112, (int *)&v144, &v148), v64 = v126 - v123, v93) )
            {
              if ( **(_WORD **)(v59 + 16) == 45 || (v94 = v135, !**(_WORD **)(v59 + 16)) )
              {
                if ( (unsigned __int8)sub_1E45F30(a6, (__int64)a1, v59) )
                  v94 = 32LL * (a10 - v36 - v65) + a7;
                else
                  v94 = v135;
              }
              if ( (unsigned __int8)sub_1932870(v94, (int *)&v144, &v148) )
              {
                v95 = sub_1E49390(v94, (int *)&v144)[1];
                sub_1E46110((__int64)a1, a2, a6, a8, a10, v36, v146, v143, v95, 0);
                v141 = v95;
                sub_1E49390(v135, (int *)&v143)[1] = v95;
                if ( (unsigned __int8)sub_1932870(a7 + 32LL * (unsigned int)(a9 - 1 - v36), (int *)&v144, &v148) )
                  v139 = sub_1E49390(a7 + 32LL * (unsigned int)(a9 - 1 - v36), (int *)&v144)[1];
                else
                  v139 = v95;
                if ( a11 && v128 - 1 == v36 )
                  sub_1E42770(v143, v95, a1[115], a1[5], a1[266], v96);
                goto LABEL_68;
              }
            }
            else if ( v64 > 0
                   && a9 == a10
                   && (unsigned __int8)sub_1932870(a7 + 32LL * (a10 + v123 - v126 - v36), (int *)&v144, &v148) )
            {
              v139 = sub_1E49390(a7 + 32LL * (a10 + v123 - v126 - v36), (int *)&v144)[1];
            }
          }
LABEL_55:
          v141 = sub_1E6B9A0(
                   a1[5],
                   *(_QWORD *)(*(_QWORD *)(a1[5] + 24LL) + 16LL * (v143 & 0x7FFFFFFF)) & 0xFFFFFFFFFFFFFFF8LL,
                   byte_3F871B3,
                   0);
          v41 = *(_QWORD *)(a1[2] + 8LL);
          v147 = 0;
          v42 = sub_1DD5D10(a2);
          v43 = *(_QWORD *)(a2 + 56);
          v129 = (__int64 *)v42;
          v44 = (__int64)sub_1E0B640(v43, v41, &v147, 0);
          sub_1DD5BA0((__int64 *)(a2 + 16), v44);
          v45 = *v129;
          v46 = *(_QWORD *)v44 & 7LL;
          *(_QWORD *)(v44 + 8) = v129;
          v45 &= 0xFFFFFFFFFFFFFFF8LL;
          *(_QWORD *)v44 = v45 | v46;
          *(_QWORD *)(v45 + 8) = v44;
          *v129 = v44 | *v129 & 7;
          v148.m128i_i64[0] = 0x10000000;
          v149 = 0;
          v148.m128i_i32[2] = v141;
          v150 = 0;
          v151 = 0;
          sub_1E1A9C0(v44, v43, &v148);
          if ( v147 )
            sub_161E7C0((__int64)&v147, v147);
          v148.m128i_i64[0] = 0;
          v149 = 0;
          v148.m128i_i32[2] = v145;
          v150 = 0;
          v151 = 0;
          sub_1E1A9C0(v44, v43, &v148);
          v148.m128i_i8[0] = 4;
          v149 = 0;
          v148.m128i_i32[0] &= 0xFFF000FF;
          v150 = a3;
          sub_1E1A9C0(v44, v43, &v148);
          v148.m128i_i64[0] = 0;
          v149 = 0;
          v148.m128i_i32[2] = v139;
          v150 = 0;
          v151 = 0;
          sub_1E1A9C0(v44, v43, &v148);
          v148.m128i_i8[0] = 4;
          v149 = 0;
          v148.m128i_i32[0] &= 0xFFF000FF;
          v150 = a4;
          sub_1E1A9C0(v44, v43, &v148);
          if ( v36 )
          {
            v47 = 0;
            if ( a9 == a10 )
              goto LABEL_108;
            goto LABEL_59;
          }
          v147 = v44;
          v66 = *(_DWORD *)(a8 + 24);
          if ( !v66 )
          {
            ++*(_QWORD *)a8;
LABEL_180:
            v107 = a8;
            v66 *= 2;
LABEL_181:
            sub_1E4BEC0(v107, v66);
            sub_1E48B10(a8, &v147, &v148);
            v69 = (__int64 *)v148.m128i_i64[0];
            v44 = v147;
            v106 = *(_DWORD *)(a8 + 16) + 1;
            goto LABEL_175;
          }
          v67 = *(_QWORD *)(a8 + 8);
          v68 = (v66 - 1) & (((unsigned int)v44 >> 9) ^ ((unsigned int)v44 >> 4));
          v69 = (__int64 *)(v67 + 16LL * v68);
          v70 = *v69;
          if ( v44 == *v69 )
            goto LABEL_107;
          v104 = 1;
          v105 = 0;
          while ( v70 != -8 )
          {
            if ( !v105 && v70 == -16 )
              v105 = v69;
            v68 = (v66 - 1) & (v104 + v68);
            v69 = (__int64 *)(v67 + 16LL * v68);
            v70 = *v69;
            if ( v44 == *v69 )
              goto LABEL_107;
            ++v104;
          }
          if ( v105 )
            v69 = v105;
          ++*(_QWORD *)a8;
          v106 = *(_DWORD *)(a8 + 16) + 1;
          if ( 4 * v106 >= 3 * v66 )
            goto LABEL_180;
          if ( v66 - *(_DWORD *)(a8 + 20) - v106 <= v66 >> 3 )
          {
            v107 = a8;
            goto LABEL_181;
          }
LABEL_175:
          *(_DWORD *)(a8 + 16) = v106;
          if ( *v69 != -8 )
            --*(_DWORD *)(a8 + 20);
          *v69 = v44;
          v69[1] = 0;
LABEL_107:
          v47 = 0;
          v69[1] = v146;
          if ( a9 == a10 )
          {
LABEL_108:
            v131 = a7 + 32LL * (v127 - v36);
            if ( (unsigned __int8)sub_1932870(v131, (int *)&v144, &v148) )
              v47 = sub_1E49390(v131, (int *)&v144)[1];
          }
LABEL_59:
          sub_1E46110((__int64)a1, a2, a6, a8, a10, v36, v146, v143, v141, v47);
          v49 = *(_DWORD *)(v133 + 24);
          if ( v49 )
          {
            v50 = *(_QWORD *)(v133 + 8);
            v51 = v49 - 1;
            v52 = 1;
            v53 = (v49 - 1) & (37 * v143);
            v54 = *(_DWORD *)(v50 + 8LL * v53);
            if ( v143 == v54 )
            {
LABEL_61:
              v55 = sub_1E49390(v135, (int *)&v143);
              sub_1E46110((__int64)a1, a2, a6, a8, a10, v36, v146, v55[1], v141, 0);
            }
            else
            {
              while ( v54 != -1 )
              {
                v48 = v52 + 1;
                v53 = v51 & (v52 + v53);
                v54 = *(_DWORD *)(v50 + 8LL * v53);
                if ( v143 == v54 )
                  goto LABEL_61;
                ++v52;
              }
            }
          }
          if ( a11 && v128 - 1 == v36 )
            sub_1E42770(v143, v141, a1[115], a1[5], a1[266], v48);
          v56 = v139;
          if ( a9 == a10 )
            v56 = v141;
          v139 = v56;
          sub_1E49390(v135, (int *)&v143)[1] = v141;
LABEL_68:
          --v124;
          v36 = v137;
          if ( v137 == v128 )
            goto LABEL_125;
          v135 = a7 + 32LL * (a10 - v137);
        }
      }
      v141 = 0;
LABEL_125:
      v78 = v128 + 1;
      if ( v128 < v125 )
      {
        do
        {
          sub_1E46110((__int64)a1, a2, a6, a8, a10, v78, v146, v143, v141, 0);
          v79 = v78++;
        }
        while ( v125 != v79 );
      }
      if ( v113 )
      {
        if ( (unsigned __int8)sub_1932870(v112, (int *)&v144, &v148) )
        {
          v99 = a1[266];
          v100 = a1[5];
          v101 = a1[115];
          v102 = sub_1E49390(v112, (int *)&v144);
          sub_1E42770(v143, v102[1], v101, v100, v99, v103);
        }
      }
      sub_1E47360(&v146);
      result = v146;
    }
    while ( v146 != v110 );
  }
  return result;
}
