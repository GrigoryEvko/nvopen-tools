// Function: sub_17DC8F0
// Address: 0x17dc8f0
//
__int64 __fastcall sub_17DC8F0(__int128 a1, double a2, double a3, double a4)
{
  __int64 v4; // rax
  __int64 v5; // r13
  unsigned int v6; // eax
  __int64 result; // rax
  __int64 v8; // rax
  __int64 v9; // rdx
  __int64 v10; // r8
  __int64 v11; // r14
  __int64 *v12; // rbx
  __int64 v13; // rax
  _BYTE *v14; // rdx
  __int64 *v15; // r9
  const char *v16; // r14
  __int64 v17; // rax
  __int64 v18; // r12
  __int64 v19; // rax
  __int64 v20; // rdx
  __int64 v21; // rcx
  __int64 v22; // rax
  int v23; // r12d
  __int64 v24; // rax
  __int64 v25; // rdx
  __int64 v26; // rbx
  int v27; // ebx
  __int64 v28; // rax
  __int64 v29; // rdx
  int v30; // edx
  int v31; // eax
  _QWORD *v32; // r12
  __int64 v33; // rax
  char v34; // dl
  __int64 v35; // rax
  __int64 *v36; // r14
  _QWORD *v37; // rax
  __int64 *v38; // rax
  __int64 v39; // rax
  __int64 v40; // rax
  __int64 v41; // rax
  __int64 *v42; // r14
  __int64 v43; // rsi
  __int64 *v44; // rax
  __int64 v45; // rdx
  __int64 v46; // rcx
  _BYTE *v47; // rax
  __int64 v48; // rax
  int v49; // edx
  unsigned int v50; // edx
  int v51; // edx
  __int64 *v52; // rax
  __int64 *v53; // r14
  __int64 *v54; // rax
  __int64 v55; // rax
  _BYTE *v56; // r14
  __int64 v57; // rdx
  __int64 v58; // rax
  __int64 v59; // rax
  __int64 v60; // rax
  __int64 *v61; // r14
  __int64 v62; // rbx
  __int64 v63; // rax
  __int64 v64; // rbx
  __int64 v65; // rax
  __int64 v66; // r14
  __int64 *v67; // rax
  __int64 v68; // rax
  __int64 v69; // rdx
  __int64 v70; // rbx
  _QWORD *v71; // rbx
  __int64 v72; // rcx
  __int64 v73; // rax
  __int64 v74; // rdx
  int v75; // r8d
  int v76; // r9d
  __int64 v77; // rax
  __int64 *v78; // rax
  unsigned __int64 v79; // rcx
  __int64 v80; // rax
  __int64 v81; // rbx
  char v82; // al
  char v83; // dl
  int v84; // r12d
  __int64 v85; // rax
  __int64 v86; // rdx
  __int64 v87; // r14
  int v88; // r14d
  __int64 v89; // rax
  __int64 v90; // rdx
  int v91; // eax
  int v92; // r14d
  _QWORD **v93; // rax
  __int64 v94; // r15
  __int64 v95; // rdx
  __int128 v96; // rdi
  __int64 *v97; // rax
  __int64 v98; // rax
  __int64 v99; // rbx
  __int64 v100; // rdx
  __int64 v101; // rcx
  __int64 v102; // rax
  unsigned int v103; // r9d
  __int64 v104; // rax
  __int64 v105; // rbx
  __int64 *v106; // r14
  __int64 v107; // rax
  __int64 v108; // rdx
  _QWORD *v109; // rax
  __int64 v110; // rdx
  __int64 v111; // rcx
  __int64 v112; // r10
  __int64 v113; // rax
  __int64 v114; // r14
  __int64 *v115; // rcx
  __int64 v116; // rax
  __int64 v117; // rdx
  _QWORD *v118; // rbx
  __int64 v119; // rdx
  __int64 v120; // rcx
  _QWORD *v121; // rax
  __int64 v122; // rsi
  __int64 *v123; // rax
  __int64 v124; // rcx
  __int64 v125; // rdx
  __int64 v126; // rax
  __int64 *v127; // [rsp-100h] [rbp-100h]
  unsigned int v128; // [rsp-F8h] [rbp-F8h]
  _BYTE *v129; // [rsp-F8h] [rbp-F8h]
  unsigned int v130; // [rsp-F0h] [rbp-F0h]
  __int64 v131; // [rsp-F0h] [rbp-F0h]
  __int64 v132; // [rsp-E8h] [rbp-E8h]
  __int64 v133; // [rsp-E8h] [rbp-E8h]
  unsigned __int64 *v134; // [rsp-E8h] [rbp-E8h]
  __int64 v135; // [rsp-E0h] [rbp-E0h]
  const char *v136; // [rsp-E0h] [rbp-E0h]
  __int64 *v137; // [rsp-E0h] [rbp-E0h]
  __int64 **v138; // [rsp-E0h] [rbp-E0h]
  __int64 v139; // [rsp-E0h] [rbp-E0h]
  __int64 v140; // [rsp-E0h] [rbp-E0h]
  __int64 v141; // [rsp-E0h] [rbp-E0h]
  __int64 v142; // [rsp-D0h] [rbp-D0h] BYREF
  __int64 v143[2]; // [rsp-C8h] [rbp-C8h] BYREF
  __int16 v144; // [rsp-B8h] [rbp-B8h]
  __m128i v145; // [rsp-A8h] [rbp-A8h] BYREF
  __int64 *v146; // [rsp-98h] [rbp-98h]
  _QWORD *v147; // [rsp-90h] [rbp-90h]
  __int64 v148; // [rsp-88h] [rbp-88h] BYREF
  __int64 v149; // [rsp-80h] [rbp-80h]
  unsigned __int64 *v150; // [rsp-78h] [rbp-78h]
  _QWORD *v151; // [rsp-70h] [rbp-70h]

  v4 = *(_QWORD *)(*((_QWORD *)&a1 + 1) - 24LL);
  if ( *(_BYTE *)(v4 + 16) )
    BUG();
  v5 = *((_QWORD *)&a1 + 1);
  v6 = *(_DWORD *)(v4 + 36);
  if ( v6 <= 0x1D1B )
  {
    if ( v6 > 0x1C09 )
    {
      result = v6 - 7178;
      switch ( (int)result )
      {
        case 0:
          v50 = 32;
          return sub_17DB790(a1, v50);
        case 1:
        case 2:
          v50 = 16;
          return sub_17DB790(a1, v50);
        case 24:
          v51 = 16;
          return sub_17DA370(a1, v51);
        case 35:
        case 183:
          return sub_17DB110(a1, a2, a3, a4);
        case 36:
        case 37:
        case 38:
        case 39:
        case 40:
        case 41:
        case 42:
        case 43:
        case 44:
        case 45:
        case 46:
        case 47:
        case 48:
        case 49:
        case 50:
        case 51:
        case 184:
        case 185:
        case 186:
        case 187:
        case 188:
        case 189:
        case 190:
        case 191:
        case 192:
        case 193:
        case 194:
        case 195:
        case 196:
        case 197:
        case 198:
        case 199:
          goto LABEL_37;
        case 107:
        case 145:
          sub_17CE510((__int64)&v148, *((__int64 *)&a1 + 1), 0, 0, 0);
          v52 = sub_17CD8D0((_QWORD *)a1, **((_QWORD **)&a1 + 1));
          LOWORD(v146) = 257;
          v138 = (__int64 **)v52;
          v53 = sub_17D57D0(a1, 1);
          v54 = sub_17D57D0(a1, 0);
          v55 = sub_156D390(&v148, (__int64)v54, (__int64)v53, (__int64)&v145);
          LOWORD(v146) = 257;
          v56 = (_BYTE *)v55;
          v144 = 257;
          v58 = sub_15A06D0(v138, 257, v57, 257);
          v59 = sub_12AA0C0(&v148, 0x21u, v56, v58, (__int64)v143);
          v60 = sub_12AA3B0(&v148, 0x26u, v59, (__int64)v138, (__int64)&v145);
          sub_17D4920(a1, *((__int64 **)&a1 + 1), v60);
          if ( *(_DWORD *)(*(_QWORD *)(a1 + 8) + 156LL) )
            sub_17D9C10((_QWORD *)a1, *((__int64 *)&a1 + 1));
          return sub_17CD270(&v148);
        case 108:
        case 109:
        case 110:
        case 111:
        case 112:
        case 113:
        case 114:
        case 138:
        case 139:
        case 140:
        case 141:
        case 142:
        case 143:
        case 146:
        case 147:
        case 148:
        case 149:
        case 150:
        case 151:
        case 152:
        case 204:
        case 205:
        case 206:
        case 207:
        case 208:
        case 209:
          sub_17CE510((__int64)&v148, *((__int64 *)&a1 + 1), 0, 0, 0);
          LOWORD(v146) = 257;
          if ( (*(_BYTE *)(*((_QWORD *)&a1 + 1) + 23LL) & 0x40) != 0 )
            v35 = *(_QWORD *)(*((_QWORD *)&a1 + 1) - 8LL);
          else
            v35 = *((_QWORD *)&a1 + 1) - 24LL * (*(_DWORD *)(*((_QWORD *)&a1 + 1) + 20LL) & 0xFFFFFFF);
          *((_QWORD *)&a1 + 1) = *(_QWORD *)(v35 + 24);
          v36 = sub_17D4DA0(a1);
          if ( (*(_BYTE *)(v5 + 23) & 0x40) != 0 )
            v37 = *(_QWORD **)(v5 - 8);
          else
            v37 = (_QWORD *)(v5 - 24LL * (*(_DWORD *)(v5 + 20) & 0xFFFFFFF));
          *((_QWORD *)&a1 + 1) = *v37;
          v38 = sub_17D4DA0(a1);
          v39 = sub_156D390(&v148, (__int64)v38, (__int64)v36, (__int64)&v145);
          *((_QWORD *)&a1 + 1) = *(_QWORD *)v5;
          v133 = v39;
          v144 = 257;
          v137 = sub_17CD8D0((_QWORD *)a1, *((__int64 *)&a1 + 1));
          v40 = sub_1643360(v151);
          v41 = sub_159C470(v40, 0, 0);
          if ( *(_BYTE *)(v133 + 16) > 0x10u || *(_BYTE *)(v41 + 16) > 0x10u )
          {
            v131 = v41;
            LOWORD(v146) = 257;
            v78 = sub_1648A60(56, 2u);
            v42 = v78;
            if ( v78 )
              sub_15FA320((__int64)v78, (_QWORD *)v133, v131, (__int64)&v145, 0);
            if ( v149 )
            {
              v134 = v150;
              sub_157E9D0(v149 + 40, (__int64)v42);
              v79 = *v134;
              v80 = v42[3] & 7;
              v42[4] = (__int64)v134;
              v79 &= 0xFFFFFFFFFFFFFFF8LL;
              v42[3] = v79 | v80;
              *(_QWORD *)(v79 + 8) = v42 + 3;
              *v134 = *v134 & 7 | (unsigned __int64)(v42 + 3);
            }
            sub_164B780((__int64)v42, v143);
            sub_12A86E0(&v148, (__int64)v42);
          }
          else
          {
            v42 = (__int64 *)sub_15A37D0((_BYTE *)v133, v41, 0);
          }
          LOWORD(v146) = 257;
          v43 = *v42;
          v44 = sub_17CD8D0((_QWORD *)a1, *v42);
          v46 = (__int64)v44;
          if ( v44 )
            v46 = sub_15A06D0((__int64 **)v44, v43, v45, (__int64)v44);
          v47 = (_BYTE *)sub_12AA0C0(&v148, 0x21u, v42, v46, (__int64)&v145);
          v48 = sub_17CF940((_QWORD *)a1, &v148, v47, (__int64)v137, 1);
          sub_17D4920(a1, (__int64 *)v5, v48);
          result = *(_QWORD *)(a1 + 8);
          if ( *(_DWORD *)(result + 156) )
            result = sub_17D9C10((_QWORD *)a1, v5);
          goto LABEL_14;
        case 118:
        case 122:
          v49 = 2;
          return sub_17D8620(a1, v49);
        case 119:
        case 120:
        case 123:
        case 124:
        case 156:
        case 157:
        case 158:
        case 161:
        case 162:
          goto LABEL_51;
        case 125:
          if ( !*(_BYTE *)(a1 + 488) )
            return result;
          sub_17CE510((__int64)&v148, *((__int64 *)&a1 + 1), 0, 0, 0);
          v66 = *(_QWORD *)(*((_QWORD *)&a1 + 1) - 24LL * (*(_DWORD *)(*((_QWORD *)&a1 + 1) + 20LL) & 0xFFFFFFF));
          v67 = (__int64 *)sub_1643350(v151);
          v68 = sub_17CFB40(a1, v66, &v148, v67, 1u);
          v140 = v69;
          v70 = v68;
          if ( byte_4FA4600 )
          {
            *((_QWORD *)&a1 + 1) = v66;
            sub_17D5820(a1, v5);
          }
          v71 = sub_17D3810(&v148, v70, "_ldmxcsr");
          sub_15F8F50((__int64)v71, 1u);
          v73 = *(_QWORD *)(a1 + 8);
          v74 = *(unsigned int *)(v73 + 156);
          if ( (_DWORD)v74 )
          {
            LOWORD(v146) = 257;
            result = (__int64)sub_156E5B0(&v148, v140, (__int64)&v145);
          }
          else
          {
            result = sub_15A06D0(*(__int64 ***)(v73 + 184), 1, v74, v72);
          }
          if ( *(_BYTE *)(a1 + 488) )
          {
            v145.m128i_i64[1] = result;
            v77 = *(unsigned int *)(a1 + 504);
            v145.m128i_i64[0] = (__int64)v71;
            v146 = (__int64 *)v5;
            if ( (unsigned int)v77 >= *(_DWORD *)(a1 + 508) )
            {
              sub_16CD150(a1 + 496, (const void *)(a1 + 512), 0, 24, v75, v76);
              v77 = *(unsigned int *)(a1 + 504);
            }
            result = *(_QWORD *)(a1 + 496) + 24 * v77;
            *(__m128i *)result = _mm_loadu_si128(&v145);
            *(_QWORD *)(result + 16) = v146;
            ++*(_DWORD *)(a1 + 504);
          }
          break;
        case 137:
          sub_17CE510((__int64)&v148, *((__int64 *)&a1 + 1), 0, 0, 0);
          v139 = *(_QWORD *)(*((_QWORD *)&a1 + 1) - 24LL * (*(_DWORD *)(*((_QWORD *)&a1 + 1) + 20LL) & 0xFFFFFFF));
          v61 = (__int64 *)sub_1643350(v151);
          v62 = sub_17CFB40(a1, v139, &v148, v61, 1u);
          LOWORD(v146) = 257;
          v63 = sub_1647190(v61, 0);
          v64 = sub_12A95D0(&v148, v62, v63, (__int64)&v145);
          v65 = sub_17CDAE0((_QWORD *)a1, (__int64)v61);
          sub_12A8F50(&v148, v65, v64, 0);
          if ( byte_4FA4600 )
          {
            *((_QWORD *)&a1 + 1) = v139;
            sub_17D5820(a1, v5);
          }
          return sub_17CD270(&v148);
        case 171:
        case 172:
        case 173:
        case 225:
          goto LABEL_53;
        case 179:
        case 273:
          goto LABEL_55;
        case 272:
          v51 = 8;
          return sub_17DA370(a1, v51);
        default:
          goto LABEL_21;
      }
      goto LABEL_14;
    }
    if ( v6 > 0x196F )
    {
      switch ( v6 )
      {
        case 0x1B39u:
        case 0x1B3Au:
        case 0x1B3Bu:
        case 0x1B3Cu:
        case 0x1B3Du:
        case 0x1B3Eu:
        case 0x1B44u:
        case 0x1B45u:
        case 0x1B46u:
        case 0x1B47u:
        case 0x1B48u:
        case 0x1B49u:
        case 0x1B4Au:
        case 0x1B4Bu:
        case 0x1B4Cu:
        case 0x1B4Du:
        case 0x1B55u:
        case 0x1B56u:
        case 0x1B57u:
        case 0x1B58u:
        case 0x1B59u:
        case 0x1B5Au:
LABEL_37:
          v34 = 0;
          return sub_17DB440(a1, v34);
        case 0x1B3Fu:
        case 0x1B40u:
        case 0x1B4Eu:
        case 0x1B4Fu:
        case 0x1B50u:
        case 0x1B51u:
        case 0x1B5Bu:
        case 0x1B5Cu:
LABEL_39:
          v34 = 1;
          return sub_17DB440(a1, v34);
        case 0x1BA2u:
        case 0x1BA3u:
        case 0x1BA6u:
        case 0x1BA7u:
LABEL_51:
          v49 = 1;
          return sub_17D8620(a1, v49);
        default:
          goto LABEL_21;
      }
    }
    if ( v6 > 0x1919 )
    {
      switch ( v6 )
      {
        case 0x191Au:
        case 0x191Bu:
        case 0x191Cu:
        case 0x191Du:
LABEL_53:
          v50 = 0;
          return sub_17DB790(a1, v50);
        case 0x192Bu:
        case 0x192Cu:
LABEL_55:
          v51 = 0;
          return sub_17DA370(a1, v51);
        case 0x1931u:
          return sub_17DB110(a1, a2, a3, a4);
        case 0x1936u:
        case 0x1937u:
        case 0x1938u:
        case 0x1939u:
        case 0x193Au:
        case 0x193Bu:
        case 0x1940u:
        case 0x1941u:
        case 0x1942u:
        case 0x1943u:
        case 0x1946u:
        case 0x1947u:
        case 0x1948u:
        case 0x1949u:
        case 0x194Au:
        case 0x194Bu:
          goto LABEL_37;
        case 0x193Cu:
        case 0x193Du:
        case 0x193Eu:
        case 0x193Fu:
        case 0x1944u:
        case 0x1945u:
        case 0x194Cu:
        case 0x194Du:
        case 0x194Eu:
        case 0x194Fu:
          goto LABEL_39;
        case 0x1967u:
        case 0x1968u:
        case 0x196Bu:
        case 0x196Cu:
        case 0x196Du:
        case 0x196Eu:
        case 0x196Fu:
          goto LABEL_51;
        default:
          goto LABEL_21;
      }
    }
    switch ( v6 )
    {
      case 0x81u:
        return sub_17DC390(a1, a2, a3, a4);
      case 0x83u:
        sub_17CE510((__int64)&v148, *((__int64 *)&a1 + 1), 0, 0, 0);
        v8 = *(_DWORD *)(*((_QWORD *)&a1 + 1) + 20LL) & 0xFFFFFFF;
        v136 = *(const char **)(*((_QWORD *)&a1 + 1) - 24 * v8);
        v132 = *(_QWORD *)(*((_QWORD *)&a1 + 1) + 24 * (1 - v8));
        v9 = *(_QWORD *)(*((_QWORD *)&a1 + 1) + 24 * (2 - v8));
        if ( *(_DWORD *)(v9 + 32) <= 0x40u )
          v10 = *(_QWORD *)(v9 + 24);
        else
          v10 = **(_QWORD **)(v9 + 24);
        *((_QWORD *)&a1 + 1) = *(_QWORD *)(*((_QWORD *)&a1 + 1)
                                         - 24LL * (*(_DWORD *)(*((_QWORD *)&a1 + 1) + 20LL) & 0xFFFFFFF));
        v130 = v10;
        v128 = v10;
        v11 = *(_QWORD *)(v5 + 24 * (3 - v8));
        v12 = sub_17D4DA0(a1);
        v13 = sub_17CFB40(a1, v132, &v148, (__int64 *)*v12, v128);
        v129 = v14;
        v15 = (__int64 *)v13;
        if ( byte_4FA4600 )
        {
          *((_QWORD *)&a1 + 1) = v132;
          v127 = (__int64 *)v13;
          sub_17D5820(a1, v5);
          *((_QWORD *)&a1 + 1) = v11;
          sub_17D5820(a1, v5);
          v15 = v127;
        }
        sub_15E80D0(&v148, (__int64)v12, v15, v130, v11);
        result = *(_QWORD *)(a1 + 8);
        if ( *(_DWORD *)(result + 156) )
        {
          v98 = sub_1632FA0(*(_QWORD *)(*(_QWORD *)a1 + 40LL));
          v99 = sub_127FA20(v98, *v12);
          v102 = sub_17D4880(a1, v136, v100, v101);
          v103 = 4;
          if ( v130 >= 4 )
            v103 = v130;
          result = sub_17D3020((_QWORD *)a1, &v148, v102, v129, (unsigned __int64)(v99 + 7) >> 3, v103, a2, a3, a4);
        }
        goto LABEL_14;
      case 6u:
        sub_17CE510((__int64)&v148, *((__int64 *)&a1 + 1), 0, 0, 0);
        v16 = *(const char **)(*((_QWORD *)&a1 + 1) - 24LL * (*(_DWORD *)(*((_QWORD *)&a1 + 1) + 20LL) & 0xFFFFFFF));
        v142 = *(_QWORD *)v16;
        v17 = sub_15E26F0(*(__int64 **)(*(_QWORD *)a1 + 40LL), 6, &v142, 1);
        *((_QWORD *)&a1 + 1) = v16;
        LOWORD(v146) = 257;
        v18 = v17;
        v143[0] = (__int64)sub_17D4DA0(a1);
        v19 = sub_1285290(&v148, *(_QWORD *)(v18 + 24), v18, (int)v143, 1, (__int64)&v145, 0);
        sub_17D4920(a1, (__int64 *)v5, v19);
        v22 = sub_17D4880(a1, v16, v20, v21);
        sub_17D4B80(a1, v5, v22);
        return sub_17CD270(&v148);
    }
  }
LABEL_21:
  v23 = *(_DWORD *)(*((_QWORD *)&a1 + 1) + 20LL) & 0xFFFFFFF;
  if ( *(char *)(*((_QWORD *)&a1 + 1) + 23LL) >= 0 )
    goto LABEL_79;
  v24 = sub_1648A40(*((__int64 *)&a1 + 1));
  v26 = v24 + v25;
  if ( *(char *)(*((_QWORD *)&a1 + 1) + 23LL) >= 0 )
  {
    if ( (unsigned int)(v26 >> 4) )
LABEL_148:
      BUG();
LABEL_79:
    v30 = 0;
    goto LABEL_27;
  }
  if ( !(unsigned int)((v26 - sub_1648A40(*((__int64 *)&a1 + 1))) >> 4) )
    goto LABEL_79;
  if ( *(char *)(*((_QWORD *)&a1 + 1) + 23LL) >= 0 )
    goto LABEL_148;
  v27 = *(_DWORD *)(sub_1648A40(*((__int64 *)&a1 + 1)) + 8);
  if ( *(char *)(*((_QWORD *)&a1 + 1) + 23LL) >= 0 )
    BUG();
  v28 = sub_1648A40(*((__int64 *)&a1 + 1));
  v30 = *(_DWORD *)(v28 + v29 - 4) - v27;
LABEL_27:
  v31 = v23 - 1 - v30;
  if ( v23 - 1 == v30 )
    return sub_17D7760((_QWORD *)a1, *((__int64 *)&a1 + 1));
  if ( v31 == 2 )
  {
    v32 = (_QWORD *)(*((_QWORD *)&a1 + 1) + 56LL);
    if ( *(_BYTE *)(**(_QWORD **)(*((_QWORD *)&a1 + 1) - 24LL * (*(_DWORD *)(*((_QWORD *)&a1 + 1) + 20LL) & 0xFFFFFFF))
                  + 8LL) == 15
      && *(_BYTE *)(**(_QWORD **)(*((_QWORD *)&a1 + 1)
                                + 24 * (1LL - (*(_DWORD *)(*((_QWORD *)&a1 + 1) + 20LL) & 0xFFFFFFF)))
                  + 8LL) == 16
      && !*(_BYTE *)(**((_QWORD **)&a1 + 1) + 8LL)
      && !(unsigned __int8)sub_1560260((_QWORD *)(*((_QWORD *)&a1 + 1) + 56LL), -1, 36) )
    {
      if ( (unsigned int)sub_165C280(*((__int64 *)&a1 + 1))
        || (v104 = *(_QWORD *)(*((_QWORD *)&a1 + 1) - 24LL), *(_BYTE *)(v104 + 16))
        || (v148 = *(_QWORD *)(v104 + 112), !(unsigned __int8)sub_1560260(&v148, -1, 36)) )
      {
        if ( !(unsigned __int8)sub_17CEE20(*((__int64 *)&a1 + 1), 37) )
        {
          sub_17CE510((__int64)&v148, *((__int64 *)&a1 + 1), 0, 0, 0);
          v105 = *(_QWORD *)(*((_QWORD *)&a1 + 1) - 24LL * (*(_DWORD *)(*((_QWORD *)&a1 + 1) + 20LL) & 0xFFFFFFF));
          v106 = sub_17D57D0(a1, 1);
          v107 = sub_17CFB40(a1, v105, &v148, (__int64 *)*v106, 1u);
          v141 = v108;
          v109 = sub_12A8F50(&v148, (__int64)v106, v107, 0);
          sub_15F9450((__int64)v109, 1u);
          if ( byte_4FA4600 )
          {
            *((_QWORD *)&a1 + 1) = v105;
            sub_17D5820(a1, v5);
          }
          if ( *(_DWORD *)(*(_QWORD *)(a1 + 8) + 156LL) )
          {
            if ( (*(_BYTE *)(v5 + 23) & 0x40) != 0 )
              v112 = *(_QWORD *)(v5 - 8);
            else
              v112 = v5 - 24LL * (*(_DWORD *)(v5 + 20) & 0xFFFFFFF);
            v113 = sub_17D4880(a1, *(const char **)(v112 + 24), v110, v111);
            sub_12A8F50(&v148, v113, v141, 0);
          }
          return sub_17CD270(&v148);
        }
      }
    }
    goto LABEL_31;
  }
  v32 = (_QWORD *)(*((_QWORD *)&a1 + 1) + 56LL);
  if ( v31 != 1
    || (v32 = (_QWORD *)(*((_QWORD *)&a1 + 1) + 56LL),
        *(_BYTE *)(**(_QWORD **)(*((_QWORD *)&a1 + 1) - 24LL * (*(_DWORD *)(*((_QWORD *)&a1 + 1) + 20LL) & 0xFFFFFFF))
                 + 8LL) != 15)
    || *(_BYTE *)(**((_QWORD **)&a1 + 1) + 8LL) != 16
    || !(unsigned __int8)sub_17CEE20(*((__int64 *)&a1 + 1), 36)
    && !(unsigned __int8)sub_17CEE20(*((__int64 *)&a1 + 1), 37) )
  {
LABEL_31:
    if ( !(unsigned __int8)sub_1560260(v32, -1, 36) )
    {
      if ( (unsigned int)sub_165C280(*((__int64 *)&a1 + 1)) )
        return sub_17D7760((_QWORD *)a1, *((__int64 *)&a1 + 1));
      v33 = *(_QWORD *)(*((_QWORD *)&a1 + 1) - 24LL);
      if ( *(_BYTE *)(v33 + 16) )
        return sub_17D7760((_QWORD *)a1, *((__int64 *)&a1 + 1));
      v148 = *(_QWORD *)(v33 + 112);
      if ( !(unsigned __int8)sub_1560260(&v148, -1, 36) )
        return sub_17D7760((_QWORD *)a1, *((__int64 *)&a1 + 1));
    }
    v81 = **((_QWORD **)&a1 + 1);
    v82 = *(_BYTE *)(**((_QWORD **)&a1 + 1) + 8LL);
    if ( v82 == 16 )
    {
      v83 = *(_BYTE *)(**(_QWORD **)(v81 + 16) + 8LL);
      if ( v83 == 11 )
        goto LABEL_91;
    }
    else
    {
      v83 = *(_BYTE *)(**((_QWORD **)&a1 + 1) + 8LL);
      if ( v82 == 11 )
        goto LABEL_91;
    }
    if ( (unsigned __int8)(v83 - 1) <= 5u || v82 == 9 )
    {
LABEL_91:
      v84 = *(_DWORD *)(*((_QWORD *)&a1 + 1) + 20LL) & 0xFFFFFFF;
      if ( *(char *)(*((_QWORD *)&a1 + 1) + 23LL) < 0 )
      {
        v85 = sub_1648A40(*((__int64 *)&a1 + 1));
        v87 = v85 + v86;
        if ( *(char *)(*((_QWORD *)&a1 + 1) + 23LL) >= 0 )
        {
          if ( (unsigned int)(v87 >> 4) )
            goto LABEL_150;
        }
        else if ( (unsigned int)((v87 - sub_1648A40(*((__int64 *)&a1 + 1))) >> 4) )
        {
          if ( *(char *)(*((_QWORD *)&a1 + 1) + 23LL) < 0 )
          {
            v88 = *(_DWORD *)(sub_1648A40(*((__int64 *)&a1 + 1)) + 8);
            if ( *(char *)(*((_QWORD *)&a1 + 1) + 23LL) >= 0 )
              BUG();
            v89 = sub_1648A40(*((__int64 *)&a1 + 1));
            v91 = *(_DWORD *)(v89 + v90 - 4) - v88;
            goto LABEL_97;
          }
LABEL_150:
          BUG();
        }
      }
      v91 = 0;
LABEL_97:
      v92 = v84 - 1 - v91;
      if ( v84 - 1 != v91 )
      {
        v93 = (_QWORD **)(*((_QWORD *)&a1 + 1) - 24LL * (*(_DWORD *)(*((_QWORD *)&a1 + 1) + 20LL) & 0xFFFFFFF));
        while ( v81 == **v93 )
        {
          v93 += 3;
          if ( (_QWORD **)(*((_QWORD *)&a1 + 1)
                         + 24
                         * ((unsigned int)(v92 - 1)
                          - (unsigned __int64)(*(_DWORD *)(*((_QWORD *)&a1 + 1) + 20LL) & 0xFFFFFFF))
                         + 24) == v93 )
          {
            sub_17CE510((__int64)&v148, *((__int64 *)&a1 + 1), 0, 0, 0);
            v146 = &v148;
            v145 = 0u;
            v147 = (_QWORD *)a1;
            v94 = 0;
            do
            {
              v95 = v94;
              *(_QWORD *)&v96 = &v145;
              ++v94;
              *((_QWORD *)&v96 + 1) = *(_QWORD *)(v5 + 24 * (v95 - (*(_DWORD *)(v5 + 20) & 0xFFFFFFF)));
              sub_17D7560(v96);
            }
            while ( v94 != v92 );
            goto LABEL_103;
          }
        }
        return sub_17D7760((_QWORD *)a1, *((__int64 *)&a1 + 1));
      }
      sub_17CE510((__int64)&v148, *((__int64 *)&a1 + 1), 0, 0, 0);
      v146 = &v148;
      v145 = 0u;
      v147 = (_QWORD *)a1;
LABEL_103:
      v97 = sub_17CD8D0(v147, *(_QWORD *)v5);
      v145.m128i_i64[0] = sub_17CF940(v147, v146, v145.m128i_i64[0], (__int64)v97, 0);
      sub_17D4920((__int64)v147, (__int64 *)v5, v145.m128i_i64[0]);
      if ( *(_DWORD *)(v147[1] + 156LL) )
        sub_17D4B80((__int64)v147, v5, v145.m128i_i64[1]);
      return sub_17CD270(&v148);
    }
    return sub_17D7760((_QWORD *)a1, *((__int64 *)&a1 + 1));
  }
  sub_17CE510((__int64)&v148, *((__int64 *)&a1 + 1), 0, 0, 0);
  v114 = *(_QWORD *)(*((_QWORD *)&a1 + 1) - 24LL * (*(_DWORD *)(*((_QWORD *)&a1 + 1) + 20LL) & 0xFFFFFFF));
  v115 = sub_17CD8D0((_QWORD *)a1, **((_QWORD **)&a1 + 1));
  if ( *(_BYTE *)(a1 + 489) )
  {
    v116 = sub_17CFB40(a1, v114, &v148, v115, 1u);
    v135 = v117;
    v118 = sub_17D3810(&v148, v116, "_msld");
    sub_15F8F50((__int64)v118, 1u);
    sub_17D4920(a1, *((__int64 **)&a1 + 1), (__int64)v118);
  }
  else
  {
    v122 = **((_QWORD **)&a1 + 1);
    v123 = sub_17CD8D0((_QWORD *)a1, *(_QWORD *)v5);
    v125 = (__int64)v123;
    if ( v123 )
      v125 = sub_15A06D0((__int64 **)v123, v122, (__int64)v123, v124);
    *((_QWORD *)&a1 + 1) = v5;
    sub_17D4920(a1, (__int64 *)v5, v125);
  }
  if ( byte_4FA4600 )
  {
    *((_QWORD *)&a1 + 1) = v114;
    sub_17D5820(a1, v5);
  }
  result = *(_QWORD *)(a1 + 8);
  if ( *(_DWORD *)(result + 156) )
  {
    if ( *(_BYTE *)(a1 + 489) )
    {
      LOWORD(v146) = 257;
      v121 = sub_156E5B0(&v148, v135, (__int64)&v145);
      result = sub_17D4B80(a1, v5, (__int64)v121);
    }
    else
    {
      v126 = sub_15A06D0(*(__int64 ***)(result + 184), *((__int64 *)&a1 + 1), v119, v120);
      result = sub_17D4B80(a1, v5, v126);
    }
  }
LABEL_14:
  if ( v148 )
    return sub_161E7C0((__int64)&v148, v148);
  return result;
}
