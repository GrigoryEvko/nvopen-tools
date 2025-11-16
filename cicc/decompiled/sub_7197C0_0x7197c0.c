// Function: sub_7197C0
// Address: 0x7197c0
//
__int64 __fastcall sub_7197C0(__int64 a1, __m128i *a2, int a3, _DWORD *a4, _DWORD *a5)
{
  __int64 v6; // r12
  __int64 v8; // r15
  int v9; // eax
  __int64 v10; // rcx
  __int64 v11; // r15
  char v12; // cl
  __int64 result; // rax
  unsigned int v14; // r8d
  __int64 v15; // rbx
  __int64 v16; // r15
  __int64 v17; // rbx
  __int64 v18; // r15
  int v19; // eax
  int v20; // r15d
  __int64 v21; // rax
  __int64 v22; // rbx
  __int64 v23; // r15
  int v24; // eax
  __int64 v25; // rax
  _BYTE *v26; // r15
  __int64 v27; // rbx
  __int64 v28; // rbx
  __int64 v29; // r15
  char v30; // di
  unsigned __int64 v31; // r15
  __int64 v32; // rax
  __int64 v33; // rax
  __int64 v34; // r15
  __int64 v35; // rbx
  __int64 v36; // rcx
  _BOOL8 v37; // r8
  __int64 v38; // rbx
  __int64 v39; // r15
  char v40; // cl
  int v41; // eax
  __int64 mm; // rax
  __int64 v43; // rbx
  __int64 jj; // r15
  int v45; // eax
  char v46; // cl
  unsigned __int64 kk; // rbx
  __int64 v48; // r15
  __int64 v49; // rax
  char v50; // cl
  __int64 v51; // rax
  __int64 v52; // r9
  char m; // dl
  __int64 v54; // r8
  char n; // al
  int v56; // eax
  __int64 v57; // r8
  __int64 v58; // rdi
  __int64 ii; // r9
  __int64 *v60; // r15
  __int64 v61; // rcx
  char v62; // dl
  __int64 j; // rax
  __int64 v64; // r8
  int v65; // eax
  __int64 v66; // rcx
  __int64 v67; // rdi
  __int64 v68; // r8
  __int64 k; // rcx
  __int64 v70; // rax
  _BYTE *v71; // r14
  __int64 v72; // r15
  int v73; // eax
  __int64 v74; // rbx
  __int64 v75; // r15
  char v76; // dl
  __int64 i; // r8
  char v78; // al
  _BOOL8 v79; // r9
  bool v80; // al
  __int64 v81; // rax
  __int64 v82; // r15
  __int64 v83; // rbx
  int v84; // eax
  int v85; // ebx
  __int64 v86; // rax
  char v87; // dl
  __int64 v88; // rcx
  __int64 v89; // rax
  __int64 v90; // r15
  __int64 v91; // rbx
  __int64 v92; // rdx
  __int64 v93; // r8
  __int64 v94; // rax
  __int64 v95; // r15
  int v96; // ecx
  __int64 v97; // rax
  __int64 v98; // rdx
  char v99; // dl
  __int64 v100; // rdx
  __int64 v101; // rax
  __int64 v102; // r15
  __int64 v103; // rbx
  char v104; // dl
  char v105; // al
  __int64 v106; // rdx
  __int64 v107; // rbx
  __int64 v108; // r15
  __int64 v109; // rax
  __int64 v110; // rdx
  __int64 v111; // rcx
  char v112; // dl
  __int64 v113; // rax
  int v114; // eax
  __int64 v115; // rax
  __int64 v116; // rax
  __int64 v117; // rbx
  char v118; // al
  char v119; // al
  __int64 v120; // r15
  __int64 v121; // rbx
  __int64 v122; // rax
  __int64 v123; // rbx
  __int64 v124; // rdx
  __int64 v125; // rax
  __int64 v126; // rdx
  __int64 v127; // rax
  unsigned __int64 v128; // rbx
  __int64 v129; // [rsp+8h] [rbp-78h]
  __int64 v130; // [rsp+10h] [rbp-70h]
  __int64 v131; // [rsp+10h] [rbp-70h]
  __int64 v132; // [rsp+10h] [rbp-70h]
  __int64 v133; // [rsp+10h] [rbp-70h]
  __int64 v134; // [rsp+18h] [rbp-68h]
  __int64 v135; // [rsp+18h] [rbp-68h]
  __int64 v136; // [rsp+18h] [rbp-68h]
  int v137; // [rsp+18h] [rbp-68h]
  __int64 *v138; // [rsp+20h] [rbp-60h]
  __int64 v139; // [rsp+28h] [rbp-58h]
  __int64 *v140; // [rsp+28h] [rbp-58h]
  __int64 v141; // [rsp+28h] [rbp-58h]
  __int64 v142; // [rsp+28h] [rbp-58h]
  __int64 v143; // [rsp+28h] [rbp-58h]
  __int64 v144; // [rsp+30h] [rbp-50h]
  __int64 v145; // [rsp+30h] [rbp-50h]
  __int64 v146; // [rsp+30h] [rbp-50h]
  __int64 v147; // [rsp+30h] [rbp-50h]
  __int64 *v148; // [rsp+30h] [rbp-50h]
  __int64 v149; // [rsp+30h] [rbp-50h]
  __int64 v150; // [rsp+30h] [rbp-50h]
  FILE *v151; // [rsp+30h] [rbp-50h]
  __int64 v152; // [rsp+30h] [rbp-50h]
  __int64 v153; // [rsp+30h] [rbp-50h]
  char v155; // [rsp+38h] [rbp-48h]
  _BYTE *v156; // [rsp+38h] [rbp-48h]
  char v157; // [rsp+38h] [rbp-48h]
  unsigned __int8 v158; // [rsp+38h] [rbp-48h]
  __int64 v159; // [rsp+38h] [rbp-48h]
  char v160; // [rsp+38h] [rbp-48h]
  char v161; // [rsp+38h] [rbp-48h]
  char v162; // [rsp+38h] [rbp-48h]
  __int64 v163; // [rsp+38h] [rbp-48h]
  __int64 v164; // [rsp+38h] [rbp-48h]
  __int64 *v165; // [rsp+38h] [rbp-48h]
  __int64 v166; // [rsp+38h] [rbp-48h]
  __int64 v167; // [rsp+38h] [rbp-48h]
  int v168[13]; // [rsp+4Ch] [rbp-34h] BYREF

  v6 = a1;
  v8 = *(_QWORD *)(a1 + 64);
  *a5 = 0;
  if ( !v8 )
    goto LABEL_14;
  v9 = 0;
  do
  {
    while ( 1 )
    {
      if ( !*(_BYTE *)(v8 + 24) )
        return sub_724C70(a2, 0);
      if ( !v9 )
      {
        if ( dword_4F04C44 != -1
          || (v10 = qword_4F04C68[0] + 776LL * dword_4F04C64, (*(_BYTE *)(v10 + 6) & 6) != 0)
          || *(_BYTE *)(v10 + 4) == 12 )
        {
          a1 = v8;
          v9 = sub_731EE0(v8);
          if ( v9 )
            break;
        }
      }
      v8 = *(_QWORD *)(v8 + 16);
      if ( !v8 )
        goto LABEL_13;
    }
    v11 = *(_QWORD *)(v8 + 16);
    if ( !v11 )
      goto LABEL_15;
    if ( !*(_BYTE *)(v11 + 24) )
      return sub_724C70(a2, 0);
    v8 = *(_QWORD *)(v11 + 16);
    v9 = 1;
  }
  while ( v8 );
LABEL_13:
  if ( !v9 )
    goto LABEL_14;
LABEL_15:
  if ( (*(_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 12) & 0x10) == 0 )
    return sub_70FD90((__int64 *)v6, (__int64)a2);
LABEL_14:
  v12 = *(_BYTE *)(v6 + 56);
  switch ( v12 )
  {
    case 0:
      v93 = *(_QWORD *)(v6 + 64);
      v94 = *(_QWORD *)(v93 + 16);
      *a5 = 0;
      v153 = v93;
      v95 = v94;
      v96 = sub_8DBE70(*(_QWORD *)(v93 + 56));
      if ( v96 )
        goto LABEL_173;
      v97 = v95;
      if ( *(_BYTE *)(v95 + 24) != 2 )
      {
        do
        {
          v98 = v97;
          v97 = *(_QWORD *)(v97 + 72);
          v99 = *(_BYTE *)(v98 + 56);
          if ( v99 == 92 )
          {
            v100 = *(_QWORD *)(v97 + 16);
            if ( *(_BYTE *)(v100 + 24) == 2 )
            {
              if ( *(_BYTE *)(*(_QWORD *)(v100 + 56) + 173LL) == 12 )
                v96 = 1;
            }
            else
            {
              *a5 = 1;
            }
          }
          else if ( v99 == 100 )
          {
            v96 = 1;
          }
        }
        while ( *(_BYTE *)(v97 + 24) != 2 );
        if ( v96 )
        {
LABEL_173:
          sub_724C70(a2, 12);
          sub_7249B0(a2, 1);
          a2[11].m128i_i64[1] = v6;
          result = *(_QWORD *)v6;
          a2[8].m128i_i64[0] = *(_QWORD *)v6;
          return result;
        }
      }
      if ( !*a5 )
      {
        sub_72BBE0(a2, 0, byte_4F06A51[0]);
        if ( (unsigned int)sub_70D8E0(v95, a2, a4) )
          *(_BYTE *)(v153 + 26) |= 0x40u;
        else
          sub_724C70(a2, 0);
        goto LABEL_70;
      }
      result = (__int64)&dword_4F077BC;
      if ( dword_4F077BC )
      {
        result = (__int64)&qword_4F077A8;
        if ( qword_4F077A8 <= 0x9E97u )
        {
          result = (__int64)a4;
          if ( a4 )
            return sub_684AA0(7u, 0x6F6u, a4);
        }
      }
      return result;
    case 1:
    case 2:
    case 3:
    case 4:
    case 5:
    case 6:
    case 7:
    case 8:
    case 9:
    case 10:
    case 11:
    case 12:
    case 16:
    case 18:
    case 19:
    case 23:
    case 24:
    case 25:
    case 26:
    case 27:
    case 28:
    case 29:
    case 40:
    case 61:
    case 64:
    case 65:
    case 116:
      v14 = 1;
      return sub_70E4A0((__int64 *)v6, (__int64)a2, a3, a4, v14);
    case 13:
      v74 = *(_QWORD *)(v6 + 64);
      v75 = *(_QWORD *)(v74 + 56);
      v152 = *(_QWORD *)(v74 + 16);
      v166 = *(_QWORD *)(v152 + 56);
      if ( (unsigned int)sub_8DBE70(v75) || (unsigned int)sub_8DBE70(v166) )
        goto LABEL_37;
      v76 = *(_BYTE *)(v75 + 140);
      for ( i = v166; v76 == 12; v76 = *(_BYTE *)(v75 + 140) )
        v75 = *(_QWORD *)(v75 + 160);
      while ( 1 )
      {
        v78 = *(_BYTE *)(i + 140);
        if ( v78 != 12 )
          break;
        i = *(_QWORD *)(i + 160);
      }
      v79 = 0;
      if ( (unsigned __int8)(v76 - 9) <= 1u && (unsigned __int8)(v78 - 9) <= 2u )
      {
        v80 = 1;
        if ( i != v75
          && (!dword_4F07588 || !(v80 = *(_QWORD *)(i + 32) != 0 && *(_QWORD *)(v75 + 32) == *(_QWORD *)(i + 32))) )
        {
          v80 = sub_8D5CE0(i, v75) != 0;
        }
        v79 = v80;
      }
      v167 = v79;
      *(_BYTE *)(v74 + 26) |= 0x40u;
      *(_BYTE *)(v152 + 26) |= 0x40u;
      sub_724C70(a2, 1);
      sub_620D80((__m128i *)a2[11].m128i_i16, v167);
      if ( a3 )
        goto LABEL_32;
      goto LABEL_25;
    case 14:
    case 17:
    case 20:
    case 69:
    case 79:
    case 82:
    case 83:
    case 84:
    case 85:
    case 86:
    case 87:
    case 88:
    case 89:
    case 90:
    case 91:
    case 92:
    case 93:
    case 94:
    case 95:
    case 96:
    case 97:
    case 98:
    case 99:
    case 100:
    case 101:
    case 102:
    case 103:
    case 104:
    case 105:
    case 106:
    case 107:
    case 114:
    case 115:
      v14 = 0;
      return sub_70E4A0((__int64 *)v6, (__int64)a2, a3, a4, v14);
    case 15:
    case 108:
    case 111:
      v17 = *(_QWORD *)(v6 + 64);
      v18 = *(_QWORD *)(v17 + 56);
      v145 = *(_QWORD *)(v17 + 16);
      v156 = *(_BYTE **)(v145 + 56);
      if ( (unsigned int)sub_8DBE70(v18) || (unsigned int)sub_8DBE70(v156) )
        goto LABEL_24;
      v19 = sub_699F10(v18, v156, *(_BYTE *)(v6 + 56));
      *(_BYTE *)(v17 + 26) |= 0x40u;
      v20 = v19;
      v21 = v145;
      goto LABEL_30;
    case 21:
      v89 = *(_QWORD *)(v6 + 64);
      v90 = *(_QWORD *)(v89 + 56);
      v91 = *(_QWORD *)(*(_QWORD *)(v89 + 16) + 56LL);
      if ( (unsigned int)sub_8DBE70(v90) || (unsigned int)sub_8DBE70(v91) )
        goto LABEL_24;
      v92 = 3;
      if ( dword_4F077C0 )
        v92 = qword_4F077A8 < 0x9C40u ? 3 : 524291;
      v84 = sub_8DED30(v90, v91, v92);
      goto LABEL_130;
    case 30:
    case 31:
    case 41:
      v28 = *(_QWORD *)(v6 + 64);
      v158 = *(_BYTE *)(v6 + 56);
      v29 = *(_QWORD *)(v28 + 56);
      v146 = v28;
      if ( (unsigned int)sub_8DBE70(v29) )
        goto LABEL_24;
      while ( 1 )
      {
        v28 = *(_QWORD *)(v28 + 16);
        if ( !v28 )
          break;
        if ( (unsigned int)sub_8DBE70(*(_QWORD *)(v28 + 56)) )
          goto LABEL_24;
      }
      if ( (unsigned int)sub_6CC470(v158, v29, v6) )
      {
        v30 = 44;
        if ( v158 != 41 )
          v30 = (v158 == 31) + 42;
        v31 = (int)sub_69A120(v30, v29);
      }
      else
      {
        v31 = 0;
      }
      v32 = v146;
      do
      {
        *(_BYTE *)(v32 + 26) |= 0x40u;
        v32 = *(_QWORD *)(v32 + 16);
      }
      while ( v32 );
      goto LABEL_48;
    case 42:
    case 43:
    case 44:
      v22 = *(_QWORD *)(v6 + 64);
      v157 = *(_BYTE *)(v6 + 56);
      v23 = *(_QWORD *)(v22 + 56);
      if ( (unsigned int)sub_8DBE70(v23) )
        goto LABEL_24;
      v24 = sub_69A120(v157, v23);
      *(_BYTE *)(v22 + 26) |= 0x40u;
      v20 = v24;
      goto LABEL_31;
    case 45:
    case 46:
    case 60:
    case 62:
      v15 = *(_QWORD *)(v6 + 64);
      v155 = *(_BYTE *)(v6 + 56);
      v16 = *(_QWORD *)(v15 + 56);
      v139 = *(_QWORD *)(v15 + 16);
      v144 = *(_QWORD *)(v139 + 56);
      if ( (unsigned int)sub_8DBE70(v16) || (unsigned int)sub_8DBE70(v144) )
        goto LABEL_24;
      v114 = sub_69A3A0(v155, v16, v144);
      *(_BYTE *)(v15 + 26) |= 0x40u;
      v20 = v114;
      v21 = v139;
LABEL_30:
      *(_BYTE *)(v21 + 26) |= 0x40u;
      goto LABEL_31;
    case 63:
      result = (unsigned int)sub_719770(v6, (__int64)a2, 0, 0) == 0;
      *a5 = result;
      return result;
    case 66:
    case 109:
    case 110:
      v25 = *(_QWORD *)(v6 + 64);
      v26 = *(_BYTE **)(v25 + 56);
      v27 = *(_QWORD *)(*(_QWORD *)(v25 + 16) + 56LL);
      if ( (unsigned int)sub_8DBE70(v26) || (unsigned int)sub_8DBE70(v27) )
        goto LABEL_37;
      v84 = sub_69A560(v26, v27, *(_BYTE *)(v6 + 56));
      goto LABEL_130;
    case 67:
    case 68:
      v33 = *(_QWORD *)(v6 + 64);
      v34 = *(_QWORD *)(v33 + 56);
      v35 = *(_QWORD *)(*(_QWORD *)(v33 + 16) + 56LL);
      if ( (unsigned int)sub_8DBE70(v34) || (unsigned int)sub_8DBE70(v35) )
        goto LABEL_24;
      v37 = 1;
      if ( v34 != v35 )
        v37 = (unsigned int)sub_8D97D0(v34, v35, 0, v36, 1) != 0;
      goto LABEL_54;
    case 70:
      v86 = *(_QWORD *)(v6 + 64);
      v87 = *(_BYTE *)(v86 + 24);
      v88 = v86;
      if ( v87 != 1 )
        goto LABEL_133;
      v112 = *(_BYTE *)(v86 + 56);
      if ( (unsigned __int8)(v112 - 94) > 1u && (unsigned __int8)(v112 - 100) > 1u )
        goto LABEL_172;
      v88 = *(_QWORD *)(*(_QWORD *)(v86 + 72) + 16LL);
      v87 = *(_BYTE *)(v88 + 24);
LABEL_133:
      switch ( v87 )
      {
        case 1:
          goto LABEL_172;
        case 3:
        case 4:
        case 20:
        case 22:
          v106 = *(_QWORD *)(v88 + 56);
          if ( !v106 )
            goto LABEL_172;
          v107 = *(_QWORD *)(v106 + 104);
          if ( !v107 )
            goto LABEL_172;
          v108 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(v86 + 16) + 56LL) + 104LL);
          break;
        default:
          goto LABEL_21;
      }
      do
      {
        if ( *(_BYTE *)(v108 + 8) == *(_BYTE *)(v107 + 8) )
        {
          v109 = *(_QWORD *)(v108 + 32);
          if ( !v109
            || (v110 = *(_QWORD *)(v107 + 32)) != 0
            && *(_BYTE *)(v109 + 10) == *(_BYTE *)(v110 + 10)
            && ((unsigned int)sub_5D19D0(v108) || (unsigned int)sub_5CB890(v108, v107, 1, v111)) )
          {
            v31 = 1;
            goto LABEL_48;
          }
        }
        v107 = *(_QWORD *)v107;
      }
      while ( v107 );
LABEL_172:
      v31 = 0;
LABEL_48:
      sub_724C70(a2, 1);
      sub_620D80((__m128i *)a2[11].m128i_i16, v31);
      if ( a3 )
        goto LABEL_32;
      goto LABEL_25;
    case 72:
      v81 = *(_QWORD *)(v6 + 64);
      v82 = *(_QWORD *)(v81 + 56);
      v83 = *(_QWORD *)(*(_QWORD *)(v81 + 16) + 56LL);
      if ( (unsigned int)sub_8DBE70(v82) || (unsigned int)sub_8DBE70(v83) )
        goto LABEL_37;
      v84 = sub_8E4940(v82, v83);
LABEL_130:
      v85 = v84;
      sub_724C70(a2, 1);
      sub_620D80((__m128i *)a2[11].m128i_i16, v85);
      if ( a3 )
        goto LABEL_32;
      goto LABEL_25;
    case 73:
      v101 = *(_QWORD *)(v6 + 64);
      v102 = *(_QWORD *)(v101 + 56);
      v103 = *(_QWORD *)(*(_QWORD *)(v101 + 16) + 56LL);
      if ( (unsigned int)sub_8DBE70(v102) || (unsigned int)sub_8DBE70(v103) )
        goto LABEL_37;
      while ( 1 )
      {
        v104 = *(_BYTE *)(v102 + 140);
        if ( v104 != 12 )
          break;
        v102 = *(_QWORD *)(v102 + 160);
      }
      while ( 1 )
      {
        v105 = *(_BYTE *)(v103 + 140);
        if ( v105 != 12 )
          break;
        v103 = *(_QWORD *)(v103 + 160);
      }
      if ( (unsigned __int8)(v104 - 9) > 1u || (unsigned __int8)(v105 - 9) > 1u )
        goto LABEL_158;
      v37 = 1;
      if ( v102 != v103 )
      {
        if ( !dword_4F07588 || (v115 = *(_QWORD *)(v102 + 32), *(_QWORD *)(v103 + 32) != v115) || !v115 )
        {
          v116 = sub_8D5CE0(v103, v102);
          if ( !v116 || (*(_BYTE *)(v116 + 96) & 6) != 0 )
            goto LABEL_158;
          while ( *(_BYTE *)(v103 + 140) == 12 )
            v103 = *(_QWORD *)(v103 + 160);
          if ( *(char *)(*(_QWORD *)(*(_QWORD *)v103 + 96LL) + 181LL) >= 0 )
LABEL_158:
            v37 = 0;
          else
            v37 = *(_QWORD *)(v116 + 104) == 0;
        }
      }
LABEL_54:
      v159 = v37;
      sub_724C70(a2, 1);
      sub_620D80((__m128i *)a2[11].m128i_i16, v159);
      if ( a3 )
        goto LABEL_32;
      goto LABEL_25;
    case 74:
    case 75:
      v60 = *(__int64 **)(v6 + 64);
      if ( v12 == 74 )
      {
        v64 = v60[7];
        v165 = (__int64 *)v60[2];
        v61 = *v165;
      }
      else
      {
        v61 = *v60;
        v62 = *(_BYTE *)(*v60 + 140);
        for ( j = *v60; v62 == 12; v62 = *(_BYTE *)(j + 140) )
          j = *(_QWORD *)(j + 160);
        if ( v62 == 13 )
        {
          v165 = *(__int64 **)(v6 + 64);
          v64 = *(_QWORD *)(j + 160);
        }
        else
        {
          v113 = sub_72C930(a1);
          v165 = v60;
          v61 = *v60;
          v64 = v113;
        }
      }
      v141 = v61;
      v149 = v64;
      v65 = sub_8DBE70(v64);
      v66 = v141;
      if ( v65 )
        goto LABEL_37;
      v67 = v141;
      v142 = v149;
      v150 = v66;
      if ( (unsigned int)sub_8DBE70(v67) )
        goto LABEL_37;
      v68 = v142;
      for ( k = v150; *(_BYTE *)(v68 + 140) == 12; v68 = *(_QWORD *)(v68 + 160) )
        ;
      while ( *(_BYTE *)(k + 140) == 12 )
        k = *(_QWORD *)(k + 160);
      v143 = k;
      v151 = (FILE *)v68;
      if ( !(unsigned int)sub_8D3A70(v68) )
      {
        if ( (unsigned int)sub_6E5430() )
          sub_6851C0(0x58Fu, (_DWORD *)v60 + 7);
        goto LABEL_105;
      }
      if ( (unsigned int)sub_8D23B0(v151) )
      {
        sub_6E5F60((_DWORD *)v60 + 7, v151, 8);
        goto LABEL_105;
      }
      if ( *(_BYTE *)(v143 + 140) != 13 )
      {
        if ( (unsigned int)sub_6E5430() )
          sub_6851C0(0xCB2u, (_DWORD *)v165 + 7);
        goto LABEL_105;
      }
      if ( *((_BYTE *)v165 + 24) != 2 || (v120 = v165[7], *(_BYTE *)(v120 + 173) != 7) )
      {
        *a5 = 1;
        goto LABEL_172;
      }
      v121 = *(_QWORD *)(v143 + 160);
      if ( !sub_8D5CE0(v151, v121) && (*(_BYTE *)(v120 + 192) & 2) == 0 )
      {
        v122 = *(_QWORD *)(v120 + 200);
        if ( v122 )
        {
          if ( !*(_QWORD *)(v122 + 128) )
          {
            v31 = 1;
            if ( (unsigned int)sub_8D3B10(v121) )
              goto LABEL_48;
            if ( *(char *)(*(_QWORD *)(*(_QWORD *)v121 + 96LL) + 181LL) < 0 )
            {
              v31 = (unsigned __int64)*(char *)(*(_QWORD *)(*(_QWORD *)&v151->_flags + 96LL) + 181LL) >> 63;
              goto LABEL_48;
            }
          }
        }
      }
      goto LABEL_172;
    case 76:
    case 77:
      v148 = *(__int64 **)(v6 + 64);
      v48 = sub_72C930(a1);
      v49 = sub_72C930(a1);
      v50 = *(_BYTE *)(v6 + 56);
      v164 = v49;
      v140 = v148;
      if ( v50 == 76 )
      {
        v48 = v148[7];
        v51 = v148[2];
        v164 = *(_QWORD *)(v51 + 56);
        v140 = *(__int64 **)(v51 + 16);
      }
      v52 = *v140;
      v138 = (__int64 *)v140[2];
      for ( m = *(_BYTE *)(*v140 + 140); m == 12; m = *(_BYTE *)(v52 + 140) )
        v52 = *(_QWORD *)(v52 + 160);
      v54 = *v138;
      for ( n = *(_BYTE *)(*v138 + 140); n == 12; n = *(_BYTE *)(v54 + 140) )
        v54 = *(_QWORD *)(v54 + 160);
      if ( v50 == 77 )
      {
        if ( m == 13 )
          v48 = *(_QWORD *)(v52 + 160);
        if ( n == 13 )
          v164 = *(_QWORD *)(v54 + 160);
      }
      v130 = v54;
      v134 = v52;
      if ( (unsigned int)sub_8DBE70(v48)
        || (unsigned int)sub_8DBE70(v164)
        || (v56 = sub_8DBE70(v134), v57 = v130, v56)
        || (v58 = v130, v131 = v134, v135 = v57, (unsigned int)sub_8DBE70(v58)) )
      {
LABEL_37:
        sub_70FD90((__int64 *)v6, (__int64)a2);
        goto LABEL_25;
      }
      for ( ii = v131; *(_BYTE *)(v48 + 140) == 12; v48 = *(_QWORD *)(v48 + 160) )
        ;
      for ( ; *(_BYTE *)(v164 + 140) == 12; v164 = *(_QWORD *)(v164 + 160) )
        ;
      v132 = v135;
      v136 = ii;
      if ( !(unsigned int)sub_8D3A70(v48) )
      {
        if ( (unsigned int)sub_6E5430() )
          sub_6851C0(0x58Fu, (_DWORD *)v148 + 7);
        goto LABEL_105;
      }
      if ( (unsigned int)sub_8D23B0(v48) )
      {
        sub_6E5F60((_DWORD *)v148 + 7, (FILE *)v48, 8);
LABEL_105:
        sub_72C970(a2);
        goto LABEL_25;
      }
      if ( !(unsigned int)sub_8D3A70(v164) )
      {
        v123 = v148[2];
        if ( (unsigned int)sub_6E5430() )
          sub_6851C0(0x58Fu, (_DWORD *)(v123 + 28));
        goto LABEL_105;
      }
      v129 = v132;
      v133 = v136;
      v137 = sub_8D23B0(v164);
      if ( v137 )
      {
        sub_6E5F60((_DWORD *)(v148[2] + 28), (FILE *)v164, 8);
        goto LABEL_105;
      }
      if ( *(_BYTE *)(v133 + 140) != 13 )
      {
        if ( (unsigned int)sub_6E5430() )
          sub_6851C0(0xCB2u, (_DWORD *)v140 + 7);
        goto LABEL_105;
      }
      if ( *(_BYTE *)(v129 + 140) != 13 )
      {
        if ( (unsigned int)sub_6E5430() )
          sub_6851C0(0xCB2u, (_DWORD *)v138 + 7);
        goto LABEL_105;
      }
      if ( *((_BYTE *)v140 + 24) == 2
        && *((_BYTE *)v138 + 24) == 2
        && *(_BYTE *)(v140[7] + 173) == 7
        && *(_BYTE *)(v138[7] + 173) == 7 )
      {
        if ( !sub_8D5CE0(v133, v48) && !sub_8D5CE0(v129, v164) )
        {
          v124 = v140[7];
          if ( (*(_BYTE *)(v124 + 192) & 2) == 0 )
          {
            v125 = v138[7];
            if ( (*(_BYTE *)(v125 + 192) & 2) == 0 )
            {
              v126 = *(_QWORD *)(v124 + 200);
              if ( v126 )
              {
                v127 = *(_QWORD *)(v125 + 200);
                if ( v127 )
                {
                  if ( *(char *)(*(_QWORD *)(*(_QWORD *)v48 + 96LL) + 181LL) < 0
                    && *(char *)(*(_QWORD *)(*(_QWORD *)v164 + 96LL) + 181LL) < 0 )
                  {
                    v128 = *(_QWORD *)(v126 + 128);
                    if ( v128 == *(_QWORD *)(v127 + 128) )
                      v137 = v128 < sub_8E4DB0(v48, v164);
                  }
                }
              }
            }
          }
        }
      }
      else
      {
        *a5 = 1;
      }
      sub_724C70(a2, 1);
      sub_620D80((__m128i *)a2[11].m128i_i16, v137);
      if ( a3 )
        goto LABEL_32;
      goto LABEL_25;
    case 78:
      v70 = *(_QWORD *)(v6 + 64);
      v71 = *(_BYTE **)(v70 + 56);
      if ( v71[120] == 8
        || *(_BYTE *)(*(_QWORD *)v71 + 80LL) == 19 && (*(_BYTE *)(*(_QWORD *)(*(_QWORD *)v71 + 88LL) + 160LL) & 2) != 0 )
      {
        goto LABEL_24;
      }
      v72 = *(_QWORD *)(*(_QWORD *)(v70 + 16) + 56LL);
      if ( (unsigned int)sub_8DBE70(v72) )
        goto LABEL_24;
      sub_724C70(a2, 1);
      v73 = sub_8B5250(v71, v72);
      sub_620D80((__m128i *)a2[11].m128i_i16, v73);
      goto LABEL_25;
    case 80:
    case 81:
      v43 = *(_QWORD *)(v6 + 64);
      v168[0] = 0;
      for ( jj = *(_QWORD *)(v43 + 56); *(_BYTE *)(jj + 140) == 12; jj = *(_QWORD *)(jj + 160) )
        ;
      v162 = v12;
      v45 = sub_8DBE70(jj);
      v46 = v162;
      if ( v45 )
        goto LABEL_24;
      v163 = *(_QWORD *)(v43 + 16);
      if ( v46 == 81 )
      {
        if ( (unsigned int)sub_8DBE70(**(_QWORD **)(v43 + 16)) )
          goto LABEL_24;
        v163 = *(_QWORD *)(v43 + 16);
        if ( *(_BYTE *)(v163 + 24) != 2
          || (v117 = *(_QWORD *)(v163 + 56), *(_BYTE *)(v117 + 173) != 1)
          || !(unsigned int)sub_8D2780(*(_QWORD *)(v117 + 128))
          || (int)sub_6210B0(v117, 0) < 0 )
        {
          v168[0] = 1;
          goto LABEL_195;
        }
        for ( kk = sub_620FD0(v117, v168); ; --kk )
        {
          v118 = *(_BYTE *)(jj + 140);
          if ( !kk )
            break;
          while ( v118 == 12 )
          {
            jj = *(_QWORD *)(jj + 160);
            v118 = *(_BYTE *)(jj + 140);
          }
          if ( v118 != 8 )
          {
            kk = 0;
            goto LABEL_68;
          }
          jj = sub_8D4050(jj);
        }
        while ( 1 )
        {
          v119 = *(_BYTE *)(jj + 140);
          if ( v119 != 12 )
            break;
          jj = *(_QWORD *)(jj + 160);
        }
        if ( v119 == 8 )
          kk = *(_QWORD *)(jj + 176);
      }
      else
      {
        kk = sub_8DBF80(jj);
      }
LABEL_68:
      if ( v168[0] )
      {
LABEL_195:
        if ( (unsigned int)sub_6E5430() )
          sub_6851C0(0xC64u, (_DWORD *)(v163 + 28));
        sub_724C70(a2, 0);
        goto LABEL_25;
      }
      sub_724C70(a2, 1);
      sub_620DE0((__m128i *)a2[11].m128i_i16, kk);
LABEL_70:
      if ( a3 )
LABEL_32:
        a2[9].m128i_i64[0] = v6;
LABEL_25:
      result = *(_QWORD *)v6;
      a2[8].m128i_i64[0] = *(_QWORD *)v6;
      return result;
    case 112:
    case 113:
      v38 = *(_QWORD *)(v6 + 64);
      v160 = *(_BYTE *)(v6 + 56);
      v147 = *(_QWORD *)(v38 + 56);
      if ( (unsigned int)sub_8DBE70(v147) )
        goto LABEL_24;
      v39 = *(_QWORD *)(v38 + 16);
      v40 = v160;
      if ( !v39 )
        goto LABEL_60;
      do
      {
        v161 = v40;
        if ( (unsigned int)sub_8DBE70(*(_QWORD *)(v39 + 56)) )
        {
LABEL_24:
          sub_724C70(a2, 12);
          sub_7249B0(a2, 1);
          a2[11].m128i_i64[1] = v6;
          goto LABEL_25;
        }
        v39 = *(_QWORD *)(v39 + 16);
        v40 = v161;
      }
      while ( v39 );
LABEL_60:
      v41 = sub_6C4500(v40, v147, v6);
      *(_BYTE *)(v38 + 26) |= 0x40u;
      v20 = v41;
      for ( mm = *(_QWORD *)(v38 + 16); mm; mm = *(_QWORD *)(mm + 16) )
        *(_BYTE *)(mm + 26) |= 0x40u;
LABEL_31:
      sub_724C70(a2, 1);
      sub_620D80((__m128i *)a2[11].m128i_i16, v20);
      if ( a3 )
        goto LABEL_32;
      goto LABEL_25;
    default:
LABEL_21:
      sub_721090(a1);
  }
}
