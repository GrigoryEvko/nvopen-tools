// Function: sub_8B3500
// Address: 0x8b3500
//
__int64 __fastcall sub_8B3500(__m128i *a1, __int64 a2, __int64 *a3, __int64 a4, unsigned int a5)
{
  const __m128i *v8; // rax
  __int8 v10; // cl
  const __m128i *v11; // rdi
  const __m128i *v12; // rax
  __int64 v13; // rdx
  unsigned int v14; // r15d
  unsigned int v16; // r14d
  int v17; // r15d
  __int8 v18; // al
  __int64 v19; // rdx
  __int64 v20; // rcx
  __int64 v21; // r8
  __m128i *v22; // rdi
  __m128i *v23; // rsi
  __m128i *v24; // rax
  __int64 v25; // rcx
  __int64 v26; // r8
  const __m128i *v27; // r10
  __m128i *v28; // rdi
  __int8 v29; // al
  __int8 v30; // dl
  __m128i *v31; // rax
  const __m128i *v32; // rdi
  __int64 v33; // r15
  __int64 v34; // rsi
  bool v35; // al
  __m128i *v36; // r14
  _QWORD *v37; // r14
  __int64 v38; // r15
  _QWORD *v39; // r13
  int v40; // eax
  __int64 v41; // rsi
  __int64 v42; // r15
  __int64 v43; // r8
  __m128i *v44; // rdi
  __int64 v45; // r15
  int v46; // eax
  int v47; // edx
  unsigned int v48; // r15d
  __int64 v49; // r15
  __int64 v50; // rax
  char v51; // al
  __int64 v52; // rsi
  int v53; // eax
  __int64 v54; // rdi
  __int64 v55; // r8
  __int64 v56; // rsi
  __int64 v57; // rcx
  __int64 *v58; // r9
  __int64 *v59; // r11
  __int64 *v60; // r15
  int v61; // eax
  __int64 *v62; // rbx
  __int64 v63; // r10
  __int64 v64; // r9
  __int8 v65; // al
  __int64 v66; // rax
  __int64 *v67; // rax
  __int64 *v68; // r12
  __m128i *v69; // rdi
  __int16 v70; // dx
  __int64 v71; // rax
  __int64 v72; // rdi
  __int64 v73; // rdi
  __int64 v74; // rdi
  int v75; // eax
  __int64 v76; // rax
  __int64 v77; // rdi
  char v78; // al
  __int64 v79; // rdx
  char v80; // al
  int v81; // eax
  int v82; // eax
  __int64 v83; // rax
  int v84; // eax
  __int64 v85; // rax
  __int64 *v86; // r11
  int v87; // ebx
  __int64 v88; // r10
  __m128i *i; // rax
  __int64 v90; // r11
  __int64 v91; // rax
  const __m128i *j; // rax
  __int64 v93; // rsi
  __int64 v94; // rax
  int v95; // eax
  __int64 v96; // r10
  __int64 v98; // rax
  _BYTE *v99; // r14
  __int64 v100; // r15
  __int64 v101; // rdi
  unsigned int v102; // eax
  __int64 v103; // [rsp+0h] [rbp-A0h]
  __int64 v104; // [rsp+0h] [rbp-A0h]
  __int64 v105; // [rsp+8h] [rbp-98h]
  __int64 v106; // [rsp+8h] [rbp-98h]
  __int64 v107; // [rsp+8h] [rbp-98h]
  __int64 v108; // [rsp+8h] [rbp-98h]
  __int64 v109; // [rsp+10h] [rbp-90h]
  __int64 v110; // [rsp+10h] [rbp-90h]
  int v111; // [rsp+10h] [rbp-90h]
  __int64 v112; // [rsp+10h] [rbp-90h]
  __int64 v113; // [rsp+10h] [rbp-90h]
  unsigned int v114; // [rsp+18h] [rbp-88h]
  __int64 v115; // [rsp+18h] [rbp-88h]
  __int64 v116; // [rsp+18h] [rbp-88h]
  unsigned int v117; // [rsp+20h] [rbp-80h]
  __int64 v118; // [rsp+20h] [rbp-80h]
  __int64 v119; // [rsp+20h] [rbp-80h]
  int v120; // [rsp+28h] [rbp-78h]
  __int64 v121; // [rsp+28h] [rbp-78h]
  __int64 *v122; // [rsp+30h] [rbp-70h]
  __int64 v123; // [rsp+30h] [rbp-70h]
  int v124; // [rsp+38h] [rbp-68h]
  __int64 v125; // [rsp+38h] [rbp-68h]
  __int64 v126; // [rsp+38h] [rbp-68h]
  __m128i *v127; // [rsp+40h] [rbp-60h] BYREF
  __m128i *v128[2]; // [rsp+48h] [rbp-58h] BYREF
  int v129; // [rsp+5Ch] [rbp-44h] BYREF
  __int64 v130; // [rsp+60h] [rbp-40h] BYREF
  __int64 *v131[7]; // [rsp+68h] [rbp-38h] BYREF

  v128[0] = a1;
  v8 = (const __m128i *)sub_8D2250(a2);
  v127 = (__m128i *)v8;
  if ( v8 == a1 )
    return 1;
  v10 = v8[8].m128i_i8[12];
  v11 = v8;
  if ( (unsigned __int8)(v10 - 9) <= 2u )
  {
    v12 = *(const __m128i **)(v8[10].m128i_i64[1] + 256);
    if ( v12 )
    {
      v127 = *(__m128i **)(v11[10].m128i_i64[1] + 256);
      v10 = v12[8].m128i_i8[12];
      v11 = v12;
    }
  }
  if ( (unsigned __int8)(v128[0][8].m128i_i8[12] - 9) <= 2u && *(_QWORD *)(v128[0][10].m128i_i64[1] + 256) )
    v128[0] = *(__m128i **)(a1[10].m128i_i64[1] + 256);
  if ( (a5 & 8) != 0 )
  {
    if ( (v10 & 0xFB) != 8 )
      goto LABEL_10;
    if ( (unsigned int)sub_8D4C10(v11, dword_4F077C4 != 2) )
    {
      sub_73E8D0(v128, (const __m128i **)&v127);
      v127 = (__m128i *)sub_8D2220(v127);
    }
    v11 = v127;
    v10 = v127[8].m128i_i8[12];
  }
  if ( v10 == 12 )
  {
    if ( (v11[11].m128i_i8[10] & 8) != 0 )
      return 1;
    goto LABEL_23;
  }
LABEL_10:
  if ( unk_4D04318 || (v11[5].m128i_i8[9] & 4) == 0 )
    goto LABEL_24;
  if ( (unsigned __int8)(v10 - 9) <= 2u )
  {
    if ( (v11[11].m128i_i8[1] & 0x20) != 0 )
    {
      v13 = *(_QWORD *)(*(_QWORD *)(v11->m128i_i64[0] + 96) + 72LL);
      switch ( *(_BYTE *)(v13 + 80) )
      {
        case 4:
        case 5:
          v66 = *(_QWORD *)(*(_QWORD *)(v13 + 96) + 80LL);
          break;
        case 6:
          v66 = *(_QWORD *)(*(_QWORD *)(v13 + 96) + 32LL);
          break;
        case 9:
        case 0xA:
          v66 = *(_QWORD *)(*(_QWORD *)(v13 + 96) + 56LL);
          break;
        case 0x13:
        case 0x14:
        case 0x15:
        case 0x16:
          v66 = *(_QWORD *)(v13 + 88);
          break;
        default:
          BUG();
      }
      if ( (*(_BYTE *)(v66 + 160) & 2) != 0 )
        return 1;
    }
    goto LABEL_24;
  }
  if ( v10 != 12 )
    goto LABEL_24;
LABEL_23:
  if ( (unsigned __int8)(v11[11].m128i_i8[8] - 11) <= 1u )
    return 0;
LABEL_24:
  v16 = a5 & 0x880;
  v14 = sub_8D3D40(v11);
  if ( v14 )
  {
    if ( (v127[8].m128i_i8[12] & 0xFB) != 8 )
      goto LABEL_26;
    if ( (unsigned int)sub_8D4C10(v127, dword_4F077C4 != 2) )
      sub_73E8D0(v128, (const __m128i **)&v127);
    if ( (v127[8].m128i_i8[12] & 0xFB) != 8 || !(unsigned int)sub_8D4C10(v127, dword_4F077C4 != 2) )
    {
LABEL_26:
      v17 = 0;
      if ( a4 )
        v17 = *(_DWORD *)(sub_892BC0(a4) + 4);
      v127 = (__m128i *)sub_8D21C0(v127);
      v18 = v127[10].m128i_i8[0];
      if ( !v18 )
      {
        v19 = v127[10].m128i_i64[1];
        if ( *(_DWORD *)(v19 + 28) == v17 )
        {
          v67 = sub_8A4360(a4, a3, (unsigned int *)(v19 + 24), 0, 0);
          v23 = (__m128i *)v67[4];
          v68 = v67;
          if ( !v23 || (*((_BYTE *)v67 + 25) & 1) != 0 )
          {
            v69 = v128[0];
            if ( !unk_4D04314 )
            {
              v128[0] = (__m128i *)sub_8E3250(v128[0]);
              v69 = v128[0];
            }
            v128[0] = (__m128i *)sub_8E32F0(v69);
            if ( (unsigned int)sub_8D2B80(v128[0]) )
              sub_73C7D0(v128);
            v68[4] = (__int64)v128[0]->m128i_i64;
            LOBYTE(v70) = (a5 & 0x100) != 0;
            *((_WORD *)v68 + 12) = v68[3] & 0xFEEF | (v70 << 8) | (16 * ((a5 >> 11) & 1));
            return 1;
          }
          if ( (a5 & 0x100) != 0 )
            return 1;
          v22 = v128[0];
          if ( v23 == v128[0] )
            return 1;
        }
        else
        {
          if ( (v127[10].m128i_i8[1] & 1) != 0 )
            return 1;
          if ( (unsigned int)sub_8D3EA0(v127) )
            return 1;
          if ( (unsigned int)sub_8D3F00(v127) )
            return 1;
          v22 = v128[0];
          v23 = v127;
          if ( v128[0] == v127 )
            return 1;
        }
        return (unsigned int)sub_8D97D0(v22, v23, 0, v20, v21) != 0;
      }
      if ( v18 == 2 )
        return 1;
      v31 = (__m128i *)sub_8D21C0(v128[0]);
      v32 = v127;
      v128[0] = v31;
      if ( (v127[5].m128i_i8[9] & 4) != 0 )
      {
        if ( !unk_4D04318 )
        {
          v71 = *(_QWORD *)(v127[2].m128i_i64[1] + 32);
          if ( (*(_BYTE *)(v71 + 177) & 0x20) != 0 )
            return 1;
          v72 = *(_QWORD *)(*(_QWORD *)(v71 + 168) + 256LL);
          if ( !v72 )
            v72 = *(_QWORD *)(v127[2].m128i_i64[1] + 32);
          return (unsigned int)sub_8DC1A0(v72);
        }
        if ( (v31[5].m128i_i8[9] & 4) != 0 && *(_QWORD *)v31->m128i_i64[0] == *(_QWORD *)v127->m128i_i64[0] )
        {
          v33 = *(_QWORD *)(v31[2].m128i_i64[1] + 32);
          v34 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(v127[2].m128i_i64[1] + 32) + 168LL) + 256LL);
          if ( v34 )
          {
            if ( (unsigned int)sub_8B3500(v33, v34, a3, a4, a5 & 0x880) )
              return 1;
            v32 = v127;
          }
          return (unsigned int)sub_8B3500(v33, *(_QWORD *)(v32[2].m128i_i64[1] + 32), a3, a4, a5 & 0x880) != 0;
        }
      }
    }
    return 0;
  }
  v24 = (__m128i *)sub_8D21C0(v128[0]);
  v27 = v127;
  v28 = v24;
  v128[0] = v24;
  v29 = v127[8].m128i_i8[12];
  v30 = v28[8].m128i_i8[12];
  if ( v29 == 10 )
  {
    if ( (unsigned __int8)(v30 - 9) <= 1u )
      goto LABEL_56;
    v35 = 0;
    goto LABEL_54;
  }
  if ( v30 == 10 )
  {
    if ( v29 != 9 )
      return 0;
LABEL_56:
    v14 = sub_8B3280((__int64)v28, (__int64)v127, a3, a4, a5);
    if ( v14 )
      return v14;
    if ( (a5 & 1) != 0 )
    {
      v36 = v128[0];
      if ( dword_4F077C4 == 2 )
      {
        if ( (unsigned int)sub_8D23B0(v128[0]) && (unsigned int)sub_8D3A70(v36) )
          sub_8AD220((__int64)v36, 0);
        v36 = v128[0];
      }
      v37 = *(_QWORD **)v36[10].m128i_i64[1];
      if ( v37 )
      {
        v122 = a3;
        v38 = 0;
        v39 = v37;
        do
        {
          v131[0] = 0;
          v40 = sub_8B3280(v39[5], (__int64)v127, (__int64 *)v131, a4, a5);
          if ( v131[0] )
          {
            v124 = v40;
            sub_725130(v131[0]);
            v40 = v124;
          }
          if ( v40 )
          {
            v41 = v39[5];
            if ( v38 )
            {
              if ( v38 != v41 && !sub_8D5CE0(v38, v41) )
              {
                if ( !sub_8D5CE0(v39[5], v38) )
                  return 0;
                v38 = v39[5];
              }
            }
            else
            {
              v38 = v39[5];
            }
          }
          v39 = (_QWORD *)*v39;
        }
        while ( v39 );
        if ( v38 )
          return (unsigned int)sub_8B3280(v38, (__int64)v127, v122, a4, a5);
      }
    }
    return 0;
  }
  if ( v29 != v30 )
  {
    v35 = v29 != 9;
LABEL_54:
    if ( v30 != 11 || v35 )
      return 0;
    goto LABEL_56;
  }
  switch ( v29 )
  {
    case 6:
      v65 = v28[10].m128i_i8[8];
      if ( (((unsigned __int8)v65 ^ v127[10].m128i_i8[8]) & 1) != 0 )
        return 0;
      if ( !dword_4D04474 || (v65 & 1) == 0 )
        return (unsigned int)sub_8B3500(v28[10].m128i_i64[0], v27[10].m128i_i64[0], a3, a4, v16);
      v87 = sub_8D30C0(v28);
      if ( v87 != (unsigned int)sub_8D30C0(v127) )
        return 0;
      v28 = v128[0];
      v27 = v127;
      return (unsigned int)sub_8B3500(v28[10].m128i_i64[0], v27[10].m128i_i64[0], a3, a4, v16);
    case 7:
      if ( !(unsigned int)sub_8B3500(v28[10].m128i_i64[0], v127[10].m128i_i64[0], a3, a4, a5 & 0x880) )
        return 0;
      v56 = v128[0][10].m128i_i64[1];
      v57 = v127[10].m128i_i64[1];
      v126 = v56;
      v123 = v57;
      if ( ((*(_BYTE *)(v57 + 16) ^ *(_BYTE *)(v56 + 16)) & 1) != 0 )
        return 0;
      v130 = 0;
      v58 = *(__int64 **)v57;
      v59 = *(__int64 **)v56;
      if ( !*(_QWORD *)v57 || !v59 )
      {
        if ( v59 == v58 )
          goto LABEL_179;
        goto LABEL_168;
      }
      v117 = v14;
      v60 = *(__int64 **)v56;
      v61 = a5 & 0x10;
      v114 = a5;
      v62 = *(__int64 **)v57;
      v120 = v61;
      do
      {
        if ( (*((_BYTE *)v62 + 33) & 1) != 0 )
        {
          if ( *v62 )
            goto LABEL_99;
          v73 = v62[10];
          if ( v73 && !v130 )
            sub_869480(v73, a4, a3, &v130);
        }
        v63 = v60[1];
        v64 = v62[1];
        if ( dword_4D04474 )
        {
          if ( v120 )
          {
            v105 = v60[1];
            v109 = v62[1];
            v75 = sub_8D3110(v109);
            v64 = v109;
            v63 = v105;
            if ( v75 )
            {
              v76 = sub_8D46C0(v109);
              v64 = v109;
              v63 = v105;
              v77 = v76;
              v78 = *(_BYTE *)(v76 + 140);
              v79 = v77;
              if ( v78 == 12 )
              {
                do
                {
                  v79 = *(_QWORD *)(v79 + 160);
                  v80 = *(_BYTE *)(v79 + 140);
                }
                while ( v80 == 12 );
                if ( v80 != 14 )
                  goto LABEL_106;
                v103 = v105;
                v106 = v109;
                v110 = v79;
                v81 = sub_8D4C10(v77, dword_4F077C4 != 2);
                v79 = v110;
                v64 = v106;
                v63 = v103;
                if ( v81 )
                  goto LABEL_106;
              }
              else if ( v78 != 14 )
              {
                goto LABEL_106;
              }
              v111 = *(_DWORD *)(*(_QWORD *)(v79 + 168) + 28LL);
              v82 = 0;
              if ( a4 )
              {
                v104 = v63;
                v107 = v64;
                v83 = sub_892BC0(a4);
                v63 = v104;
                v64 = v107;
                v82 = *(_DWORD *)(v83 + 4);
              }
              if ( v111 == v82 )
              {
                v108 = v64;
                v112 = v63;
                v84 = sub_8D3070(v63);
                v63 = v112;
                v64 = v108;
                if ( v84 )
                {
                  v85 = sub_8D46C0(v108);
                  v63 = v112;
                  v64 = v85;
                }
              }
            }
          }
        }
LABEL_106:
        if ( !(unsigned int)sub_8B3500(v63, v64, a3, a4, v16 | (*((unsigned __int8 *)v60 + 33) << 11) & 0x800) )
        {
          v14 = v117;
          if ( !v130 )
            return v14;
          sub_866BE0(v130);
          return 0;
        }
LABEL_99:
        v60 = (__int64 *)*v60;
        if ( v130 )
          sub_866B90(v130);
        else
          v62 = (__int64 *)*v62;
      }
      while ( v60 && v62 );
      v58 = v62;
      v86 = v60;
      a5 = v114;
      v14 = v117;
      if ( v86 == v58 )
        goto LABEL_177;
LABEL_168:
      if ( !v58 || (*((_BYTE *)v58 + 33) & 1) == 0 || *v58 )
      {
        if ( v130 )
          sub_866BE0(v130);
        return v14;
      }
LABEL_177:
      if ( v130 )
        sub_866BE0(v130);
LABEL_179:
      v88 = *(_QWORD *)(v56 + 40);
      v121 = *(_QWORD *)(v123 + 40);
      if ( !v88 || !*(_QWORD *)(v123 + 40) )
      {
        if ( ((*(_BYTE *)(v123 + 18) ^ *(_BYTE *)(v56 + 18)) & 0x7F) != 0 || v88 != v121 && (a5 & 2) == 0 )
          return v14;
        goto LABEL_195;
      }
      for ( i = v128[0]; i[8].m128i_i8[12] == 12; i = (__m128i *)i[10].m128i_i64[0] )
        ;
      v90 = *(_QWORD *)(i[10].m128i_i64[1] + 40);
      if ( v90 )
      {
        v118 = *(_QWORD *)(v56 + 40);
        v91 = sub_8D71D0(v128[0]);
        v88 = v118;
        v90 = v91;
      }
      for ( j = v127; j[8].m128i_i8[12] == 12; j = (const __m128i *)j[10].m128i_i64[0] )
        ;
      v93 = *(_QWORD *)(j[10].m128i_i64[1] + 40);
      if ( v93 )
      {
        v115 = v88;
        v119 = v90;
        v94 = sub_8D71D0(v127);
        v88 = v115;
        v90 = v119;
        v93 = v94;
      }
      v113 = v88;
      v116 = v90;
      v95 = sub_8DBE70(v93);
      v96 = v113;
      if ( v95 )
      {
        v102 = sub_8B3500(v116, v93, a3, a4, v16);
        v96 = v113;
        if ( !v102 )
        {
LABEL_191:
          if ( !dword_4D0443C )
            return v14;
          if ( !((a5 & 0x20) != 0 ? sub_8D5CE0(v121, v96) != 0 : sub_8D5CE0(v96, v121) != 0) )
            return v14;
          goto LABEL_195;
        }
        v14 = v102;
      }
      else
      {
        if ( ((*(_BYTE *)(v123 + 18) ^ *(_BYTE *)(v126 + 18)) & 0x7F) != 0 )
          goto LABEL_191;
LABEL_195:
        v14 = 1;
      }
      if ( !dword_4F06978 )
        return v14;
      v98 = *(_QWORD *)(v127[10].m128i_i64[1] + 56);
      if ( v98 )
      {
        if ( (*(_BYTE *)v98 & 1) != 0 )
        {
          v99 = *(_BYTE **)(v98 + 8);
          if ( v99 )
          {
            if ( unk_4F06974 )
            {
              if ( (a5 & 0x40) == 0 )
              {
                v100 = *(_QWORD *)(v128[0][10].m128i_i64[1] + 56);
                if ( !sub_70FCE0((__int64)v99) )
                {
                  if ( v100 && (v101 = *(_QWORD *)(v100 + 8)) != 0 )
                  {
                    v131[0] = *(__int64 **)(v100 + 8);
                    return (unsigned int)sub_8B46F0(v101, v99, a3, a4, a5);
                  }
                  else
                  {
                    v131[0] = sub_724DC0();
                    sub_72C470(v100 != 0, (__int64)v131[0]);
                    v14 = sub_8B46F0(v131[0], v99, a3, a4, a5);
                    if ( v14 )
                    {
                      sub_7296C0(&v129);
                      sub_724E50((__int64 *)v131, v99);
                      sub_729730(v129);
                    }
                    else
                    {
                      sub_724E30((__int64)v131);
                    }
                  }
                  return v14;
                }
              }
            }
          }
        }
      }
      return 1;
    case 8:
      if ( (v28[10].m128i_i8[9] & 1) != 0 )
        return 0;
      v51 = v127[10].m128i_i8[8];
      v52 = v127[11].m128i_i64[0];
      if ( v28[10].m128i_i8[8] < 0 )
      {
        if ( v51 < 0 )
        {
          v74 = v28[11].m128i_i64[0];
          if ( !v74 || !v52 )
            return 0;
          v53 = sub_8B46F0(v74, v52, a3, a4, a5);
          goto LABEL_88;
        }
      }
      else if ( v51 < 0 )
      {
        if ( !v52 )
          return 0;
        v53 = sub_8B4E50(v28[11].m128i_i64[0], v52, a3, a4, a5);
LABEL_88:
        if ( !v53 )
          return 0;
        v28 = v128[0];
        v27 = v127;
        goto LABEL_90;
      }
      if ( v52 != v28[11].m128i_i64[0] )
        return 0;
LABEL_90:
      v54 = v28[10].m128i_i64[0];
      v55 = v16 | 8;
      if ( (a5 & 1) == 0 )
        v55 = a5 & 0x880;
      return (unsigned int)sub_8B3500(v54, v27[10].m128i_i64[0], a3, a4, v55);
    case 9:
    case 11:
      goto LABEL_56;
    case 12:
      v46 = sub_8D4C10(v28, dword_4F077C4 != 2);
      v47 = 0;
      v48 = v46 & 0xFFFFFF8F;
      if ( (v127[8].m128i_i8[12] & 0xFB) == 8 )
        v47 = sub_8D4C10(v127, dword_4F077C4 != 2) & 0xFFFFFF8F;
      if ( v47 != v48 )
        return 0;
      v49 = sub_8D2220(v128[0][10].m128i_i64[0]);
      v50 = sub_8D2220(v127[10].m128i_i64[0]);
      return (unsigned int)sub_8B3500(v49, v50, a3, a4, a5 & 0x880);
    case 13:
      v42 = v127[10].m128i_i64[0];
      v125 = v28[10].m128i_i64[0];
      if ( !(unsigned int)sub_8B3500(v125, v42, a3, a4, a5 & 0x880)
        && ((a5 & 1) == 0 || !dword_4D0443C || !sub_8D5CE0(v42, v125)) )
      {
        return 0;
      }
      v43 = v16 | 0x20;
      v44 = (__m128i *)v128[0][10].m128i_i64[1];
      v45 = v127[10].m128i_i64[1];
      if ( v44[8].m128i_i8[12] != 7 )
        return (unsigned int)sub_8B3500(v44, v45, a3, a4, v43);
      if ( *(_BYTE *)(v45 + 140) == 7 )
      {
        if ( dword_4F06978 )
LABEL_154:
          v43 = v16 | 0x21;
      }
      else
      {
        v44 = sub_73F430(v44, 0);
        v43 = v16 | 0x20;
        if ( dword_4F06978 && v44[8].m128i_i8[12] == 7 )
          goto LABEL_154;
      }
      return (unsigned int)sub_8B3500(v44, v45, a3, a4, v43);
    case 15:
      if ( !(unsigned int)sub_8DBF30(v28) && !(unsigned int)sub_8DBF30(v127) )
      {
        v22 = v127;
        v23 = v128[0];
        v14 = 1;
        if ( v127 != v128[0] )
          return (unsigned int)sub_8D97D0(v22, v23, 0, v20, v21) != 0;
      }
      return v14;
    default:
      v14 = 1;
      if ( v28 != v127 )
        return (unsigned int)sub_8D97D0(v127, v28, 0, v25, v26) != 0;
      return v14;
  }
}
