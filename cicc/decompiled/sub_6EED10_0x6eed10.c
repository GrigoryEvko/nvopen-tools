// Function: sub_6EED10
// Address: 0x6eed10
//
__int64 __fastcall sub_6EED10(__int64 a1, int *a2, __int64 a3, __int64 a4, __int64 a5, _QWORD *a6)
{
  unsigned int v6; // r15d
  unsigned int v8; // r13d
  _QWORD *v9; // r12
  __int64 v11; // rdi
  char v12; // al
  int v13; // edx
  __int64 result; // rax
  bool v15; // cl
  __int64 v16; // rax
  __int64 v17; // rdi
  __int64 v18; // rax
  int v19; // eax
  __int64 v20; // rax
  __int64 v21; // rax
  __int64 v22; // rsi
  __int64 v23; // r13
  __int64 v24; // rax
  __int64 v25; // rax
  _QWORD *v26; // r10
  unsigned __int8 v27; // r11
  int v28; // eax
  __int64 v29; // r13
  __int64 v30; // rax
  int v31; // eax
  _QWORD *v32; // r10
  __int64 v33; // rax
  __int64 v34; // rax
  __int64 v35; // rdx
  __int64 v36; // rcx
  __int64 v37; // rdi
  __int64 v38; // r8
  __int64 v39; // r9
  __int64 v40; // rax
  __int64 v41; // rsi
  __int64 v42; // rax
  __int64 v43; // rax
  __int64 v44; // rax
  int v45; // eax
  __int64 v46; // rax
  _QWORD *v47; // r10
  __int64 v48; // rax
  __int64 v49; // rsi
  int v50; // eax
  int v51; // eax
  __int64 v52; // rax
  __int64 v53; // rax
  __int64 v54; // rax
  __int64 v55; // rax
  __int64 v56; // rax
  bool v57; // zf
  __int64 v58; // rax
  int v59; // eax
  int v60; // r11d
  int v61; // eax
  int v62; // eax
  __int64 v63; // rdi
  int v64; // eax
  __int64 v65; // rax
  int v66; // eax
  int v67; // eax
  int v68; // eax
  __int64 v69; // rax
  __int64 v70; // rdx
  unsigned int v71; // eax
  int v72; // eax
  int v73; // eax
  int v74; // eax
  unsigned int v75; // eax
  __int64 v76; // rax
  __int64 v77; // rax
  _QWORD *v78; // [rsp+8h] [rbp-98h]
  int v79; // [rsp+10h] [rbp-90h]
  __int64 v80; // [rsp+18h] [rbp-88h]
  _QWORD *v81; // [rsp+18h] [rbp-88h]
  unsigned __int8 v82; // [rsp+18h] [rbp-88h]
  unsigned int v83; // [rsp+18h] [rbp-88h]
  unsigned int v84; // [rsp+20h] [rbp-80h]
  unsigned int v85; // [rsp+20h] [rbp-80h]
  unsigned int v86; // [rsp+20h] [rbp-80h]
  _QWORD *v87; // [rsp+20h] [rbp-80h]
  unsigned int v88; // [rsp+20h] [rbp-80h]
  unsigned int v89; // [rsp+20h] [rbp-80h]
  unsigned int v90; // [rsp+20h] [rbp-80h]
  unsigned __int8 v91; // [rsp+28h] [rbp-78h]
  _QWORD *v92; // [rsp+28h] [rbp-78h]
  unsigned int v93; // [rsp+28h] [rbp-78h]
  _QWORD *v94; // [rsp+28h] [rbp-78h]
  _QWORD *v95; // [rsp+28h] [rbp-78h]
  _QWORD *v96; // [rsp+28h] [rbp-78h]
  unsigned int v97; // [rsp+28h] [rbp-78h]
  _QWORD *v98; // [rsp+28h] [rbp-78h]
  int v99; // [rsp+28h] [rbp-78h]
  _QWORD *v100; // [rsp+28h] [rbp-78h]
  _QWORD *v101; // [rsp+28h] [rbp-78h]
  unsigned int v102; // [rsp+30h] [rbp-70h]
  __int64 v103; // [rsp+30h] [rbp-70h]
  _QWORD *v104; // [rsp+30h] [rbp-70h]
  _QWORD *v105; // [rsp+30h] [rbp-70h]
  __int64 v106; // [rsp+30h] [rbp-70h]
  __int64 v107; // [rsp+30h] [rbp-70h]
  _QWORD *v108; // [rsp+30h] [rbp-70h]
  unsigned int v109; // [rsp+38h] [rbp-68h]
  _QWORD *v110; // [rsp+38h] [rbp-68h]
  _QWORD *v111; // [rsp+38h] [rbp-68h]
  _QWORD *v112; // [rsp+38h] [rbp-68h]
  _QWORD *v113; // [rsp+38h] [rbp-68h]
  int v114; // [rsp+4Ch] [rbp-54h] BYREF
  int v115; // [rsp+50h] [rbp-50h] BYREF
  int v116; // [rsp+54h] [rbp-4Ch] BYREF
  __int64 v117; // [rsp+58h] [rbp-48h] BYREF
  __int64 v118; // [rsp+60h] [rbp-40h] BYREF
  _QWORD v119[7]; // [rsp+68h] [rbp-38h] BYREF

  v6 = a4;
  v8 = a3;
  v9 = (_QWORD *)a1;
  v114 = 0;
  if ( (_DWORD)a5 )
  {
    v109 = a5;
    v21 = sub_6E8430(a1);
    a5 = v109;
    v9 = (_QWORD *)v21;
  }
  v11 = v9[1];
  v12 = *((_BYTE *)v9 + 24);
  if ( !v11 )
    v11 = *v9;
  v117 = v11;
  switch ( v12 )
  {
    case 3:
      v16 = v9[7];
      v114 = 1;
      v17 = *(_QWORD *)(v16 + 120);
      a5 = dword_4D03F94;
      v117 = v17;
      if ( !dword_4D03F94
        || *(char *)(v16 + 169) >= 0
        || (v18 = *(_QWORD *)(v16 + 128)) == 0
        || (*(_BYTE *)(v18 + 32) & 1) == 0 )
      {
LABEL_15:
        if ( v8 )
        {
LABEL_16:
          v13 = v114;
LABEL_17:
          result = (__int64)v9;
          v15 = v13 != 0;
          goto LABEL_18;
        }
        goto LABEL_35;
      }
      v19 = sub_8D2E30(v17);
      v13 = v114;
      if ( !v19 )
        goto LABEL_24;
      if ( !(unsigned int)sub_8D2E30(*v9) )
      {
        v20 = sub_8D46C0(v117);
        v13 = v114;
        v117 = v20;
        goto LABEL_24;
      }
LABEL_23:
      v13 = v114;
      goto LABEL_24;
    case 24:
      if ( *((_DWORD *)v9 + 14) )
        goto LABEL_14;
      goto LABEL_23;
    case 5:
      goto LABEL_14;
    case 2:
      v24 = v9[7];
      if ( (*(_BYTE *)(v24 + 171) & 2) != 0 )
      {
        v25 = *(_QWORD *)(v24 + 144);
        if ( v25 )
        {
          if ( *(_BYTE *)(v25 + 24) == 5 )
          {
            v114 = 1;
            if ( v8 )
            {
              v13 = v8;
              result = (__int64)v9;
              v15 = 1;
              goto LABEL_18;
            }
            v9 = (_QWORD *)v25;
            goto LABEL_35;
          }
        }
      }
      goto LABEL_23;
    case 6:
LABEL_14:
      v114 = 1;
      goto LABEL_15;
    case 11:
      v114 = 1;
      v53 = sub_73C570(v11, 1, -1);
      v13 = 1;
      v117 = v53;
      goto LABEL_24;
    case 10:
      sub_721090(v11);
  }
  if ( v12 != 1 )
  {
    if ( !v12 )
      goto LABEL_14;
    goto LABEL_23;
  }
  v26 = (_QWORD *)v9[9];
  v27 = *((_BYTE *)v9 + 56);
  v110 = (_QWORD *)v26[2];
  switch ( v27 )
  {
    case 3u:
    case 4u:
      v114 = 1;
      v113 = v26;
      v31 = sub_8D32B0(*v26);
      v32 = v113;
      if ( !v31 )
        goto LABEL_23;
      goto LABEL_63;
    case 7u:
    case 8u:
    case 0x13u:
      v114 = 1;
      if ( !v8 )
        goto LABEL_35;
      goto LABEL_49;
    case 9u:
      v43 = sub_6EED10(v26, &v114, v8, v6, 0, 0);
      v13 = v114;
      if ( v114 && (v8 & 1) == 0 )
      {
        sub_73D8E0(v9, 8, *v9, 1, v43);
        v13 = v114;
      }
      goto LABEL_24;
    case 0xEu:
    case 0xFu:
      if ( (*((_BYTE *)v26 + 25) & 3) != 0 )
      {
        v114 = 1;
      }
      else
      {
        v54 = sub_6EED10(v26, &v114, v8, v6, 0, &v117);
        v13 = v114;
        v26 = (_QWORD *)v54;
        if ( !v114 )
          goto LABEL_109;
      }
      v112 = v26;
      v117 = sub_73CA70(*v9, v117);
      if ( v8 )
        goto LABEL_23;
      v9[9] = v112;
      v13 = v114;
      goto LABEL_25;
    case 0x19u:
      v44 = sub_6EED10(v26, &v114, v8, v6, a5, &v117);
      v13 = v114;
      if ( v114 && (v8 & 1) == 0 )
        v9[9] = v44;
      goto LABEL_24;
    case 0x21u:
    case 0x22u:
      if ( (*((_BYTE *)v26 + 25) & 3) != 0 )
      {
        v114 = 1;
        v117 = sub_73CA70(*v9, *v26);
      }
      goto LABEL_23;
    case 0x47u:
    case 0x48u:
      v102 = a5;
      v80 = v9[9];
      sub_6EED10(v26, &v116, 1, 0, a5, &v117);
      sub_6EED10(v110, &v118, 1, 0, v102, &v117);
      if ( v116 && (_DWORD)v118 )
      {
        v114 = 1;
        if ( v8 )
          goto LABEL_49;
        *(_QWORD *)(v80 + 16) = 0;
        v29 = sub_6EED10(v80, &v116, 0, v6, v102, 0);
        v30 = sub_6EED10(v110, &v118, 0, v6, v102, 0);
        v9[9] = v29;
        v13 = v114;
        *(_QWORD *)(v29 + 16) = v30;
        goto LABEL_25;
      }
      v13 = 0;
      if ( !v8 )
        goto LABEL_26;
      goto LABEL_17;
    case 0x49u:
      goto LABEL_47;
    case 0x5Bu:
      if ( dword_4F077C4 != 2 && !v6 )
        goto LABEL_23;
      v107 = v9[9];
      v52 = sub_6EED10(v110, &v114, v8, v6, a5, &v117);
      v13 = v114;
      if ( (v8 & 1) == 0 && v114 )
      {
        *((_BYTE *)v9 + 58) |= 1u;
        *(_QWORD *)(v107 + 16) = v52;
      }
      goto LABEL_24;
    case 0x5Cu:
      v114 = 1;
      v105 = v26;
      if ( (unsigned int)sub_8D2E30(*v26) )
      {
        v32 = v105;
LABEL_63:
        v33 = sub_8D46C0(*v32);
        v13 = v114;
        v117 = v33;
      }
      else
      {
        if ( !(unsigned int)sub_8D2E30(*v110) )
          goto LABEL_23;
        v55 = sub_8D46C0(*v110);
        v13 = v114;
        v117 = v55;
      }
      goto LABEL_24;
    case 0x5Eu:
    case 0x60u:
      if ( (*((_BYTE *)v26 + 25) & 3) != 0 )
      {
        v114 = 1;
        v119[0] = *v26;
      }
      else
      {
        v82 = *((_BYTE *)v9 + 56);
        v108 = (_QWORD *)v9[9];
        v85 = a5;
        sub_6EED10(v26, &v116, 1, v6, a5, v119);
        v26 = v108;
        v27 = v82;
        if ( !v116 )
        {
          v13 = v114;
LABEL_109:
          if ( v8 )
            goto LABEL_17;
          goto LABEL_26;
        }
        v114 = 1;
        a5 = v85;
        if ( !v8 )
        {
          v108[2] = 0;
          v56 = sub_6EED10(v108, &v116, 0, v6, v85, 0);
          v57 = v114 == 0;
          v27 = v82;
          v9[9] = v56;
          v26 = (_QWORD *)v56;
          *(_QWORD *)(v56 + 16) = v110;
          if ( v57 )
            goto LABEL_26;
        }
      }
      if ( v27 == 94 )
      {
        v37 = v119[0];
LABEL_74:
        v41 = 0;
        if ( (*(_BYTE *)(v37 + 140) & 0xFB) == 8 )
          v41 = (unsigned int)sub_8D4C10(v37, dword_4F077C4 != 2);
        v42 = sub_73CB50(v110[7], v41);
        v13 = v114;
        v117 = v42;
      }
      else
      {
        v58 = sub_73CAD0(*v26, *v110, a3, a4, a5, a6);
        v13 = v114;
        v117 = v58;
      }
      goto LABEL_24;
    case 0x5Fu:
    case 0x61u:
      v114 = 1;
      v91 = v27;
      v104 = v26;
      if ( !(unsigned int)sub_8D2E30(*v26) )
        goto LABEL_23;
      v37 = sub_8D46C0(*v104);
      if ( v91 == 95 )
        goto LABEL_74;
      v40 = sub_73CAD0(v37, *v110, v35, v36, v38, v39);
      v13 = v114;
      v117 = v40;
      goto LABEL_24;
    case 0x64u:
    case 0x65u:
      v103 = v9[9];
      v34 = sub_6EED10(v110, &v114, v8, v6, a5, &v117);
      v13 = v114;
      if ( (v8 & 1) == 0 && v114 )
        *(_QWORD *)(v103 + 16) = v34;
      goto LABEL_24;
    case 0x67u:
      v106 = v110[2];
      if ( dword_4F077C4 == 2 )
      {
        v86 = a5;
        v96 = (_QWORD *)v9[9];
        v59 = sub_8D2600(v117);
        v26 = v96;
        a5 = v86;
        if ( v59 )
          goto LABEL_88;
        if ( v6 )
        {
LABEL_127:
          v87 = v26;
          v97 = a5;
          sub_6EED10(v110, &v115, 1, 1, a5, &v118);
          sub_6EED10(v106, &v116, 1, 1, v97, v119);
          a5 = v97;
          v26 = v87;
          if ( !v115 || !v116 )
            goto LABEL_129;
          v88 = v97;
          v98 = v26;
          v62 = sub_8D3A70(v118);
          v26 = v98;
          a5 = v88;
          if ( v62 )
          {
            v63 = v118;
            if ( v118 == v119[0] )
            {
LABEL_135:
              v117 = v63;
              v61 = 1;
              goto LABEL_130;
            }
            v64 = sub_8D97D0(v118, v119[0], 0, a4, v88);
            v26 = v98;
            a5 = v88;
            if ( v64 )
            {
LABEL_134:
              v63 = v118;
              goto LABEL_135;
            }
            goto LABEL_129;
          }
          v66 = sub_8D3350(v118);
          v26 = v98;
          a5 = v88;
          if ( !v66 )
            goto LABEL_129;
          v67 = sub_8D3350(v119[0]);
          v26 = v98;
          a5 = v88;
          if ( !v67 )
            goto LABEL_129;
          v63 = v118;
          if ( v118 == v119[0] )
            goto LABEL_135;
          v68 = sub_8D97D0(v118, v119[0], 0, a4, v88);
          v63 = v118;
          v26 = v98;
          a5 = v88;
          if ( v68 )
            goto LABEL_135;
          if ( (*(_BYTE *)(v118 + 140) & 0xFB) == 8 )
          {
            v75 = sub_8D4C10(v118, dword_4F077C4 != 2);
            a4 = 0;
            v26 = v98;
            a5 = v88;
            v70 = v75;
            v69 = v119[0];
            if ( (*(_BYTE *)(v119[0] + 140LL) & 0xFB) != 8 )
              goto LABEL_154;
          }
          else
          {
            v69 = v119[0];
            v70 = 0;
            if ( (*(_BYTE *)(v119[0] + 140LL) & 0xFB) != 8 )
            {
LABEL_156:
              v90 = a5;
              v101 = v26;
              v72 = sub_6EEB90(v63, v69, v70);
              v26 = v101;
              a5 = v90;
              if ( v72 )
              {
                v73 = sub_8D2780(v118);
                v26 = v101;
                a5 = v90;
                if ( !v73 )
                {
                  v74 = sub_8D2780(v119[0]);
                  v26 = v101;
                  a5 = v90;
                  if ( v74 )
                  {
                    v63 = v119[0];
                    goto LABEL_135;
                  }
                }
                goto LABEL_134;
              }
LABEL_129:
              v61 = 0;
LABEL_130:
              v114 = v61;
              goto LABEL_88;
            }
          }
          v83 = a5;
          v89 = v70;
          v100 = v26;
          v71 = sub_8D4C10(v69, dword_4F077C4 != 2);
          a5 = v83;
          v70 = v89;
          v26 = v100;
          a4 = v71;
LABEL_154:
          if ( (_DWORD)a4 != (_DWORD)v70 )
            goto LABEL_129;
          v63 = v118;
          v69 = v119[0];
          goto LABEL_156;
        }
        sub_6EED10(v110, &v118, 1, 0, v86, &v117);
        v60 = v118;
        if ( (_DWORD)v118 )
        {
          v60 = 0;
        }
        else if ( *((_BYTE *)v110 + 24) == 8 )
        {
          v60 = 1;
          LODWORD(v118) = dword_4F077BC == 0;
        }
        v78 = v96;
        v79 = v60;
        sub_6EED10(v106, v119, 1, 0, v86, &v117);
        a5 = v86;
        v47 = v96;
        if ( LODWORD(v119[0]) )
        {
          v99 = 0;
        }
        else
        {
          if ( *(_BYTE *)(v106 + 24) != 8 || dword_4F077BC )
            goto LABEL_126;
          LODWORD(v119[0]) = 1;
          v99 = 1;
        }
        if ( (_DWORD)v118 )
        {
          v114 = 1;
          v13 = 1;
          if ( (v8 & 1) != 0 )
            goto LABEL_24;
          *((_BYTE *)v9 + 25) |= 1u;
          *((_BYTE *)v9 + 58) |= 1u;
          v110[2] = 0;
          if ( v79 )
          {
            if ( !v99 )
              goto LABEL_91;
          }
          else
          {
            v65 = sub_6EED10(v110, &v118, 0, 0, v86, 0);
            a5 = v86;
            v110 = (_QWORD *)v65;
            v47 = v78;
            if ( !v99 )
            {
LABEL_91:
              v94 = v47;
              v48 = sub_6EED10(v106, v119, 0, v6, a5, 0);
              v47 = v94;
              v106 = v48;
              if ( v6 )
              {
                v49 = v117;
                if ( *v110 != v117 )
                {
                  v50 = sub_8D97D0(*v110, v117, 0, a4, a5);
                  v47 = v94;
                  if ( v50 )
                  {
                    v49 = v117;
                  }
                  else
                  {
                    v76 = sub_691700((__int64)v110, v117, 1);
                    v49 = v117;
                    v47 = v94;
                    v110 = (_QWORD *)v76;
                  }
                }
                if ( *(_QWORD *)v106 != v49 )
                {
                  v95 = v47;
                  v51 = sub_8D97D0(*(_QWORD *)v106, v49, 0, a4, a5);
                  v47 = v95;
                  if ( !v51 )
                  {
                    v77 = sub_691700(v106, v117, 1);
                    v47 = v95;
                    v106 = v77;
                  }
                }
              }
            }
          }
          v13 = v114;
          v47[2] = v110;
          v110[2] = v106;
          goto LABEL_24;
        }
LABEL_126:
        v114 = 0;
        v13 = 0;
        goto LABEL_24;
      }
      if ( v6 )
      {
        v84 = a5;
        v92 = (_QWORD *)v9[9];
        v45 = sub_8D2600(v117);
        v26 = v92;
        a5 = v84;
        if ( !v45 )
          goto LABEL_127;
      }
LABEL_88:
      v13 = v114;
      if ( v114 && (v8 & 1) == 0 )
      {
        v81 = v26;
        *((_BYTE *)v9 + 25) |= 1u;
        *((_BYTE *)v9 + 58) |= 1u;
        v110[2] = 0;
        v93 = a5;
        v46 = sub_6EED10(v110, &v118, 0, v6, a5, 0);
        a5 = v93;
        v47 = v81;
        v110 = (_QWORD *)v46;
        goto LABEL_91;
      }
LABEL_24:
      if ( v8 )
        goto LABEL_17;
LABEL_25:
      if ( !v13 )
      {
LABEL_26:
        *a2 = 0;
        return (__int64)v9;
      }
LABEL_35:
      v22 = v117;
      v9[1] = 0;
      *((_BYTE *)v9 + 25) |= 1u;
      v23 = *v9;
      *v9 = v22;
      if ( !dword_4D03F94
        || (*(_BYTE *)(v9 - 1) & 8) == 0
        || v22 == v23
        || (unsigned int)sub_8D97D0(v23, v22, 0, a4, a5) )
      {
        goto LABEL_16;
      }
      result = sub_73DC30(8, v23, v9);
      v13 = v114;
      v15 = v114 != 0;
LABEL_18:
      *a2 = v13;
      if ( a6 )
      {
        if ( v15 )
          *a6 = v117;
      }
      return result;
    default:
      v111 = (_QWORD *)v9[9];
      v28 = sub_730030(v27);
      v26 = v111;
      if ( !v28 )
        goto LABEL_23;
LABEL_47:
      v114 = 1;
      if ( !v8 )
      {
        *((_BYTE *)v9 + 58) |= 1u;
        v117 = *v26;
        goto LABEL_35;
      }
      v117 = *v26;
LABEL_49:
      v13 = v8;
      goto LABEL_17;
  }
}
