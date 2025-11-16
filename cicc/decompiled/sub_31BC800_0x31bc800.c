// Function: sub_31BC800
// Address: 0x31bc800
//
__int64 __fastcall sub_31BC800(__int64 a1, char *a2, __int64 a3)
{
  int v4; // ebx
  char *v5; // rax
  char *v6; // rdx
  char *v7; // rsi
  __int64 v8; // rdx
  __int64 v9; // rcx
  __int64 v10; // rdx
  char *v11; // rcx
  __int64 v13; // rdi
  __int64 *v14; // r13
  __int64 *v15; // r12
  __int64 *v16; // rdx
  __int64 v17; // rdx
  __int64 v18; // rax
  __int64 v19; // rdx
  __int64 v20; // r15
  __int64 *v21; // rdi
  __int64 v22; // r15
  bool v23; // al
  __int64 *v24; // rdi
  __int64 v25; // r15
  bool v26; // al
  __int64 *v27; // rdi
  __int64 v28; // r15
  bool v29; // al
  __int64 *v30; // rdi
  bool v31; // al
  unsigned __int8 *v32; // r13
  unsigned __int64 v33; // rdx
  __int64 v34; // r12
  int v35; // eax
  unsigned int v36; // ecx
  unsigned __int8 v37; // si
  _QWORD *v38; // r12
  _QWORD *v39; // rdx
  __int64 v40; // rax
  __int64 v41; // rcx
  __int64 v42; // r13
  bool v43; // al
  __int64 v44; // rdi
  __int64 *v45; // rdi
  bool v46; // al
  __int64 v47; // rdi
  __int64 *v48; // rdi
  bool v49; // al
  __int64 v50; // rdi
  __int64 *v51; // rdi
  int v52; // r12d
  char *v53; // r13
  char *v54; // rdx
  __int64 v55; // rax
  char *v56; // r15
  __int64 v57; // rax
  _QWORD *v58; // r15
  __int64 v59; // r13
  __int64 v60; // r13
  __int64 v61; // r13
  __int64 v62; // r13
  __int64 v63; // r12
  bool v64; // al
  int v65; // ebx
  __int64 v66; // rax
  __int64 v67; // rdx
  int v68; // eax
  __int64 v69; // r13
  bool v70; // al
  _QWORD *v71; // r13
  char *v72; // rbx
  char *v73; // rdx
  __int64 v74; // rax
  char *v75; // r14
  __int64 v76; // r15
  __int64 v77; // r15
  bool v78; // al
  __int64 v79; // r15
  bool v80; // al
  __int64 v81; // r15
  bool v82; // al
  bool v83; // al
  char *v84; // rcx
  __int64 v85; // rdx
  int v86; // eax
  char *v87; // r14
  __int64 v88; // r13
  __int64 v89; // rax
  __int64 *v90; // rdx
  __int64 *v91; // rbx
  __int64 v92; // r12
  __int64 v93; // r14
  __int64 *v94; // rax
  __int64 v95; // rdx
  unsigned __int8 *v96; // r15
  unsigned __int8 *v97; // r12
  _QWORD *v98; // rax
  __int64 v99; // r12
  __int64 v100; // rax
  __int64 v101; // rax
  __int64 v102; // r13
  __int64 v103; // rax
  __int64 *v104; // rdx
  __int64 *v105; // rbx
  __int64 v106; // r12
  __int64 v107; // r14
  __int64 *v108; // rax
  __int64 v109; // rdx
  unsigned __int8 *v110; // r15
  unsigned __int8 *v111; // r12
  _QWORD *v112; // rax
  __int64 v113; // r12
  __int64 v114; // rax
  __int64 v115; // r13
  __int64 v116; // r13
  __int64 v117; // r13
  signed __int64 v118; // rax
  signed __int64 v119; // rax
  __int64 v120; // r12
  bool v121; // al
  signed __int64 v122; // rdx
  __int64 v123; // r14
  bool v124; // al
  __int64 v125; // r14
  bool v126; // al
  _QWORD *v127; // [rsp+0h] [rbp-A0h]
  _QWORD *v128; // [rsp+8h] [rbp-98h]
  __int64 *v129; // [rsp+10h] [rbp-90h]
  bool v130; // [rsp+10h] [rbp-90h]
  char *v132; // [rsp+20h] [rbp-80h]
  __int64 *v133; // [rsp+20h] [rbp-80h]
  __int64 *v134; // [rsp+20h] [rbp-80h]
  __int64 *v136; // [rsp+28h] [rbp-78h]
  __int64 *v137; // [rsp+28h] [rbp-78h]
  __int64 v138; // [rsp+30h] [rbp-70h]
  unsigned __int8 *v139; // [rsp+30h] [rbp-70h]
  unsigned __int8 *v140; // [rsp+30h] [rbp-70h]
  __int64 *v141; // [rsp+38h] [rbp-68h]
  bool v142; // [rsp+38h] [rbp-68h]
  __int64 v143; // [rsp+38h] [rbp-68h]
  __int64 v144; // [rsp+38h] [rbp-68h]
  __int64 v145; // [rsp+38h] [rbp-68h]
  char *v146; // [rsp+38h] [rbp-68h]
  char *v147; // [rsp+38h] [rbp-68h]
  int v148; // [rsp+38h] [rbp-68h]
  int v149; // [rsp+38h] [rbp-68h]
  __int64 v150; // [rsp+48h] [rbp-58h]
  __int64 v151; // [rsp+50h] [rbp-50h] BYREF
  __int64 v152; // [rsp+58h] [rbp-48h]

  v132 = a2;
  v4 = *(_DWORD *)(*(_QWORD *)a2 + 32LL);
  v138 = *(_QWORD *)a2;
  v5 = (char *)sub_31BC590((__int64)a2, a3, 1);
  v7 = v6;
  v8 = v6 - v5;
  v9 = v8 >> 5;
  v10 = v8 >> 3;
  if ( v9 <= 0 )
  {
LABEL_12:
    if ( v10 != 2 )
    {
      if ( v10 != 3 )
      {
        if ( v10 != 1 )
          goto LABEL_15;
        goto LABEL_67;
      }
      if ( v4 != *(_DWORD *)(*(_QWORD *)v5 + 32LL) )
        goto LABEL_8;
      v5 += 8;
    }
    if ( v4 != *(_DWORD *)(*(_QWORD *)v5 + 32LL) )
      goto LABEL_8;
    v5 += 8;
LABEL_67:
    if ( v4 == *(_DWORD *)(*(_QWORD *)v5 + 32LL) )
      goto LABEL_15;
    goto LABEL_8;
  }
  v11 = &v5[32 * v9];
  while ( 1 )
  {
    if ( v4 != *(_DWORD *)(*(_QWORD *)v5 + 32LL) )
      goto LABEL_8;
    if ( v4 != *(_DWORD *)(*((_QWORD *)v5 + 1) + 32LL) )
    {
      v5 += 8;
      goto LABEL_8;
    }
    if ( v4 != *(_DWORD *)(*((_QWORD *)v5 + 2) + 32LL) )
    {
      v5 += 16;
      goto LABEL_8;
    }
    if ( v4 != *(_DWORD *)(*((_QWORD *)v5 + 3) + 32LL) )
      break;
    v5 += 32;
    if ( v11 == v5 )
    {
      v10 = (v7 - v5) >> 3;
      goto LABEL_12;
    }
  }
  v5 += 24;
LABEL_8:
  if ( v7 != v5 )
  {
LABEL_9:
    LODWORD(v151) = 1;
    BYTE4(v151) = 1;
    return v151;
  }
LABEL_15:
  if ( sub_318B630(v138) && (*(_DWORD *)(v138 + 8) != 37 || sub_318B6C0(v138)) )
  {
    if ( sub_318B670(v138) )
    {
      v13 = sub_318B680(v138);
    }
    else
    {
      v13 = v138;
      if ( *(_DWORD *)(v138 + 8) == 37 )
        v13 = sub_318B6C0(v138);
    }
  }
  else
  {
    v13 = v138;
  }
  v14 = sub_318EB80(v13);
  if ( (unsigned int)*(unsigned __int8 *)(*v14 + 8) - 17 <= 1 )
    v14 = sub_318E560(v14);
  v15 = (__int64 *)sub_31BC590((__int64)a2, a3, 1);
  v129 = v16;
  v17 = (char *)v16 - (char *)v15;
  v18 = v17 >> 5;
  v19 = v17 >> 3;
  if ( v18 <= 0 )
  {
LABEL_83:
    if ( v19 != 2 )
    {
      if ( v19 != 3 )
      {
        if ( v19 != 1 )
          goto LABEL_86;
        goto LABEL_136;
      }
      v143 = *v15;
      v43 = sub_318B630(*v15);
      v44 = v143;
      if ( v143 )
      {
        if ( v43 )
        {
          if ( *(_DWORD *)(v143 + 8) != 37 || (v44 = v143, sub_318B6C0(v143)) )
          {
            if ( sub_318B670(v44) )
            {
              v44 = sub_318B680(v44);
            }
            else if ( *(_DWORD *)(v44 + 8) == 37 )
            {
              v44 = sub_318B6C0(v44);
            }
          }
        }
      }
      v45 = sub_318EB80(v44);
      if ( (unsigned int)*(unsigned __int8 *)(*v45 + 8) - 17 <= 1 )
        v45 = sub_318E560(v45);
      if ( v14 != v45 )
        goto LABEL_76;
      ++v15;
    }
    v144 = *v15;
    v46 = sub_318B630(*v15);
    v47 = v144;
    if ( v144 )
    {
      if ( v46 )
      {
        if ( *(_DWORD *)(v144 + 8) != 37 || (v47 = v144, sub_318B6C0(v144)) )
        {
          if ( sub_318B670(v47) )
          {
            v47 = sub_318B680(v47);
          }
          else if ( *(_DWORD *)(v47 + 8) == 37 )
          {
            v47 = sub_318B6C0(v47);
          }
        }
      }
    }
    v48 = sub_318EB80(v47);
    if ( (unsigned int)*(unsigned __int8 *)(*v48 + 8) - 17 <= 1 )
      v48 = sub_318E560(v48);
    if ( v14 != v48 )
      goto LABEL_76;
    ++v15;
LABEL_136:
    v145 = *v15;
    v49 = sub_318B630(*v15);
    v50 = v145;
    if ( v145 )
    {
      if ( v49 )
      {
        if ( *(_DWORD *)(v145 + 8) != 37 || (v50 = v145, sub_318B6C0(v145)) )
        {
          if ( sub_318B670(v50) )
          {
            v50 = sub_318B680(v50);
          }
          else if ( *(_DWORD *)(v50 + 8) == 37 )
          {
            v50 = sub_318B6C0(v50);
          }
        }
      }
    }
    v51 = sub_318EB80(v50);
    if ( (unsigned int)*(unsigned __int8 *)(*v51 + 8) - 17 <= 1 )
      v51 = sub_318E560(v51);
    if ( v14 == v51 )
      goto LABEL_86;
    goto LABEL_76;
  }
  v141 = &v15[4 * v18];
  while ( 1 )
  {
    v20 = *v15;
    v31 = sub_318B630(*v15);
    if ( v20 && v31 && (*(_DWORD *)(v20 + 8) != 37 || sub_318B6C0(v20)) )
    {
      if ( sub_318B670(v20) )
      {
        v20 = sub_318B680(v20);
      }
      else if ( *(_DWORD *)(v20 + 8) == 37 )
      {
        v20 = sub_318B6C0(v20);
      }
    }
    v21 = sub_318EB80(v20);
    if ( (unsigned int)*(unsigned __int8 *)(*v21 + 8) - 17 <= 1 )
      v21 = sub_318E560(v21);
    if ( v14 != v21 )
      goto LABEL_76;
    v22 = v15[1];
    v23 = sub_318B630(v22);
    if ( v22 && v23 && (*(_DWORD *)(v22 + 8) != 37 || sub_318B6C0(v22)) )
    {
      if ( sub_318B670(v22) )
      {
        v22 = sub_318B680(v22);
      }
      else if ( *(_DWORD *)(v22 + 8) == 37 )
      {
        v22 = sub_318B6C0(v22);
      }
    }
    v24 = sub_318EB80(v22);
    if ( (unsigned int)*(unsigned __int8 *)(*v24 + 8) - 17 <= 1 )
      v24 = sub_318E560(v24);
    if ( v14 != v24 )
    {
      ++v15;
      goto LABEL_76;
    }
    v25 = v15[2];
    v26 = sub_318B630(v25);
    if ( v25 && v26 && (*(_DWORD *)(v25 + 8) != 37 || sub_318B6C0(v25)) )
    {
      if ( sub_318B670(v25) )
      {
        v25 = sub_318B680(v25);
      }
      else if ( *(_DWORD *)(v25 + 8) == 37 )
      {
        v25 = sub_318B6C0(v25);
      }
    }
    v27 = sub_318EB80(v25);
    if ( (unsigned int)*(unsigned __int8 *)(*v27 + 8) - 17 <= 1 )
      v27 = sub_318E560(v27);
    if ( v14 != v27 )
    {
      v15 += 2;
      goto LABEL_76;
    }
    v28 = v15[3];
    v29 = sub_318B630(v28);
    if ( v28 && v29 && (*(_DWORD *)(v28 + 8) != 37 || sub_318B6C0(v28)) )
    {
      if ( sub_318B670(v28) )
      {
        v28 = sub_318B680(v28);
      }
      else if ( *(_DWORD *)(v28 + 8) == 37 )
      {
        v28 = sub_318B6C0(v28);
      }
    }
    v30 = sub_318EB80(v28);
    if ( (unsigned int)*(unsigned __int8 *)(*v30 + 8) - 17 <= 1 )
      v30 = sub_318E560(v30);
    if ( v14 != v30 )
      break;
    v15 += 4;
    if ( v141 == v15 )
    {
      v19 = v129 - v15;
      goto LABEL_83;
    }
  }
  v15 += 3;
LABEL_76:
  if ( v129 != v15 )
  {
LABEL_77:
    LODWORD(v151) = 2;
    BYTE4(v151) = 1;
    return v151;
  }
LABEL_86:
  v32 = *(unsigned __int8 **)(v138 + 16);
  v33 = *v32;
  if ( (unsigned __int8)v33 <= 0x1Cu )
  {
LABEL_155:
    v57 = 0x40540000000000LL;
    if ( _bittest64(&v57, v33) )
      goto LABEL_107;
  }
  else
  {
    switch ( (char)v33 )
    {
      case ')':
      case '+':
      case '-':
      case '/':
      case '2':
      case '5':
      case 'J':
      case 'K':
      case 'S':
        goto LABEL_146;
      case 'T':
      case 'U':
      case 'V':
        v34 = *((_QWORD *)v32 + 1);
        v35 = *(unsigned __int8 *)(v34 + 8);
        v36 = v35 - 17;
        v37 = *(_BYTE *)(v34 + 8);
        if ( (unsigned int)(v35 - 17) <= 1 )
          v37 = *(_BYTE *)(**(_QWORD **)(v34 + 16) + 8LL);
        if ( v37 <= 3u || v37 == 5 || (v37 & 0xFD) == 4 )
          goto LABEL_146;
        if ( (_BYTE)v35 != 15 )
        {
          if ( (_BYTE)v35 == 16 )
          {
            do
            {
              v34 = *(_QWORD *)(v34 + 24);
              LOBYTE(v35) = *(_BYTE *)(v34 + 8);
            }
            while ( (_BYTE)v35 == 16 );
            v36 = (unsigned __int8)v35 - 17;
          }
LABEL_97:
          if ( v36 <= 1 )
            LOBYTE(v35) = *(_BYTE *)(**(_QWORD **)(v34 + 16) + 8LL);
          if ( (unsigned __int8)v35 > 3u && (_BYTE)v35 != 5 && (v35 & 0xFD) != 4 )
            goto LABEL_102;
LABEL_146:
          v52 = sub_B45210(*(_QWORD *)(*(_QWORD *)a2 + 16LL));
          v53 = (char *)sub_31BC590((__int64)a2, a3, 1);
          v146 = v54;
          v55 = (v54 - v53) >> 5;
          if ( v55 > 0 )
          {
            v56 = &v53[32 * v55];
            while ( 1 )
            {
              if ( v52 != (unsigned int)sub_B45210(*(_QWORD *)(*(_QWORD *)v53 + 16LL)) )
                goto LABEL_153;
              if ( v52 != (unsigned int)sub_B45210(*(_QWORD *)(*((_QWORD *)v53 + 1) + 16LL)) )
              {
                v53 += 8;
                goto LABEL_153;
              }
              if ( v52 != (unsigned int)sub_B45210(*(_QWORD *)(*((_QWORD *)v53 + 2) + 16LL)) )
              {
                v53 += 16;
                goto LABEL_153;
              }
              if ( v52 != (unsigned int)sub_B45210(*(_QWORD *)(*((_QWORD *)v53 + 3) + 16LL)) )
                break;
              v53 += 32;
              if ( v56 == v53 )
                goto LABEL_278;
            }
            v53 += 24;
LABEL_153:
            if ( v146 != v53 )
            {
              LODWORD(v151) = 3;
              BYTE4(v151) = 1;
              return v151;
            }
            goto LABEL_102;
          }
LABEL_278:
          v118 = v146 - v53;
          if ( v146 - v53 != 16 )
          {
            if ( v118 != 24 )
            {
              if ( v118 != 8 )
                goto LABEL_102;
              goto LABEL_281;
            }
            if ( v52 != (unsigned int)sub_B45210(*(_QWORD *)(*(_QWORD *)v53 + 16LL)) )
              goto LABEL_153;
            v53 += 8;
          }
          if ( v52 != (unsigned int)sub_B45210(*(_QWORD *)(*(_QWORD *)v53 + 16LL)) )
            goto LABEL_153;
          v53 += 8;
LABEL_281:
          if ( v52 == (unsigned int)sub_B45210(*(_QWORD *)(*(_QWORD *)v53 + 16LL)) )
            goto LABEL_102;
          goto LABEL_153;
        }
        if ( (*(_BYTE *)(v34 + 9) & 4) == 0 )
          goto LABEL_103;
        if ( sub_BCB420(*((_QWORD *)v32 + 1)) )
        {
          v34 = **(_QWORD **)(v34 + 16);
          v35 = *(unsigned __int8 *)(v34 + 8);
          v36 = v35 - 17;
          goto LABEL_97;
        }
LABEL_102:
        v32 = *(unsigned __int8 **)(v138 + 16);
        v33 = *v32;
LABEL_103:
        if ( (unsigned __int8)v33 <= 0x36u )
          goto LABEL_155;
        break;
      default:
        goto LABEL_103;
    }
  }
  if ( !sub_318B630(v138) || *(_DWORD *)(v138 + 32) != 57 )
    goto LABEL_170;
  v32 = *(unsigned __int8 **)(v138 + 16);
LABEL_107:
  v142 = sub_B448F0((__int64)v32);
  v130 = sub_B44900((__int64)v32);
  v38 = (_QWORD *)sub_31BC590((__int64)a2, a3, 1);
  v127 = v39;
  v40 = ((char *)v39 - (char *)v38) >> 5;
  v41 = v39 - v38;
  if ( v40 > 0 )
  {
    v128 = &v38[4 * v40];
    while ( 1 )
    {
      v42 = *(_QWORD *)(*v38 + 16LL);
      if ( v142 != sub_B448F0(v42) || v130 != sub_B44900(v42) )
        break;
      v58 = v38 + 1;
      v59 = *(_QWORD *)(v38[1] + 16LL);
      if ( v142 != sub_B448F0(v59)
        || v130 != sub_B44900(v59)
        || (v58 = v38 + 2, v60 = *(_QWORD *)(v38[2] + 16LL), v142 != sub_B448F0(v60))
        || v130 != sub_B44900(v60)
        || (v58 = v38 + 3, v61 = *(_QWORD *)(v38[3] + 16LL), v142 != sub_B448F0(v61))
        || v130 != sub_B44900(v61) )
      {
        v38 = v58;
        break;
      }
      v38 += 4;
      if ( v128 == v38 )
      {
        v41 = v127 - v38;
        goto LABEL_167;
      }
    }
LABEL_110:
    if ( v127 != v38 )
    {
      LODWORD(v151) = 4;
      BYTE4(v151) = 1;
      return v151;
    }
    goto LABEL_170;
  }
LABEL_167:
  if ( v41 != 2 )
  {
    if ( v41 != 3 )
    {
      if ( v41 != 1 )
        goto LABEL_170;
      goto LABEL_275;
    }
    v115 = *(_QWORD *)(*v38 + 16LL);
    if ( v142 != sub_B448F0(v115) || v130 != sub_B44900(v115) )
      goto LABEL_110;
    ++v38;
  }
  v116 = *(_QWORD *)(*v38 + 16LL);
  if ( v142 != sub_B448F0(v116) || v130 != sub_B44900(v116) )
    goto LABEL_110;
  ++v38;
LABEL_275:
  v117 = *(_QWORD *)(*v38 + 16LL);
  if ( v142 != sub_B448F0(v117) || v130 != sub_B44900(v117) )
    goto LABEL_110;
LABEL_170:
  switch ( v4 )
  {
    case 0:
    case 1:
    case 2:
    case 6:
    case 7:
    case 8:
    case 10:
    case 13:
    case 14:
    case 22:
    case 25:
    case 60:
    case 61:
      goto LABEL_188;
    case 3:
    case 4:
    case 5:
    case 15:
    case 16:
    case 17:
    case 18:
    case 19:
    case 20:
    case 21:
    case 23:
    case 24:
    case 45:
    case 46:
    case 47:
    case 62:
      LODWORD(v151) = 10;
      BYTE4(v151) = 1;
      return v151;
    case 9:
      v62 = *(_QWORD *)a2;
      (*(void (__fastcall **)(__int64 *, _QWORD, _QWORD, __int64))(**(_QWORD **)a2 + 16LL))(&v151, *(_QWORD *)a2, 0, 1);
      v63 = sub_318E5D0((__int64)&v151);
      v64 = sub_318B630(v63);
      if ( v63 && v64 && (*(_DWORD *)(v63 + 8) != 37 || sub_318B6C0(v63)) )
      {
        if ( sub_318B670(v63) )
        {
          v63 = sub_318B680(v63);
        }
        else if ( *(_DWORD *)(v63 + 8) == 37 )
        {
          v63 = sub_318B6C0(v63);
        }
      }
      v65 = 1;
      v66 = *sub_318EB80(v63);
      if ( *(_BYTE *)(v66 + 8) == 17 )
        v65 = *(_DWORD *)(v66 + 32);
      if ( sub_318B630(v62) && (*(_DWORD *)(v62 + 8) != 37 || sub_318B6C0(v62)) )
      {
        if ( sub_318B670(v62) )
        {
          v62 = sub_318B680(v62);
        }
        else if ( *(_DWORD *)(v62 + 8) == 37 )
        {
          v62 = sub_318B6C0(v62);
        }
      }
      v67 = *sub_318EB80(v62);
      v68 = 1;
      if ( *(_BYTE *)(v67 + 8) == 17 )
        v68 = *(_DWORD *)(v67 + 32);
      if ( v68 == v65 )
        goto LABEL_236;
LABEL_188:
      LODWORD(v151) = 9;
      BYTE4(v151) = 1;
      return v151;
    case 11:
      v88 = *(_QWORD *)a2;
      v133 = *(__int64 **)(a1 + 336);
      v89 = sub_31BC590((__int64)a2, a3, 1);
      v136 = v90;
      v91 = (__int64 *)v89;
      if ( (__int64 *)v89 == v90 )
        goto LABEL_236;
      while ( 2 )
      {
        v93 = v88;
        v88 = *v91;
        v139 = *(unsigned __int8 **)(sub_318B650(v93) + 16);
        v96 = *(unsigned __int8 **)(sub_318B650(v88) + 16);
        v97 = sub_98ACB0(v139, 6u);
        v148 = 0;
        if ( v97 != sub_98ACB0(v96, 6u) )
          goto LABEL_239;
        v98 = (_QWORD *)sub_B2BE50(*v133);
        v99 = sub_BCB2B0(v98);
        v100 = sub_B43CA0(*(_QWORD *)(v93 + 16));
        v101 = sub_D35010(v99, (__int64)v139, v99, (__int64)v96, v100 + 312, (__int64)v133, 0, 0);
        if ( BYTE4(v101) )
        {
          v148 = v101;
LABEL_239:
          v92 = sub_B43CA0(*(_QWORD *)(v93 + 16)) + 312;
          if ( sub_318B630(v93) && (*(_DWORD *)(v93 + 8) != 37 || sub_318B6C0(v93)) )
          {
            if ( sub_318B670(v93) )
            {
              v93 = sub_318B680(v93);
            }
            else if ( *(_DWORD *)(v93 + 8) == 37 )
            {
              v93 = sub_318B6C0(v93);
            }
          }
          v94 = sub_318EB80(v93);
          v151 = sub_9208B0(v92, *v94);
          v152 = v95;
          if ( v148 == (unsigned int)sub_CA1930(&v151) >> 3 )
          {
            if ( v136 == ++v91 )
              goto LABEL_236;
            continue;
          }
        }
        goto LABEL_248;
      }
    case 12:
      v102 = *(_QWORD *)a2;
      v134 = *(__int64 **)(a1 + 336);
      v103 = sub_31BC590((__int64)a2, a3, 1);
      v137 = v104;
      v105 = (__int64 *)v103;
      if ( (__int64 *)v103 == v104 )
        goto LABEL_236;
      while ( 2 )
      {
        v107 = v102;
        v102 = *v105;
        v140 = *(unsigned __int8 **)(sub_318B6A0(v107) + 16);
        v110 = *(unsigned __int8 **)(sub_318B6A0(v102) + 16);
        v111 = sub_98ACB0(v140, 6u);
        v149 = 0;
        if ( v111 != sub_98ACB0(v110, 6u) )
          goto LABEL_251;
        v112 = (_QWORD *)sub_B2BE50(*v134);
        v113 = sub_BCB2B0(v112);
        v114 = sub_B43CA0(*(_QWORD *)(v107 + 16));
        v150 = sub_D35010(v113, (__int64)v140, v113, (__int64)v110, v114 + 312, (__int64)v134, 0, 0);
        if ( BYTE4(v150) )
        {
          v149 = v150;
LABEL_251:
          v106 = sub_B43CA0(*(_QWORD *)(v107 + 16)) + 312;
          if ( sub_318B630(v107) && (*(_DWORD *)(v107 + 8) != 37 || sub_318B6C0(v107)) )
          {
            if ( sub_318B670(v107) )
            {
              v107 = sub_318B680(v107);
            }
            else if ( *(_DWORD *)(v107 + 8) == 37 )
            {
              v107 = sub_318B6C0(v107);
            }
          }
          v108 = sub_318EB80(v107);
          v151 = sub_9208B0(v106, *v108);
          v152 = v109;
          if ( (unsigned int)sub_CA1930(&v151) >> 3 == v149 )
          {
            if ( v137 == ++v105 )
              goto LABEL_236;
            continue;
          }
        }
        break;
      }
LABEL_248:
      LODWORD(v151) = 7;
      BYTE4(v151) = 1;
      return v151;
    case 48:
    case 49:
    case 50:
    case 51:
    case 52:
    case 53:
    case 54:
    case 55:
    case 56:
    case 57:
    case 58:
    case 59:
      (*(void (__fastcall **)(__int64 *, __int64, _QWORD, __int64))(*(_QWORD *)v138 + 16LL))(&v151, v138, 0, 1);
      v69 = sub_318E5D0((__int64)&v151);
      v70 = sub_318B630(v69);
      if ( v69 && v70 && (*(_DWORD *)(v69 + 8) != 37 || sub_318B6C0(v69)) )
      {
        if ( sub_318B670(v69) )
        {
          v69 = sub_318B680(v69);
        }
        else if ( *(_DWORD *)(v69 + 8) == 37 )
        {
          v69 = sub_318B6C0(v69);
        }
      }
      v71 = sub_318EB80(v69);
      v72 = (char *)sub_31BC590((__int64)a2, a3, 1);
      v147 = v73;
      v74 = (v73 - v72) >> 5;
      if ( v74 <= 0 )
        goto LABEL_310;
      v75 = &v72[32 * v74];
      while ( 1 )
      {
        (*(void (__fastcall **)(__int64 *, _QWORD, _QWORD, __int64))(**(_QWORD **)v72 + 16LL))(
          &v151,
          *(_QWORD *)v72,
          0,
          1);
        v76 = sub_318E5D0((__int64)&v151);
        v83 = sub_318B630(v76);
        if ( v76 && v83 && (*(_DWORD *)(v76 + 8) != 37 || sub_318B6C0(v76)) )
        {
          if ( sub_318B670(v76) )
          {
            v76 = sub_318B680(v76);
          }
          else if ( *(_DWORD *)(v76 + 8) == 37 )
          {
            v76 = sub_318B6C0(v76);
          }
        }
        if ( v71 != sub_318EB80(v76) )
          break;
        (*(void (__fastcall **)(__int64 *, _QWORD, _QWORD, __int64))(**((_QWORD **)v72 + 1) + 16LL))(
          &v151,
          *((_QWORD *)v72 + 1),
          0,
          1);
        v77 = sub_318E5D0((__int64)&v151);
        v78 = sub_318B630(v77);
        if ( v77 && v78 && (*(_DWORD *)(v77 + 8) != 37 || sub_318B6C0(v77)) )
        {
          if ( sub_318B670(v77) )
          {
            v77 = sub_318B680(v77);
          }
          else if ( *(_DWORD *)(v77 + 8) == 37 )
          {
            v77 = sub_318B6C0(v77);
          }
        }
        if ( v71 != sub_318EB80(v77) )
        {
          v72 += 8;
          break;
        }
        (*(void (__fastcall **)(__int64 *, _QWORD, _QWORD, __int64))(**((_QWORD **)v72 + 2) + 16LL))(
          &v151,
          *((_QWORD *)v72 + 2),
          0,
          1);
        v79 = sub_318E5D0((__int64)&v151);
        v80 = sub_318B630(v79);
        if ( v79 && v80 && (*(_DWORD *)(v79 + 8) != 37 || sub_318B6C0(v79)) )
        {
          if ( sub_318B670(v79) )
          {
            v79 = sub_318B680(v79);
          }
          else if ( *(_DWORD *)(v79 + 8) == 37 )
          {
            v79 = sub_318B6C0(v79);
          }
        }
        if ( v71 != sub_318EB80(v79) )
        {
          v72 += 16;
          break;
        }
        (*(void (__fastcall **)(__int64 *, _QWORD, _QWORD, __int64))(**((_QWORD **)v72 + 3) + 16LL))(
          &v151,
          *((_QWORD *)v72 + 3),
          0,
          1);
        v81 = sub_318E5D0((__int64)&v151);
        v82 = sub_318B630(v81);
        if ( v81 && v82 && (*(_DWORD *)(v81 + 8) != 37 || sub_318B6C0(v81)) )
        {
          if ( sub_318B670(v81) )
          {
            v81 = sub_318B680(v81);
          }
          else if ( *(_DWORD *)(v81 + 8) == 37 )
          {
            v81 = sub_318B6C0(v81);
          }
        }
        if ( v71 != sub_318EB80(v81) )
        {
          v72 += 24;
          break;
        }
        v72 += 32;
        if ( v72 == v75 )
        {
LABEL_310:
          v119 = v147 - v72;
          if ( v147 - v72 != 16 )
          {
            if ( v119 != 24 )
            {
              if ( v119 != 8 )
                goto LABEL_236;
LABEL_313:
              (*(void (__fastcall **)(__int64 *, _QWORD, _QWORD, __int64))(**(_QWORD **)v72 + 16LL))(
                &v151,
                *(_QWORD *)v72,
                0,
                1);
              v120 = sub_318E5D0((__int64)&v151);
              v121 = sub_318B630(v120);
              if ( v120 && v121 && (*(_DWORD *)(v120 + 8) != 37 || sub_318B6C0(v120)) )
              {
                if ( sub_318B670(v120) )
                {
                  v120 = sub_318B680(v120);
                }
                else if ( *(_DWORD *)(v120 + 8) == 37 )
                {
                  v120 = sub_318B6C0(v120);
                }
              }
              if ( v71 == sub_318EB80(v120) )
                goto LABEL_236;
              break;
            }
            (*(void (__fastcall **)(__int64 *, _QWORD, _QWORD, __int64))(**(_QWORD **)v72 + 16LL))(
              &v151,
              *(_QWORD *)v72,
              0,
              1);
            v123 = sub_318E5D0((__int64)&v151);
            v124 = sub_318B630(v123);
            if ( v123 && v124 && (*(_DWORD *)(v123 + 8) != 37 || sub_318B6C0(v123)) )
            {
              if ( sub_318B670(v123) )
              {
                v123 = sub_318B680(v123);
              }
              else if ( *(_DWORD *)(v123 + 8) == 37 )
              {
                v123 = sub_318B6C0(v123);
              }
            }
            if ( v71 != sub_318EB80(v123) )
              break;
            v72 += 8;
          }
          (*(void (__fastcall **)(__int64 *, _QWORD, _QWORD, __int64))(**(_QWORD **)v72 + 16LL))(
            &v151,
            *(_QWORD *)v72,
            0,
            1);
          v125 = sub_318E5D0((__int64)&v151);
          v126 = sub_318B630(v125);
          if ( v125 && v126 && (*(_DWORD *)(v125 + 8) != 37 || sub_318B6C0(v125)) )
          {
            if ( sub_318B670(v125) )
            {
              v125 = sub_318B680(v125);
            }
            else if ( *(_DWORD *)(v125 + 8) == 37 )
            {
              v125 = sub_318B6C0(v125);
            }
          }
          if ( v71 != sub_318EB80(v125) )
            break;
          v72 += 8;
          goto LABEL_313;
        }
      }
      if ( v147 != v72 )
        goto LABEL_77;
      goto LABEL_236;
    case 63:
    case 64:
      v84 = &a2[8 * a3];
      v85 = (8 * a3) >> 5;
      v86 = *(_WORD *)(*(_QWORD *)(v138 + 16) + 2LL) & 0x3F;
      if ( v85 <= 0 )
        goto LABEL_323;
      v87 = &a2[32 * v85];
      break;
    default:
      goto LABEL_236;
  }
  while ( v86 == (*(_WORD *)(*(_QWORD *)(*(_QWORD *)v132 + 16LL) + 2LL) & 0x3F) )
  {
    if ( v86 != (*(_WORD *)(*(_QWORD *)(*((_QWORD *)v132 + 1) + 16LL) + 2LL) & 0x3F) )
    {
      v132 += 8;
      break;
    }
    if ( v86 != (*(_WORD *)(*(_QWORD *)(*((_QWORD *)v132 + 2) + 16LL) + 2LL) & 0x3F) )
    {
      v132 += 16;
      break;
    }
    if ( v86 != (*(_WORD *)(*(_QWORD *)(*((_QWORD *)v132 + 3) + 16LL) + 2LL) & 0x3F) )
    {
      v132 += 24;
      break;
    }
    v132 += 32;
    if ( v87 == v132 )
    {
LABEL_323:
      v122 = v84 - v132;
      if ( v84 - v132 != 16 )
      {
        if ( v122 != 24 )
        {
          if ( v122 != 8 )
            goto LABEL_236;
LABEL_326:
          if ( v86 == (*(_WORD *)(*(_QWORD *)(*(_QWORD *)v132 + 16LL) + 2LL) & 0x3F) )
            goto LABEL_236;
          break;
        }
        if ( v86 != (*(_WORD *)(*(_QWORD *)(*(_QWORD *)v132 + 16LL) + 2LL) & 0x3F) )
          break;
        v132 += 8;
      }
      if ( v86 != (*(_WORD *)(*(_QWORD *)(*(_QWORD *)v132 + 16LL) + 2LL) & 0x3F) )
        break;
      v132 += 8;
      goto LABEL_326;
    }
  }
  if ( v84 != v132 )
    goto LABEL_9;
LABEL_236:
  BYTE4(v151) = 0;
  return v151;
}
