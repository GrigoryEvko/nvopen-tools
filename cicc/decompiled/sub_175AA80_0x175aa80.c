// Function: sub_175AA80
// Address: 0x175aa80
//
_QWORD *__fastcall sub_175AA80(__int64 a1, __int64 a2, __int64 a3, __int64 a4, double a5, double a6, double a7)
{
  int v7; // r12d
  unsigned __int8 *v8; // r14
  __int64 v9; // r10
  __int64 v10; // r15
  __int64 v11; // r8
  __int64 v12; // rax
  _QWORD *v13; // r13
  __int64 v15; // rax
  __int64 v16; // rax
  __int64 v17; // rax
  unsigned int v18; // ebx
  int v19; // eax
  bool v20; // al
  __int64 v21; // rdx
  __int64 v22; // rcx
  unsigned int v23; // ebx
  unsigned int v24; // ebx
  int v25; // eax
  bool v26; // al
  __int64 v27; // rax
  __int64 *v28; // rdi
  __int64 v29; // r10
  __int64 v30; // r8
  unsigned int v31; // edx
  __int64 v32; // rax
  __int64 v33; // r8
  __int64 v34; // r13
  const char *v35; // rax
  __int64 ***v36; // r10
  __int64 v37; // rdx
  __int64 v38; // rax
  _QWORD *v39; // rsi
  __int64 v40; // rax
  __int64 v41; // rcx
  __int64 v42; // rdx
  __int64 v43; // rax
  __int64 v44; // r14
  _QWORD *v45; // rax
  __int64 *v46; // r13
  _QWORD *v47; // rsi
  char v48; // al
  __int64 v49; // rdx
  __int64 v50; // rcx
  unsigned int v51; // r8d
  __int64 v52; // r10
  bool v53; // al
  __int64 v54; // rax
  __int64 v55; // r14
  _QWORD *v56; // rax
  __int64 v57; // rdx
  __int64 v58; // rcx
  __int64 v59; // rax
  __int64 v60; // rax
  __int64 v61; // rax
  __int64 v62; // rdx
  __int64 v63; // rcx
  __int64 v64; // rdx
  __int64 v65; // rcx
  __int64 v66; // rdx
  __int64 v67; // rcx
  _QWORD *v68; // rax
  __int64 v69; // rax
  int v70; // eax
  __int64 v71; // rcx
  _QWORD *v72; // rax
  int v73; // eax
  _QWORD *v74; // rax
  __int64 v75; // rax
  __int64 v76; // r13
  __int64 v77; // rax
  __int64 v78; // r10
  _QWORD *v79; // rax
  _QWORD *v80; // rax
  __int64 v82; // [rsp+0h] [rbp-B0h]
  __int64 v86; // [rsp+8h] [rbp-A8h]
  __int64 v87; // [rsp+8h] [rbp-A8h]
  __int64 v88; // [rsp+8h] [rbp-A8h]
  __int64 v92; // [rsp+10h] [rbp-A0h]
  __int64 v93; // [rsp+10h] [rbp-A0h]
  __int64 v94; // [rsp+10h] [rbp-A0h]
  __int64 v96; // [rsp+10h] [rbp-A0h]
  unsigned int v97; // [rsp+10h] [rbp-A0h]
  __int64 *v98; // [rsp+18h] [rbp-98h]
  __int64 v99; // [rsp+18h] [rbp-98h]
  __int64 v100; // [rsp+18h] [rbp-98h]
  __int64 ***v101; // [rsp+18h] [rbp-98h]
  __int64 v102; // [rsp+18h] [rbp-98h]
  __int64 *v103; // [rsp+18h] [rbp-98h]
  __int64 ***v104; // [rsp+18h] [rbp-98h]
  __int64 *v105; // [rsp+28h] [rbp-88h] BYREF
  _QWORD v106[2]; // [rsp+30h] [rbp-80h] BYREF
  _QWORD *v107[2]; // [rsp+40h] [rbp-70h] BYREF
  __int16 v108; // [rsp+50h] [rbp-60h]
  _QWORD *v109[2]; // [rsp+60h] [rbp-50h] BYREF
  __int16 v110; // [rsp+70h] [rbp-40h]

  v7 = *(_WORD *)(a2 + 18) & 0x7FFF;
  if ( (unsigned int)(v7 - 32) > 1 )
    return 0;
  v8 = *(unsigned __int8 **)(a3 - 48);
  v9 = a3;
  v10 = *(_QWORD *)(a3 - 24);
  v98 = *(__int64 **)(a2 - 24);
  v11 = a1;
  switch ( *(_BYTE *)(a3 + 16) )
  {
    case '#':
      v109[0] = v106;
      if ( (unsigned __int8)sub_13D2630(v109, (_BYTE *)v10) )
      {
        v16 = *(_QWORD *)(a3 + 8);
        if ( !v16 || *(_QWORD *)(v16 + 8) )
          return 0;
        v17 = sub_15A2B60(v98, v10, 0, 0, a5, a6, a7);
LABEL_17:
        v10 = v17;
      }
      else
      {
        v96 = a3;
        if ( !sub_13D01C0(a4) )
          return 0;
        v86 = sub_1705480(a5, a6, a7, a1, v10, v64, v65);
        if ( v86 )
        {
          v110 = 257;
          v68 = sub_1648A60(56, 2u);
          v13 = v68;
          if ( v68 )
            sub_17582E0((__int64)v68, v7, (__int64)v8, v86, (__int64)v109);
          return v13;
        }
        v88 = sub_1705480(a5, a6, a7, a1, (__int64)v8, v66, v67);
        if ( v88 )
        {
          v110 = 257;
          v72 = sub_1648A60(56, 2u);
          v13 = v72;
          if ( v72 )
            sub_17582E0((__int64)v72, v7, v88, v10, (__int64)v109);
          return v13;
        }
        v75 = *(_QWORD *)(v96 + 8);
        if ( !v75 || *(_QWORD *)(v75 + 8) )
          return 0;
        v76 = *(_QWORD *)(a1 + 8);
        v108 = 257;
        if ( *(_BYTE *)(v10 + 16) > 0x10u )
        {
          v110 = 257;
          v79 = (_QWORD *)sub_15FB530((__int64 *)v10, (__int64)v109, 0, v71);
          v80 = sub_171D920(v76, v79, (__int64 *)v107);
          v78 = v96;
          v10 = (__int64)v80;
        }
        else
        {
          v10 = sub_15A2B90((__int64 *)v10, 0, 0, v71, a5, a6, a7);
          v77 = sub_14DBA30(v10, *(_QWORD *)(v76 + 96), 0);
          v78 = v96;
          if ( v77 )
            v10 = v77;
        }
        sub_164B7C0(v10, v78);
      }
      goto LABEL_54;
    case '%':
      v12 = *(_QWORD *)(a3 + 8);
      if ( !v12 )
        return 0;
      v13 = *(_QWORD **)(v12 + 8);
      if ( v13 )
        return 0;
      v109[0] = v107;
      if ( !(unsigned __int8)sub_13D2630(v109, v8) )
      {
        if ( sub_13D01C0(a4) )
          goto LABEL_54;
        return v13;
      }
      v69 = sub_15A2B60((__int64 *)v8, (__int64)v98, 0, 0, a5, a6, a7);
      v110 = 257;
      v8 = (unsigned __int8 *)v69;
      v13 = sub_1648A60(56, 2u);
      if ( !v13 )
        return v13;
      goto LABEL_28;
    case '\'':
      v18 = *(_DWORD *)(a4 + 8);
      if ( v18 <= 0x40 )
      {
        v20 = *(_QWORD *)a4 == 0;
      }
      else
      {
        v19 = sub_16A57B0(a4);
        v9 = a3;
        v20 = v18 == v19;
      }
      if ( !v20 )
        return 0;
      if ( !sub_15F2380(v9) )
        return 0;
      v109[0] = v107;
      if ( !(unsigned __int8)sub_13D2630(v109, (_BYTE *)v10) || sub_13D01C0((__int64)v107[0]) )
        return 0;
      v17 = sub_15A06D0((__int64 **)*v98, v10, v21, v22);
      goto LABEL_17;
    case ')':
      v23 = *(_DWORD *)(a4 + 8);
      if ( v23 <= 0x40 )
      {
        if ( *(_QWORD *)a4 )
          return 0;
      }
      else if ( v23 != (unsigned int)sub_16A57B0(a4) )
      {
        return 0;
      }
      v110 = 257;
      LOWORD(v7) = 3 * (v7 == 33) + 34;
      v13 = sub_1648A60(56, 2u);
      if ( v13 )
LABEL_28:
        sub_17582E0((__int64)v13, v7, v10, (__int64)v8, (__int64)v109);
      return v13;
    case '-':
      v24 = *(_DWORD *)(a4 + 8);
      if ( v24 <= 0x40 )
      {
        v26 = *(_QWORD *)a4 == 0;
      }
      else
      {
        v25 = sub_16A57B0(a4);
        v11 = a1;
        v9 = a3;
        v26 = v24 == v25;
      }
      if ( !v26 )
        return 0;
      v27 = *(_QWORD *)(v9 + 8);
      v92 = v11;
      v99 = v9;
      if ( !v27 )
        return 0;
      v13 = *(_QWORD **)(v27 + 8);
      if ( v13 )
        return 0;
      v109[0] = &v105;
      if ( !(unsigned __int8)sub_13D2630(v109, (_BYTE *)v10) )
        return v13;
      v28 = v105;
      v29 = v99;
      v30 = v92;
      v31 = *((_DWORD *)v105 + 2);
      if ( v31 > 0x40 )
      {
        v82 = v99;
        v87 = v92;
        v103 = v105;
        v97 = v31 + 1;
        if ( sub_13D0200(v105, v31 - 1) )
        {
          v70 = sub_16A5810((__int64)v103);
          v28 = v103;
          v30 = v87;
          v29 = v82;
          if ( v97 - v70 > 0x40 )
            return v13;
        }
        else
        {
          v73 = sub_1455840((__int64)v103);
          v28 = v103;
          v30 = v87;
          v29 = v82;
          if ( v97 - v73 > 0x40 )
            goto LABEL_38;
        }
        v32 = *(_QWORD *)*v28;
      }
      else
      {
        v32 = *v105 << (64 - (unsigned __int8)v31) >> (64 - (unsigned __int8)v31);
      }
      if ( v32 <= 1 )
        return v13;
LABEL_38:
      v93 = v29;
      v100 = v30;
      if ( !sub_14A9C60((__int64)v28) )
        return v13;
      v33 = v100;
      v101 = (__int64 ***)v93;
      v34 = *(_QWORD *)(v33 + 8);
      v35 = sub_1649960(v93);
      v36 = (__int64 ***)v93;
      v106[0] = v35;
      v108 = 261;
      v106[1] = v37;
      v107[0] = v106;
      if ( v8[16] > 0x10u || *(_BYTE *)(v10 + 16) > 0x10u )
        goto LABEL_83;
      v38 = sub_15A2A30((__int64 *)0x14, (__int64 *)v8, v10, 0, 0, a5, a6, a7);
      v39 = *(_QWORD **)(v34 + 96);
      v94 = v38;
      v40 = sub_14DBA30(v38, (__int64)v39, 0);
      v36 = v101;
      v42 = v40;
      if ( !v40 )
      {
        v41 = v94;
        if ( v94 )
        {
          v42 = v94;
        }
        else
        {
LABEL_83:
          v104 = v36;
          v110 = 257;
          v39 = (_QWORD *)sub_15FB440(20, (__int64 *)v8, v10, (__int64)v109, 0);
          v74 = sub_171D920(v34, v39, (__int64 *)v107);
          v36 = v104;
          v42 = (__int64)v74;
        }
      }
      v102 = v42;
      v43 = sub_15A06D0(*v36, (__int64)v39, v42, v41);
      v110 = 257;
      v44 = v43;
      v45 = sub_1648A60(56, 2u);
      v13 = v45;
      if ( v45 )
        sub_17582E0((__int64)v45, v7, v102, v44, (__int64)v109);
      return v13;
    case '2':
      v109[0] = v107;
      if ( !(unsigned __int8)sub_13D2630(v109, (_BYTE *)v10) )
        return 0;
      v46 = v107[0];
      v47 = v107[0];
      v48 = sub_1455820(a4, v107[0]);
      v52 = a3;
      if ( v48 )
      {
        v53 = sub_14A9C60(a4);
        v52 = a3;
        if ( v53 )
        {
          v54 = sub_15A06D0((__int64 **)*v98, (__int64)v47, v49, v50);
          v110 = 257;
          v55 = v54;
          v56 = sub_1648A60(56, 2u);
          v13 = v56;
          if ( v56 )
            sub_17582E0((__int64)v56, (v7 != 33) + 32, a3, v55, (__int64)v109);
          return v13;
        }
      }
      v61 = *(_QWORD *)(v52 + 8);
      if ( !v61 || *(_QWORD *)(v61 + 8) || !(unsigned __int8)sub_13CFF40(v46, (__int64)v47, v49, v50, v51) )
        return 0;
      v10 = sub_15A06D0(*(__int64 ***)v8, (__int64)v47, v62, v63);
      LOWORD(v7) = (v7 == 33) + 39;
LABEL_54:
      v110 = 257;
      v13 = sub_1648A60(56, 2u);
      if ( v13 )
        goto LABEL_55;
      return v13;
    case '3':
      v109[0] = v107;
      if ( !(unsigned __int8)sub_13D2630(v109, (_BYTE *)v10) )
        return 0;
      v59 = *(_QWORD *)(a3 + 8);
      if ( !v59 || *(_QWORD *)(v59 + 8) || !sub_1596070((__int64)v98, v10, v57, v58) )
        return 0;
      v60 = sub_15A2B00((__int64 *)v10, a5, a6, a7);
      v110 = 257;
      v10 = v60;
      v8 = sub_1729500(*(_QWORD *)(a1 + 8), v8, v60, (__int64 *)v109, a5, a6, a7);
      goto LABEL_54;
    case '4':
      v15 = *(_QWORD *)(a3 + 8);
      if ( !v15 || *(_QWORD *)(v15 + 8) )
        return 0;
      if ( *(_BYTE *)(v10 + 16) > 0x10u )
      {
        if ( !sub_13D01C0(a4) )
          return 0;
      }
      else
      {
        v10 = sub_15A2D30(v98, v10, a5, a6, a7);
      }
      v110 = 257;
      v13 = sub_1648A60(56, 2u);
      if ( v13 )
LABEL_55:
        sub_17582E0((__int64)v13, v7, (__int64)v8, v10, (__int64)v109);
      break;
    default:
      return 0;
  }
  return v13;
}
