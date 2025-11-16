// Function: sub_17321F0
// Address: 0x17321f0
//
unsigned __int8 *__fastcall sub_17321F0(
        __int64 a1,
        __int64 a2,
        unsigned __int64 a3,
        __int64 a4,
        __int64 a5,
        double a6,
        double a7,
        double a8)
{
  unsigned __int64 v8; // r15
  __int64 v9; // r14
  __int64 v12; // rdi
  bool v13; // al
  unsigned __int8 *v14; // r10
  int v15; // eax
  unsigned __int8 v16; // al
  __int64 v17; // rsi
  int v18; // ecx
  int v19; // ecx
  __int64 *v20; // rsi
  int v21; // eax
  __int64 **v22; // rax
  __int64 *v23; // r10
  __int64 v24; // rdi
  __int64 v25; // rdx
  __int64 v26; // rcx
  __int64 v27; // rax
  __int64 v28; // rcx
  int v29; // eax
  int v30; // eax
  unsigned __int8 **v31; // rcx
  __int64 **v32; // rdx
  __int64 v33; // rax
  unsigned __int8 *v34; // rax
  unsigned __int8 *v35; // r15
  _QWORD *v36; // r14
  __int64 v37; // rax
  _QWORD *v39; // rax
  __int64 v40; // rdx
  unsigned __int64 v41; // rax
  __int64 v42; // rdx
  __int64 v43; // rdx
  unsigned __int64 v44; // rax
  __int64 v45; // rax
  __int64 v46; // rdx
  unsigned __int64 v47; // rax
  __int64 v48; // rax
  __int64 v49; // rdi
  unsigned __int64 *v50; // r12
  __int64 v51; // rax
  unsigned __int64 v52; // rcx
  __int64 v53; // rdi
  __int64 *v54; // rsi
  __int64 v55; // rdx
  bool v56; // zf
  __int64 v57; // rsi
  __int64 v58; // rsi
  unsigned __int8 *v59; // rsi
  __int64 v60; // rcx
  int v61; // eax
  int v62; // eax
  __int64 *v63; // r15
  __int64 *v64; // rax
  __int64 **v65; // r15
  unsigned int v66; // eax
  __int64 v67; // rax
  __int64 v68; // rdi
  unsigned __int8 *v69; // r10
  __int64 *v70; // r15
  __int64 v71; // rax
  __int64 v72; // rcx
  _QWORD *v73; // rsi
  __int64 v74; // rsi
  __int64 v75; // rdx
  unsigned __int8 *v76; // rsi
  __int64 v77; // rax
  __int64 v78; // rax
  __int64 *v79; // rcx
  char v80; // al
  __int64 v81; // rsi
  int v82; // ecx
  int v83; // ecx
  _QWORD *v84; // rsi
  __int64 v85; // r15
  __int64 v86; // rdi
  __int64 *v87; // rdi
  __int64 *v88; // r15
  __int64 *v89; // rax
  __int64 **v90; // rsi
  __int64 v91; // rax
  __int64 v92; // rcx
  int v93; // eax
  int v94; // eax
  _QWORD *v95; // rcx
  __int64 v96; // rsi
  unsigned __int64 v97; // [rsp+8h] [rbp-C8h]
  __int64 v98; // [rsp+10h] [rbp-C0h]
  unsigned __int8 *v99; // [rsp+10h] [rbp-C0h]
  __int64 v100; // [rsp+10h] [rbp-C0h]
  _QWORD *v102; // [rsp+18h] [rbp-B8h]
  __int64 **v103; // [rsp+20h] [rbp-B0h]
  __int64 **v104; // [rsp+28h] [rbp-A8h]
  unsigned __int8 *v105; // [rsp+28h] [rbp-A8h]
  __int64 v106; // [rsp+28h] [rbp-A8h]
  __int64 v107; // [rsp+28h] [rbp-A8h]
  unsigned int v108; // [rsp+28h] [rbp-A8h]
  unsigned __int8 *v109; // [rsp+28h] [rbp-A8h]
  unsigned __int8 *v110; // [rsp+28h] [rbp-A8h]
  __int64 v111; // [rsp+30h] [rbp-A0h] BYREF
  _QWORD *v112; // [rsp+38h] [rbp-98h] BYREF
  __int64 v113[2]; // [rsp+40h] [rbp-90h] BYREF
  __int16 v114; // [rsp+50h] [rbp-80h]
  __int64 v115[2]; // [rsp+60h] [rbp-70h] BYREF
  __int16 v116; // [rsp+70h] [rbp-60h]
  _QWORD *v117[2]; // [rsp+80h] [rbp-50h] BYREF
  __int16 v118; // [rsp+90h] [rbp-40h]

  v8 = a3;
  v9 = a1;
  v103 = *(__int64 ***)a1;
  v104 = *(__int64 ***)a1;
  if ( *(_BYTE *)(a1 + 16) == 71 && (v77 = *(_QWORD *)(a1 + 8)) != 0 && !*(_QWORD *)(v77 + 8) )
  {
    v9 = *(_QWORD *)(a1 - 24);
    v104 = *(__int64 ***)v9;
    if ( *(_BYTE *)(a3 + 16) != 71 )
      goto LABEL_3;
  }
  else if ( *(_BYTE *)(a3 + 16) != 71 )
  {
    goto LABEL_3;
  }
  v78 = *(_QWORD *)(a3 + 8);
  if ( v78 && !*(_QWORD *)(v78 + 8) )
    v8 = *(_QWORD *)(a3 - 24);
LABEL_3:
  v117[0] = (_QWORD *)v8;
  if ( sub_13D1F50((__int64 *)v117, v9) )
  {
    v12 = (__int64)v104;
    if ( *((_BYTE *)v104 + 8) == 16 )
      v12 = *v104[2];
    v13 = sub_1642F90(v12, 1);
    v14 = (unsigned __int8 *)v9;
    if ( v13 )
      goto LABEL_45;
  }
  v15 = *(unsigned __int8 *)(v9 + 16);
  if ( (unsigned __int8)v15 > 0x17u )
  {
    v21 = v15 - 24;
  }
  else
  {
    if ( (_BYTE)v15 != 5 )
      goto LABEL_9;
    v21 = *(unsigned __int16 *)(v9 + 18);
  }
  if ( v21 == 38 )
  {
    v22 = (*(_BYTE *)(v9 + 23) & 0x40) != 0
        ? *(__int64 ***)(v9 - 8)
        : (__int64 **)(v9 - 24LL * (*(_DWORD *)(v9 + 20) & 0xFFFFFFF));
    v23 = *v22;
    if ( *v22 )
    {
      v24 = *v23;
      if ( *(_BYTE *)(*v23 + 8) == 16 )
        v24 = **(_QWORD **)(v24 + 16);
      v99 = (unsigned __int8 *)*v22;
      if ( sub_1642F90(v24, 1) )
      {
        v117[0] = v115;
        v27 = *(_QWORD *)(v8 + 8);
        if ( v27 )
        {
          if ( !*(_QWORD *)(v27 + 8) && sub_171DA10(v117, v8, v25, v26) )
          {
            v28 = v115[0];
            v14 = v99;
            v29 = *(unsigned __int8 *)(v115[0] + 16);
            if ( (_BYTE)v29 == 71 )
            {
              v96 = *(_QWORD *)(v115[0] + 8);
              if ( !v96 || *(_QWORD *)(v96 + 8) )
                goto LABEL_146;
              v28 = *(_QWORD *)(v115[0] - 24);
              v29 = *(unsigned __int8 *)(v28 + 16);
            }
            v115[0] = v28;
            if ( (unsigned __int8)v29 <= 0x17u )
            {
              if ( (_BYTE)v29 != 5 )
                goto LABEL_9;
              v30 = *(unsigned __int16 *)(v28 + 18);
              goto LABEL_41;
            }
LABEL_146:
            v30 = v29 - 24;
LABEL_41:
            if ( v30 == 38 )
            {
              v31 = (*(_BYTE *)(v28 + 23) & 0x40) != 0
                  ? *(unsigned __int8 ***)(v28 - 8)
                  : (unsigned __int8 **)(v28 - 24LL * (*(_DWORD *)(v28 + 20) & 0xFFFFFFF));
              if ( v99 == *v31 )
                goto LABEL_45;
            }
          }
        }
      }
    }
  }
LABEL_9:
  if ( *((_BYTE *)v104 + 8) != 16 )
    return 0;
  v16 = *(_BYTE *)(v9 + 16);
  if ( v16 <= 0x10u )
  {
    if ( *(_BYTE *)(v8 + 16) > 0x10u )
      goto LABEL_15;
    if ( (unsigned __int8)sub_1727D50(v9, v8) )
    {
      v118 = 257;
      if ( *((_BYTE *)v104 + 8) == 16 )
      {
        v63 = v104[4];
        v64 = (__int64 *)sub_1643320(*v104);
        v65 = (__int64 **)sub_16463B0(v64, (unsigned int)v63);
      }
      else
      {
        v65 = (__int64 **)sub_1643320(*v104);
      }
      v108 = sub_16431D0(*(_QWORD *)v9);
      v66 = sub_16431D0((__int64)v65);
      if ( v108 < v66 )
      {
        v14 = sub_1708970(a5, 37, v9, v65, (__int64 *)v117);
      }
      else
      {
        v14 = (unsigned __int8 *)v9;
        if ( v108 > v66 )
          v14 = sub_1708970(a5, 36, v9, v65, (__int64 *)v117);
      }
LABEL_91:
      if ( !v14 )
        return v14;
LABEL_45:
      v114 = 257;
      v32 = *(__int64 ***)v9;
      if ( *(_QWORD *)v9 != *(_QWORD *)a2 )
      {
        v105 = v14;
        if ( *(_BYTE *)(a2 + 16) > 0x10u )
        {
          v118 = 257;
          v67 = sub_15FDBD0(47, a2, (__int64)v32, (__int64)v117, 0);
          v68 = *(_QWORD *)(a5 + 8);
          v69 = v105;
          a2 = v67;
          if ( v68 )
          {
            v70 = *(__int64 **)(a5 + 16);
            sub_157E9D0(v68 + 40, v67);
            v71 = *(_QWORD *)(a2 + 24);
            v69 = v105;
            v72 = *v70;
            *(_QWORD *)(a2 + 32) = v70;
            v72 &= 0xFFFFFFFFFFFFFFF8LL;
            *(_QWORD *)(a2 + 24) = v72 | v71 & 7;
            *(_QWORD *)(v72 + 8) = a2 + 24;
            *v70 = *v70 & 7 | (a2 + 24);
          }
          v54 = v113;
          v53 = a2;
          v109 = v69;
          sub_164B780(a2, v113);
          v56 = *(_QWORD *)(a5 + 80) == 0;
          v111 = a2;
          if ( v56 )
            goto LABEL_157;
          (*(void (__fastcall **)(__int64, __int64 *))(a5 + 88))(a5 + 64, &v111);
          v73 = *(_QWORD **)a5;
          v14 = v109;
          if ( *(_QWORD *)a5 )
          {
            v115[0] = *(_QWORD *)a5;
            sub_1623A60((__int64)v115, (__int64)v73, 2);
            v74 = *(_QWORD *)(a2 + 48);
            v75 = a2 + 48;
            v14 = v109;
            if ( v74 )
            {
              sub_161E7C0(a2 + 48, v74);
              v14 = v109;
              v75 = a2 + 48;
            }
            v76 = (unsigned __int8 *)v115[0];
            *(_QWORD *)(a2 + 48) = v115[0];
            if ( v76 )
            {
              v110 = v14;
              sub_1623210((__int64)v115, v76, v75);
              v14 = v110;
            }
          }
        }
        else
        {
          a2 = sub_15A46C0(47, (__int64 ***)a2, v32, 0);
          v33 = sub_14DBA30(a2, *(_QWORD *)(a5 + 96), 0);
          v14 = v105;
          if ( v33 )
            a2 = v33;
        }
      }
      v118 = 257;
      v106 = (__int64)v14;
      v34 = sub_1708970(a5, 47, a4, *(__int64 ***)v9, (__int64 *)v117);
      v116 = 257;
      v35 = v34;
      if ( *(_BYTE *)(v106 + 16) <= 0x10u && *(_BYTE *)(a2 + 16) <= 0x10u && v34[16] <= 0x10u )
      {
        v36 = (_QWORD *)sub_15A2DC0(v106, (__int64 *)a2, (__int64)v34, 0);
        v37 = sub_14DBA30((__int64)v36, *(_QWORD *)(a5 + 96), 0);
        if ( v37 )
          v36 = (_QWORD *)v37;
LABEL_54:
        v118 = 257;
        return sub_1708970(a5, 47, (__int64)v36, v103, (__int64 *)v117);
      }
      v118 = 257;
      v39 = sub_1648A60(56, 3u);
      v36 = v39;
      if ( v39 )
      {
        v100 = v106;
        v102 = v39 - 9;
        v107 = (__int64)v39;
        sub_15F1EA0((__int64)v39, *(_QWORD *)a2, 55, (__int64)(v39 - 9), 3, 0);
        if ( *(v36 - 9) )
        {
          v40 = *(v36 - 8);
          v41 = *(v36 - 7) & 0xFFFFFFFFFFFFFFFCLL;
          *(_QWORD *)v41 = v40;
          if ( v40 )
            *(_QWORD *)(v40 + 16) = *(_QWORD *)(v40 + 16) & 3LL | v41;
        }
        *(v36 - 9) = v100;
        v42 = *(_QWORD *)(v100 + 8);
        *(v36 - 8) = v42;
        if ( v42 )
          *(_QWORD *)(v42 + 16) = (unsigned __int64)(v36 - 8) | *(_QWORD *)(v42 + 16) & 3LL;
        *(v36 - 7) = *(v36 - 7) & 3LL | (v100 + 8);
        *(_QWORD *)(v100 + 8) = v102;
        if ( *(v36 - 6) )
        {
          v43 = *(v36 - 5);
          v44 = *(v36 - 4) & 0xFFFFFFFFFFFFFFFCLL;
          *(_QWORD *)v44 = v43;
          if ( v43 )
            *(_QWORD *)(v43 + 16) = *(_QWORD *)(v43 + 16) & 3LL | v44;
        }
        *(v36 - 6) = a2;
        v45 = *(_QWORD *)(a2 + 8);
        *(v36 - 5) = v45;
        if ( v45 )
          *(_QWORD *)(v45 + 16) = (unsigned __int64)(v36 - 5) | *(_QWORD *)(v45 + 16) & 3LL;
        *(v36 - 4) = (a2 + 8) | *(v36 - 4) & 3LL;
        *(_QWORD *)(a2 + 8) = v36 - 6;
        if ( *(v36 - 3) )
        {
          v46 = *(v36 - 2);
          v47 = *(v36 - 1) & 0xFFFFFFFFFFFFFFFCLL;
          *(_QWORD *)v47 = v46;
          if ( v46 )
            *(_QWORD *)(v46 + 16) = *(_QWORD *)(v46 + 16) & 3LL | v47;
        }
        *(v36 - 3) = v35;
        if ( v35 )
        {
          v48 = *((_QWORD *)v35 + 1);
          *(v36 - 2) = v48;
          if ( v48 )
            *(_QWORD *)(v48 + 16) = (unsigned __int64)(v36 - 2) | *(_QWORD *)(v48 + 16) & 3LL;
          *(v36 - 1) = (unsigned __int64)(v35 + 8) | *(v36 - 1) & 3LL;
          *((_QWORD *)v35 + 1) = v36 - 3;
        }
        sub_164B780((__int64)v36, (__int64 *)v117);
      }
      else
      {
        v107 = 0;
      }
      v49 = *(_QWORD *)(a5 + 8);
      if ( v49 )
      {
        v50 = *(unsigned __int64 **)(a5 + 16);
        sub_157E9D0(v49 + 40, (__int64)v36);
        v51 = v36[3];
        v52 = *v50;
        v36[4] = v50;
        v52 &= 0xFFFFFFFFFFFFFFF8LL;
        v36[3] = v52 | v51 & 7;
        *(_QWORD *)(v52 + 8) = v36 + 3;
        *v50 = *v50 & 7 | (unsigned __int64)(v36 + 3);
      }
      v53 = v107;
      v54 = v115;
      sub_164B780(v107, v115);
      v56 = *(_QWORD *)(a5 + 80) == 0;
      v112 = v36;
      if ( !v56 )
      {
        (*(void (__fastcall **)(__int64, _QWORD **))(a5 + 88))(a5 + 64, &v112);
        v57 = *(_QWORD *)a5;
        if ( *(_QWORD *)a5 )
        {
          v117[0] = *(_QWORD **)a5;
          sub_1623A60((__int64)v117, v57, 2);
          v58 = v36[6];
          if ( v58 )
            sub_161E7C0((__int64)(v36 + 6), v58);
          v59 = (unsigned __int8 *)v117[0];
          v36[6] = v117[0];
          if ( v59 )
            sub_1623210((__int64)v117, v59, (__int64)(v36 + 6));
        }
        goto LABEL_54;
      }
LABEL_157:
      sub_4263D6(v53, v54, v55);
    }
    v16 = *(_BYTE *)(v9 + 16);
  }
  if ( v16 != 52 )
  {
LABEL_15:
    if ( v16 != 5 || *(_WORD *)(v9 + 18) != 28 )
      return 0;
    v17 = *(_QWORD *)(v9 - 24LL * (*(_DWORD *)(v9 + 20) & 0xFFFFFFF));
    v18 = *(unsigned __int8 *)(v17 + 16);
    if ( (unsigned __int8)v18 > 0x17u )
    {
      v19 = v18 - 24;
    }
    else
    {
      if ( (_BYTE)v18 != 5 )
        return 0;
      v19 = *(unsigned __int16 *)(v17 + 18);
    }
    if ( v19 != 38 )
      return 0;
    v20 = (*(_BYTE *)(v17 + 23) & 0x40) != 0
        ? *(__int64 **)(v17 - 8)
        : (__int64 *)(v17 - 24LL * (*(_DWORD *)(v17 + 20) & 0xFFFFFFF));
    v98 = *v20;
    if ( !*v20 )
      return 0;
    v97 = *(_QWORD *)(v9 + 24 * (1LL - (*(_DWORD *)(v9 + 20) & 0xFFFFFFF)));
    if ( !v97 )
      return 0;
    goto LABEL_115;
  }
  v60 = *(_QWORD *)(v9 - 48);
  v61 = *(unsigned __int8 *)(v60 + 16);
  if ( (unsigned __int8)v61 > 0x17u )
  {
    v62 = v61 - 24;
  }
  else
  {
    if ( (_BYTE)v61 != 5 )
      return 0;
    v62 = *(unsigned __int16 *)(v60 + 18);
  }
  if ( v62 != 38 )
    return 0;
  v79 = (*(_BYTE *)(v60 + 23) & 0x40) != 0
      ? *(__int64 **)(v60 - 8)
      : (__int64 *)(v60 - 24LL * (*(_DWORD *)(v60 + 20) & 0xFFFFFFF));
  v98 = *v79;
  if ( !*v79 )
    return 0;
  v97 = *(_QWORD *)(v9 - 24);
  if ( *(_BYTE *)(v97 + 16) > 0x10u )
    return 0;
LABEL_115:
  v80 = *(_BYTE *)(v8 + 16);
  if ( v80 == 52 )
  {
    v92 = *(_QWORD *)(v8 - 48);
    v93 = *(unsigned __int8 *)(v92 + 16);
    if ( (unsigned __int8)v93 > 0x17u )
    {
      v94 = v93 - 24;
    }
    else
    {
      if ( (_BYTE)v93 != 5 )
        return 0;
      v94 = *(unsigned __int16 *)(v92 + 18);
    }
    if ( v94 == 38 )
    {
      v95 = (*(_BYTE *)(v92 + 23) & 0x40) != 0
          ? *(_QWORD **)(v92 - 8)
          : (_QWORD *)(v92 - 24LL * (*(_DWORD *)(v92 + 20) & 0xFFFFFFF));
      if ( *v95 == v98 )
      {
        v85 = *(_QWORD *)(v8 - 24);
        if ( *(_BYTE *)(v85 + 16) <= 0x10u )
          goto LABEL_126;
      }
    }
  }
  else
  {
    if ( v80 != 5 || *(_WORD *)(v8 + 18) != 28 )
      return 0;
    v81 = *(_QWORD *)(v8 - 24LL * (*(_DWORD *)(v8 + 20) & 0xFFFFFFF));
    v82 = *(unsigned __int8 *)(v81 + 16);
    if ( (unsigned __int8)v82 > 0x17u )
    {
      v83 = v82 - 24;
    }
    else
    {
      if ( (_BYTE)v82 != 5 )
        return 0;
      v83 = *(unsigned __int16 *)(v81 + 18);
    }
    if ( v83 == 38 )
    {
      v84 = (*(_BYTE *)(v81 + 23) & 0x40) != 0
          ? *(_QWORD **)(v81 - 8)
          : (_QWORD *)(v81 - 24LL * (*(_DWORD *)(v81 + 20) & 0xFFFFFFF));
      if ( v98 == *v84 )
      {
        v85 = *(_QWORD *)(v8 + 24 * (1LL - (*(_DWORD *)(v8 + 20) & 0xFFFFFFF)));
        if ( v85 )
        {
LABEL_126:
          v86 = *(_QWORD *)v98;
          if ( *(_BYTE *)(*(_QWORD *)v98 + 8LL) == 16 )
            v86 = **(_QWORD **)(v86 + 16);
          if ( !sub_1642F90(v86, 1) || !(unsigned __int8)sub_1727D50(v97, v85) )
            return 0;
          v87 = *v104;
          if ( *((_BYTE *)v104 + 8) == 16 )
          {
            v88 = v104[4];
            v89 = (__int64 *)sub_1643320(v87);
            v90 = (__int64 **)sub_16463B0(v89, (unsigned int)v88);
          }
          else
          {
            v90 = (__int64 **)sub_1643320(v87);
          }
          v91 = sub_15A43B0(v97, v90, 0);
          v118 = 257;
          v14 = sub_172B670(a5, v98, v91, (__int64 *)v117, a6, a7, a8);
          goto LABEL_91;
        }
      }
    }
  }
  return 0;
}
