// Function: sub_1722490
// Address: 0x1722490
//
__int64 __fastcall sub_1722490(
        __int64 *a1,
        _BYTE *a2,
        __m128 a3,
        double a4,
        double a5,
        double a6,
        double a7,
        double a8,
        double a9,
        __m128 a10)
{
  __int64 *v10; // r15
  unsigned __int64 v11; // r12
  __int64 v14; // rcx
  __int64 v15; // r9
  __int64 v17; // rdx
  int v18; // eax
  int v19; // eax
  __int64 *v20; // r12
  __int64 v21; // rax
  __int64 *v22; // rax
  unsigned int v23; // eax
  __int64 v24; // rsi
  unsigned int v25; // ecx
  __int64 v26; // r14
  int v27; // eax
  char v28; // al
  __int64 v29; // rcx
  __int64 *v30; // r8
  unsigned int v31; // r9d
  __int64 v32; // rsi
  __int64 v33; // rdx
  int v34; // eax
  __int64 v35; // rax
  int v36; // eax
  char v37; // al
  __int64 v38; // rdx
  __int64 v39; // rax
  char v40; // cl
  __int64 v41; // rsi
  __int64 v42; // rcx
  __int64 *v43; // rdi
  char v44; // al
  __int64 *v45; // r12
  int v46; // eax
  bool v47; // al
  __int64 v48; // rdi
  __int64 *v49; // r12
  __int64 v50; // rax
  __int64 v51; // r13
  __int64 **v52; // rax
  __int64 *v53; // rax
  __int64 v54; // rax
  _QWORD *v55; // rax
  __int64 v56; // rdi
  __int64 *v57; // rax
  int v58; // eax
  __int64 *v59; // rax
  __int64 v60; // rdx
  __int64 v61; // rcx
  __int64 *v62; // rsi
  unsigned int v63; // eax
  __int64 v64; // rdi
  unsigned int v65; // ecx
  unsigned int v66; // eax
  __int64 v67; // rax
  __int64 *v68; // r8
  int v69; // eax
  int v70; // eax
  __int64 v71; // r12
  __int64 v72; // rdi
  unsigned __int8 *v73; // rax
  __int64 v74; // r12
  _QWORD *v75; // rax
  int v76; // eax
  __int64 *v77; // rsi
  int v78; // eax
  __int64 v79; // rdi
  char v80; // al
  char v81; // al
  char v82; // al
  __int64 v83; // rax
  __int64 *v84; // [rsp+8h] [rbp-B8h]
  __int64 v85; // [rsp+8h] [rbp-B8h]
  int v86; // [rsp+10h] [rbp-B0h]
  const void **v87; // [rsp+10h] [rbp-B0h]
  __int64 v88; // [rsp+18h] [rbp-A8h]
  __int64 v89; // [rsp+18h] [rbp-A8h]
  _QWORD *v90; // [rsp+18h] [rbp-A8h]
  __int64 v91; // [rsp+18h] [rbp-A8h]
  __int64 v92; // [rsp+18h] [rbp-A8h]
  __int64 v93; // [rsp+18h] [rbp-A8h]
  __int64 *v94; // [rsp+28h] [rbp-98h] BYREF
  __int64 *v95; // [rsp+30h] [rbp-90h] BYREF
  __int64 *v96; // [rsp+38h] [rbp-88h] BYREF
  _QWORD *v97[2]; // [rsp+40h] [rbp-80h] BYREF
  unsigned __int64 v98; // [rsp+50h] [rbp-70h] BYREF
  __int64 **v99; // [rsp+58h] [rbp-68h]
  __int16 v100; // [rsp+60h] [rbp-60h]
  __int64 **v101; // [rsp+70h] [rbp-50h] BYREF
  __int64 **v102; // [rsp+78h] [rbp-48h] BYREF
  _QWORD *v103[8]; // [rsp+80h] [rbp-40h] BYREF

  v10 = (__int64 *)*((_QWORD *)a2 - 3);
  if ( *((_BYTE *)v10 + 16) > 0x10u )
    return 0;
  v11 = *((_QWORD *)a2 - 6);
  v15 = sub_1713A90(a1, a2, a3, a4, a5, a6, a7, a8, a9, a10);
  if ( v15 )
    return v15;
  v17 = *(_QWORD *)(v11 + 8);
  v18 = *(unsigned __int8 *)(v11 + 16);
  if ( v17 && !*(_QWORD *)(v17 + 8) )
  {
    if ( (_BYTE)v18 == 37 )
    {
      v17 = *(_QWORD *)(v11 - 48);
      if ( !v17 )
        goto LABEL_9;
      v94 = *(__int64 **)(v11 - 48);
      v51 = *(_QWORD *)(v11 - 24);
      if ( !v51 )
        goto LABEL_9;
    }
    else
    {
      if ( (_BYTE)v18 != 5 )
      {
        if ( (unsigned __int8)v18 <= 0x17u )
          goto LABEL_11;
        goto LABEL_9;
      }
      if ( *(_WORD *)(v11 + 18) != 13
        || (v67 = *(_DWORD *)(v11 + 20) & 0xFFFFFFF, v14 = 4 * v67, (v17 = *(_QWORD *)(v11 - 24 * v67)) == 0)
        || (v94 = *(__int64 **)(v11 - 24LL * (*(_DWORD *)(v11 + 20) & 0xFFFFFFF)),
            v17 = 1 - v67,
            (v51 = *(_QWORD *)(v11 + 24 * (1 - v67))) == 0) )
      {
LABEL_47:
        v19 = *(unsigned __int16 *)(v11 + 18);
        goto LABEL_10;
      }
    }
    if ( (unsigned __int8)sub_17198D0(v10, (__int64)a2, v17, v14) )
    {
      v56 = a1[1];
      LOWORD(v103[0]) = 257;
      v100 = 257;
      v57 = (__int64 *)sub_171CA90(v56, v51, (__int64 *)&v98, *(double *)a3.m128_u64, a4, a5);
      return sub_15FB440(11, v57, (__int64)v94, (__int64)&v101, 0);
    }
    v18 = *(unsigned __int8 *)(v11 + 16);
  }
  if ( (unsigned __int8)v18 <= 0x17u )
  {
    if ( (_BYTE)v18 != 5 )
      goto LABEL_11;
    goto LABEL_47;
  }
LABEL_9:
  v19 = v18 - 24;
LABEL_10:
  if ( v19 == 37 )
  {
    if ( (*(_BYTE *)(v11 + 23) & 0x40) != 0 )
    {
      v52 = *(__int64 ***)(v11 - 8);
    }
    else
    {
      v17 = 24LL * (*(_DWORD *)(v11 + 20) & 0xFFFFFFF);
      v52 = (__int64 **)(v11 - v17);
    }
    v53 = *v52;
    if ( v53 )
    {
      v94 = v53;
      if ( (unsigned int)sub_16431D0(*v53) == 1 )
      {
        LOWORD(v103[0]) = 257;
        v54 = sub_15A0680(*v10, 1, 0);
        v55 = (_QWORD *)sub_15A2B30(v10, v54, 0, 0, *(double *)a3.m128_u64, a4, a5);
        return sub_14EDD70((__int64)v94, v55, (__int64)v10, (__int64)&v101, 0, 0);
      }
    }
  }
LABEL_11:
  v101 = &v94;
  if ( sub_171DA10(&v101, v11, v17, v14) )
  {
    v20 = v94;
    LOWORD(v103[0]) = 257;
    v21 = sub_15A0680(*v10, 1, 0);
    v22 = (__int64 *)sub_15A2B60(v10, v21, 0, 0, *(double *)a3.m128_u64, a4, a5);
    return sub_15FB440(13, v22, (__int64)v20, (__int64)&v101, 0);
  }
  v101 = &v95;
  if ( !(unsigned __int8)sub_13D2630(&v101, v10) )
    return 0;
  v23 = *((_DWORD *)v95 + 2);
  v24 = *v95;
  v25 = v23 - 1;
  if ( v23 <= 0x40 )
  {
    if ( v24 != 1LL << v25 )
      goto LABEL_17;
LABEL_49:
    if ( sub_15F2380((__int64)a2) || sub_15F2370((__int64)a2) )
    {
      LOWORD(v103[0]) = 257;
      return sub_15FB440(27, (__int64 *)v11, (__int64)v10, (__int64)&v101, 0);
    }
    else
    {
      LOWORD(v103[0]) = 257;
      return sub_15FB440(28, (__int64 *)v11, (__int64)v10, (__int64)&v101, 0);
    }
  }
  if ( (*(_QWORD *)(v24 + 8LL * (v25 >> 6)) & (1LL << v25)) != 0 && v25 == (unsigned int)sub_16A58A0((__int64)v95) )
    goto LABEL_49;
LABEL_17:
  v26 = *(_QWORD *)a2;
  v99 = &v96;
  v98 = (unsigned __int64)&v94;
  v27 = *(unsigned __int8 *)(v11 + 16);
  if ( (unsigned __int8)v27 > 0x17u )
  {
    v58 = v27 - 24;
LABEL_66:
    if ( v58 != 37 )
      goto LABEL_19;
    v59 = (__int64 *)sub_13CF970(v11);
    if ( !(unsigned __int8)sub_171CFA0((_QWORD **)&v98, *v59, v60, v61) )
      goto LABEL_19;
    v62 = v96;
    v63 = *((_DWORD *)v96 + 2);
    v64 = *v96;
    v65 = v63 - 1;
    if ( v63 <= 0x40 )
    {
      if ( v64 != 1LL << v65 )
        goto LABEL_19;
    }
    else
    {
      if ( (*(_QWORD *)(v64 + 8LL * (v65 >> 6)) & (1LL << v65)) == 0 )
        goto LABEL_19;
      v62 = v96;
      if ( v65 != (unsigned int)sub_16A58A0((__int64)v96) )
        goto LABEL_19;
    }
    v87 = (const void **)v95;
    v66 = sub_16431D0(v26);
    sub_16A5B10((__int64)&v101, v62, v66);
    if ( (unsigned int)v102 <= 0x40 )
    {
      if ( v101 == *v87 )
        goto LABEL_73;
    }
    else if ( sub_16A5220((__int64)&v101, v87) )
    {
LABEL_73:
      sub_135E100((__int64 *)&v101);
      LOWORD(v103[0]) = 257;
      return sub_15FDBD0(38, (__int64)v94, v26, (__int64)&v101, 0);
    }
    sub_135E100((__int64 *)&v101);
    goto LABEL_19;
  }
  if ( (_BYTE)v27 == 5 )
  {
    v58 = *(unsigned __int16 *)(v11 + 18);
    goto LABEL_66;
  }
LABEL_19:
  v97[1] = &v96;
  v97[0] = &v94;
  v28 = sub_171E890(v97, v11);
  v30 = v95;
  if ( !v28 )
    goto LABEL_23;
  v31 = *((_DWORD *)v95 + 2);
  v32 = *v95;
  v29 = v31 - 1;
  if ( v31 > 0x40 )
  {
    v29 = (unsigned int)v29 >> 6;
    v32 = *(_QWORD *)(v32 + 8LL * (unsigned int)v29);
  }
  if ( (v32 & (1LL << ((unsigned __int8)v31 - 1))) == 0 )
  {
LABEL_23:
    v33 = *((unsigned int *)v30 + 2);
    if ( (unsigned int)v33 <= 0x40 )
    {
      if ( *v30 != 1 )
        return 0;
    }
    else
    {
      v86 = *((_DWORD *)v30 + 2);
      v34 = sub_16A57B0((__int64)v30);
      v33 = (unsigned int)(v86 - 1);
      if ( v34 != (_DWORD)v33 )
        return 0;
    }
    v35 = *(_QWORD *)(v11 + 8);
    if ( v35 )
    {
      v15 = *(_QWORD *)(v35 + 8);
      if ( !v15 )
      {
        v36 = *(unsigned __int8 *)(v11 + 16);
        if ( (unsigned __int8)v36 > 0x17u )
        {
          v76 = v36 - 24;
        }
        else
        {
          if ( (_BYTE)v36 != 5 )
            goto LABEL_29;
          v76 = *(unsigned __int16 *)(v11 + 18);
        }
        if ( v76 != 38
          || ((*(_BYTE *)(v11 + 23) & 0x40) == 0
            ? (v33 = v11 - 24LL * (*(_DWORD *)(v11 + 20) & 0xFFFFFFF))
            : (v33 = *(_QWORD *)(v11 - 8)),
              (v77 = *(__int64 **)v33) == 0 || (v94 = *(__int64 **)v33, v78 = sub_16431D0(*v77), v15 = 0, v78 != 1)) )
        {
LABEL_29:
          v102 = &v96;
          v101 = &v94;
          v103[0] = &v98;
          v37 = *(_BYTE *)(v11 + 16);
          if ( v37 == 49 )
          {
            v91 = v15;
            v80 = sub_171EA60(&v101, *(_QWORD *)(v11 - 48), v33, v29);
            v15 = v91;
            if ( !v80 )
              return v15;
            v81 = sub_13D2630(v103, *(_BYTE **)(v11 - 24));
            v15 = v91;
            if ( !v81 )
              return v15;
            goto LABEL_39;
          }
          if ( v37 != 5 || *(_WORD *)(v11 + 18) != 25 )
            return v15;
          v38 = *(unsigned int *)(v11 + 20);
          v39 = *(_QWORD *)(v11 - 24LL * (*(_DWORD *)(v11 + 20) & 0xFFFFFFF));
          v40 = *(_BYTE *)(v39 + 16);
          if ( v40 == 47 )
          {
            if ( !*(_QWORD *)(v39 - 48) )
              return v15;
            v94 = *(__int64 **)(v39 - 48);
            v92 = v15;
            v82 = sub_13D2630(&v102, *(_BYTE **)(v39 - 24));
            v15 = v92;
            if ( !v82 )
              return v15;
          }
          else
          {
            if ( v40 != 5 )
              return v15;
            if ( *(_WORD *)(v39 + 18) != 23 )
              return v15;
            v41 = *(_DWORD *)(v39 + 20) & 0xFFFFFFF;
            if ( !*(_QWORD *)(v39 - 24 * v41) )
              return v15;
            v94 = *(__int64 **)(v39 - 24 * v41);
            v42 = 24 * (1 - v41);
            v43 = *(__int64 **)(v39 + v42);
            if ( *((_BYTE *)v43 + 16) == 13 )
            {
              v96 = v43 + 3;
              goto LABEL_38;
            }
            if ( *(_BYTE *)(*v43 + 8) != 16 )
              return v15;
            v93 = v15;
            v83 = sub_15A1020(v43, v41, v38, v42);
            v15 = v93;
            if ( !v83 || *(_BYTE *)(v83 + 16) != 13 )
              return v15;
            *v102 = (__int64 *)(v83 + 24);
          }
          LODWORD(v38) = *(_DWORD *)(v11 + 20);
LABEL_38:
          v88 = v15;
          v44 = sub_13D7780(v103, *(_BYTE **)(v11 + 24 * (1 - (v38 & 0xFFFFFFF))));
          v15 = v88;
          if ( !v44 )
            return v15;
LABEL_39:
          v45 = v96;
          if ( v96 == (__int64 *)v98 )
          {
            v89 = v15;
            v46 = sub_16431D0(v26);
            v47 = sub_13A38F0((__int64)v45, (_QWORD *)(unsigned int)(v46 - 1));
            v15 = v89;
            if ( v47 )
            {
              v48 = a1[1];
              LOWORD(v103[0]) = 257;
              v49 = (__int64 *)sub_171CA90(v48, (__int64)v94, (__int64 *)&v101, *(double *)a3.m128_u64, a4, a5);
              LOWORD(v103[0]) = 257;
              v50 = sub_15A0680(v26, 1, 0);
              return sub_15FB440(26, v49, v50, (__int64)&v101, 0);
            }
          }
          return v15;
        }
        v79 = a1[1];
        v100 = 257;
        v73 = sub_171CA90(v79, (__int64)v77, (__int64 *)&v98, *(double *)a3.m128_u64, a4, a5);
        goto LABEL_87;
      }
    }
    return 0;
  }
  v84 = v95;
  sub_16A5B10((__int64)&v98, v96, v31);
  v68 = v84;
  if ( (unsigned int)v99 > 0x40 )
  {
    sub_16A8F40((__int64 *)&v98);
    v68 = v84;
  }
  else
  {
    v98 = ~v98 & (0xFFFFFFFFFFFFFFFFLL >> -(char)v99);
  }
  v85 = (__int64)v68;
  sub_16A7400((__int64)&v98);
  v69 = (int)v99;
  LODWORD(v99) = 0;
  LODWORD(v102) = v69;
  v101 = (__int64 **)v98;
  LODWORD(v85) = sub_16AEA10(v85, (__int64)&v101);
  sub_135E100((__int64 *)&v101);
  sub_135E100((__int64 *)&v98);
  if ( (int)v85 < 0 )
  {
    v30 = v95;
    goto LABEL_23;
  }
  sub_16A5A50((__int64)&v98, v95, *((_DWORD *)v96 + 2));
  sub_16A7200((__int64)&v98, v96);
  v70 = (int)v99;
  LODWORD(v99) = 0;
  LODWORD(v102) = v70;
  v101 = (__int64 **)v98;
  v71 = sub_15A1070(*v94, (__int64)&v101);
  sub_135E100((__int64 *)&v101);
  sub_135E100((__int64 *)&v98);
  v72 = a1[1];
  v100 = 257;
  v73 = sub_17094A0(v72, (__int64)v94, v71, (__int64 *)&v98, 1u, 0, *(double *)a3.m128_u64, a4, a5);
LABEL_87:
  v74 = (__int64)v73;
  LOWORD(v103[0]) = 257;
  v75 = sub_1648A60(56, 1u);
  v15 = (__int64)v75;
  if ( v75 )
  {
    v90 = v75;
    sub_15FC690((__int64)v75, v74, v26, (__int64)&v101, 0);
    return (__int64)v90;
  }
  return v15;
}
