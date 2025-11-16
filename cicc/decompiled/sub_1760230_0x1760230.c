// Function: sub_1760230
// Address: 0x1760230
//
__int64 __fastcall sub_1760230(
        __int64 *a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __m128 a6,
        double a7,
        double a8,
        double a9,
        double a10,
        double a11,
        double a12,
        __m128 a13)
{
  __int64 v14; // rbx
  _BYTE *v15; // r14
  char v16; // al
  __int64 v17; // r12
  unsigned __int64 v18; // r15
  unsigned int v19; // ecx
  __int64 v20; // rdx
  __int64 v21; // rdi
  unsigned int v22; // esi
  int v23; // eax
  __int64 v24; // r8
  unsigned int v25; // r12d
  unsigned int i; // eax
  __int64 v27; // rcx
  unsigned __int64 v28; // rdx
  int v29; // r9d
  char v30; // cl
  __int64 v31; // rdx
  __int64 v33; // r12
  __int64 v34; // rax
  __int64 v35; // r12
  unsigned int v36; // r12d
  __int64 v37; // r15
  int v38; // ebx
  int v39; // eax
  int v40; // eax
  __int64 *v41; // rdi
  __int64 v42; // rax
  char v43; // dl
  bool v44; // al
  int v45; // eax
  int v46; // eax
  int v47; // eax
  int v48; // eax
  __int64 v49; // rax
  __int64 v50; // r15
  __int64 v51; // rdi
  __int64 *v52; // r12
  __int64 v53; // rax
  __int64 v54; // rdi
  unsigned __int8 *v55; // rax
  __int64 v56; // r12
  __int64 v57; // rdi
  __int64 v58; // rax
  __int64 v59; // r15
  __int64 v60; // rdi
  __int64 *v61; // r12
  __int64 v62; // rax
  __int64 v63; // rdi
  unsigned __int8 *v64; // rax
  __int64 v65; // rax
  double v66; // xmm4_8
  double v67; // xmm5_8
  _QWORD *v68; // rax
  __int64 v69; // rax
  __int64 v70; // r13
  _QWORD *v71; // rax
  _QWORD *v72; // rax
  __int64 v73; // rax
  __int64 v74; // rdi
  __int64 v75; // rax
  __int64 v76; // r13
  _QWORD *v77; // rax
  __int64 v78; // rdi
  unsigned __int8 *v79; // rax
  __int64 v80; // r15
  __int64 v81; // r14
  __int64 v82; // rax
  _QWORD *v83; // rax
  __int64 v84; // r13
  __int64 v85; // r14
  unsigned __int8 *v86; // rax
  unsigned __int8 *v87; // r13
  __int64 v88; // rax
  __int64 v89; // r14
  _QWORD *v90; // rax
  __int64 v91; // rax
  __int64 v92; // rdi
  __int64 v93; // [rsp+0h] [rbp-D0h]
  unsigned __int64 v94; // [rsp+8h] [rbp-C8h]
  __int64 v95; // [rsp+10h] [rbp-C0h]
  int v96; // [rsp+1Ch] [rbp-B4h]
  int v97; // [rsp+20h] [rbp-B0h]
  int v98; // [rsp+24h] [rbp-ACh]
  int v99; // [rsp+28h] [rbp-A8h]
  unsigned int v100; // [rsp+2Ch] [rbp-A4h]
  int v101; // [rsp+2Ch] [rbp-A4h]
  __int64 v102; // [rsp+30h] [rbp-A0h]
  __int64 v103; // [rsp+30h] [rbp-A0h]
  __int64 v104; // [rsp+38h] [rbp-98h]
  int v105; // [rsp+38h] [rbp-98h]
  int v106; // [rsp+38h] [rbp-98h]
  int v107; // [rsp+40h] [rbp-90h]
  unsigned int v108; // [rsp+48h] [rbp-88h]
  int v110; // [rsp+50h] [rbp-80h]
  __int64 *v111; // [rsp+50h] [rbp-80h]
  __int64 v113[2]; // [rsp+60h] [rbp-70h] BYREF
  __int16 v114; // [rsp+70h] [rbp-60h]
  unsigned int *v115; // [rsp+80h] [rbp-50h] BYREF
  __int64 v116; // [rsp+88h] [rbp-48h]
  _BYTE v117[64]; // [rsp+90h] [rbp-40h] BYREF

  v14 = a2;
  v15 = *(_BYTE **)(a3 - 24);
  v16 = v15[16];
  if ( v16 != 11 )
  {
    v17 = 0;
    if ( v16 != 6 )
      return v17;
  }
  v18 = *(_QWORD *)(*(_QWORD *)v15 + 32LL);
  if ( a1[342] < v18 )
    return 0;
  v19 = *(_DWORD *)(a2 + 20) & 0xFFFFFFF;
  if ( v19 <= 2 )
    return 0;
  v20 = v19;
  v17 = 0;
  v21 = *(_QWORD *)(a2 + 24 * (1LL - v19));
  if ( *(_BYTE *)(v21 + 16) != 13 )
    return v17;
  v22 = *(_DWORD *)(v21 + 32);
  if ( v22 <= 0x40 )
  {
    if ( *(_QWORD *)(v21 + 24) )
      return v17;
  }
  else
  {
    v104 = v19;
    v108 = v19;
    v23 = sub_16A57B0(v21 + 24);
    v19 = v108;
    v20 = v104;
    if ( v22 != v23 )
      return v17;
  }
  if ( *(_BYTE *)(*(_QWORD *)(v14 + 24 * (2 - v20)) + 16LL) <= 0x10u )
    return 0;
  v115 = (unsigned int *)v117;
  v116 = 0x400000000LL;
  v24 = **(_QWORD **)(*(_QWORD *)v15 + 16LL);
  if ( v19 != 3 )
  {
    v25 = v19 - 1;
    for ( i = 3; ; ++i )
    {
      v27 = *(_QWORD *)(v14 + 24 * (i - v20));
      if ( *(_BYTE *)(v27 + 16) != 13 )
        break;
      v28 = *(_QWORD *)(v27 + 24);
      if ( *(_DWORD *)(v27 + 32) > 0x40u )
        v28 = *(_QWORD *)v28;
      v29 = v28;
      if ( v28 != (unsigned int)v28 )
        break;
      v30 = *(_BYTE *)(v24 + 8);
      if ( v30 == 13 )
      {
        v24 = *(_QWORD *)(*(_QWORD *)(v24 + 16) + 8 * v28);
      }
      else
      {
        if ( v30 != 14 || *(_QWORD *)(v24 + 32) <= v28 )
          break;
        v24 = *(_QWORD *)(v24 + 24);
      }
      v31 = (unsigned int)v116;
      if ( (unsigned int)v116 >= HIDWORD(v116) )
      {
        v100 = i;
        v102 = v24;
        v105 = v29;
        sub_16CD150((__int64)&v115, v117, 0, 4, v24, v29);
        v31 = (unsigned int)v116;
        i = v100;
        v24 = v102;
        v29 = v105;
      }
      v115[v31] = v29;
      LODWORD(v116) = v116 + 1;
      if ( v25 == i )
        goto LABEL_32;
      v20 = *(_DWORD *)(v14 + 20) & 0xFFFFFFF;
    }
LABEL_24:
    v17 = 0;
    goto LABEL_25;
  }
LABEL_32:
  v101 = v18;
  v103 = *(_QWORD *)(a4 - 24);
  if ( (_DWORD)v18 )
  {
    v36 = 0;
    v94 = v18;
    v37 = a5;
    v93 = v14;
    v38 = -1;
    v95 = 0;
    v106 = -2;
    v107 = -2;
    v98 = -2;
    v96 = -2;
    v99 = -2;
    v97 = -2;
    while ( 1 )
    {
      v41 = (__int64 *)sub_15A0A60((__int64)v15, v36);
      if ( !v41 )
        goto LABEL_24;
      if ( (_DWORD)v116 )
        v41 = (__int64 *)sub_15A3AE0(v41, v115, (unsigned int)v116, 0);
      if ( v37 )
        v41 = (__int64 *)sub_15A2CF0(v41, v37, *(double *)a6.m128_u64, a7, a8);
      v42 = sub_14D7760(*(_WORD *)(a4 + 18) & 0x7FFF, v41, v103, a1[333], a1[331]);
      v43 = *(_BYTE *)(v42 + 16);
      if ( v43 == 9 )
      {
        v47 = v107;
        if ( v38 == v107 )
          v47 = v36;
        v107 = v47;
        v48 = v106;
        if ( v106 == v38 )
          v48 = v36;
        v106 = v48;
      }
      else
      {
        if ( v43 != 13 )
          goto LABEL_24;
        if ( *(_DWORD *)(v42 + 32) <= 0x40u )
        {
          v44 = *(_QWORD *)(v42 + 24) == 0;
        }
        else
        {
          v110 = *(_DWORD *)(v42 + 32);
          v44 = v110 == (unsigned int)sub_16A57B0(v42 + 24);
        }
        if ( v44 )
        {
          if ( v96 == -2 )
          {
            v106 = v36;
            v96 = v36;
          }
          else
          {
            v45 = -3;
            if ( v98 == -2 )
              v45 = v36;
            v98 = v45;
            v46 = -3;
            if ( v106 == v38 )
              v46 = v36;
            v106 = v46;
          }
LABEL_59:
          if ( (v36 & 8) == 0 && v36 > 0x3F && v99 == -3 )
          {
            if ( v98 == -3 && v107 == -3 )
            {
              if ( v106 == -3 )
                goto LABEL_24;
              v107 = -3;
              v98 = -3;
              v99 = -3;
            }
            else
            {
              v99 = -3;
            }
          }
          goto LABEL_42;
        }
        if ( v97 == -2 )
        {
          v107 = v36;
          v97 = v36;
        }
        else
        {
          v39 = -3;
          if ( v99 == -2 )
            v39 = v36;
          v99 = v39;
          v40 = -3;
          if ( v38 == v107 )
            v40 = v36;
          v107 = v40;
        }
        if ( v36 > 0x3F )
          goto LABEL_59;
        v95 |= 1LL << v36;
      }
LABEL_42:
      ++v36;
      ++v38;
      if ( v101 == v36 )
      {
        v18 = v94;
        v14 = v93;
        goto LABEL_76;
      }
    }
  }
  v95 = 0;
  v106 = -2;
  v107 = -2;
  v98 = -2;
  v96 = -2;
  v99 = -2;
  v97 = -2;
LABEL_76:
  v111 = *(__int64 **)(v14 + 24 * (2LL - (*(_DWORD *)(v14 + 20) & 0xFFFFFFF)));
  if ( !sub_15FA300(v14) )
  {
    v56 = sub_15A9650(a1[333], *(_QWORD *)v14);
    if ( (unsigned int)sub_1643030(*v111) > *(_DWORD *)(v56 + 8) >> 8 )
    {
      v57 = a1[1];
      v114 = 257;
      v111 = (__int64 *)sub_1708970(v57, 36, (__int64)v111, (__int64 **)v56, v113);
    }
  }
  if ( v99 == -3 )
  {
    if ( v98 == -3 )
    {
      v35 = *v111;
      if ( v107 != -3 )
      {
        if ( v97 )
        {
          v73 = sub_15A0680(*v111, -v97, 0);
          v74 = a1[1];
          v114 = 257;
          v111 = (__int64 *)sub_17094A0(v74, (__int64)v111, v73, v113, 0, 0, *(double *)a6.m128_u64, a7, a8);
        }
        v69 = sub_15A0680(*v111, v107 - v97 + 1, 0);
        v114 = 257;
        v70 = v69;
        v71 = sub_1648A60(56, 2u);
        v17 = (__int64)v71;
        if ( v71 )
          sub_17582E0((__int64)v71, 36, (__int64)v111, v70, (__int64)v113);
        goto LABEL_25;
      }
      if ( v106 != -3 )
      {
        if ( v96 )
        {
          v91 = sub_15A0680(*v111, -v96, 0);
          v92 = a1[1];
          v114 = 257;
          v111 = (__int64 *)sub_17094A0(v92, (__int64)v111, v91, v113, 0, 0, *(double *)a6.m128_u64, a7, a8);
        }
        v75 = sub_15A0680(*v111, v106 - v96, 0);
        v114 = 257;
        v76 = v75;
        v77 = sub_1648A60(56, 2u);
        v17 = (__int64)v77;
        if ( v77 )
          sub_17582E0((__int64)v77, 34, (__int64)v111, v76, (__int64)v113);
        goto LABEL_25;
      }
      if ( *(_DWORD *)(v35 + 8) >> 8 >= v18
        || (v33 = a1[333], v34 = sub_16498A0((__int64)v15), (v35 = sub_15A9690(v33, v34, v18)) != 0) )
      {
        v78 = a1[1];
        v114 = 257;
        v79 = sub_1759360(v78, (__int64)v111, (__int64 **)v35, 0, v113);
        v80 = a1[1];
        v81 = (__int64)v79;
        v114 = 257;
        v82 = sub_15A0680(v35, v95, 0);
        v83 = sub_172C310(v80, v82, v81, v113, 0, *(double *)a6.m128_u64, a7, a8);
        v84 = a1[1];
        v85 = (__int64)v83;
        v114 = 257;
        v86 = (unsigned __int8 *)sub_15A0680(v35, 1, 0);
        v87 = sub_1729500(v84, v86, v85, v113, *(double *)a6.m128_u64, a7, a8);
        v88 = sub_15A0680(v35, 0, 0);
        v114 = 257;
        v89 = v88;
        v90 = sub_1648A60(56, 2u);
        v17 = (__int64)v90;
        if ( v90 )
          sub_17582E0((__int64)v90, 33, (__int64)v87, v89, (__int64)v113);
        goto LABEL_25;
      }
      goto LABEL_24;
    }
    if ( v96 != -2 )
    {
      v58 = sub_15A0680(*v111, v96, 0);
      v59 = v58;
      if ( v98 == -2 )
      {
        v114 = 257;
        v72 = sub_1648A60(56, 2u);
        v17 = (__int64)v72;
        if ( v72 )
          sub_17582E0((__int64)v72, 33, (__int64)v111, v59, (__int64)v113);
      }
      else
      {
        v60 = a1[1];
        v114 = 257;
        v61 = (__int64 *)sub_17203D0(v60, 33, (__int64)v111, v58, v113);
        v62 = sub_15A0680(*v111, v98, 0);
        v63 = a1[1];
        v114 = 257;
        v64 = sub_17203D0(v63, 33, (__int64)v111, v62, v113);
        v114 = 257;
        v17 = sub_15FB440(26, v61, (__int64)v64, (__int64)v113, 0);
      }
      goto LABEL_25;
    }
    v65 = sub_159C4F0(*(__int64 **)(a1[1] + 24));
  }
  else
  {
    if ( v97 != -2 )
    {
      v49 = sub_15A0680(*v111, v97, 0);
      v50 = v49;
      if ( v99 == -2 )
      {
        v114 = 257;
        v68 = sub_1648A60(56, 2u);
        v17 = (__int64)v68;
        if ( v68 )
          sub_17582E0((__int64)v68, 32, (__int64)v111, v50, (__int64)v113);
      }
      else
      {
        v51 = a1[1];
        v114 = 257;
        v52 = (__int64 *)sub_17203D0(v51, 32, (__int64)v111, v49, v113);
        v53 = sub_15A0680(*v111, v99, 0);
        v54 = a1[1];
        v114 = 257;
        v55 = sub_17203D0(v54, 32, (__int64)v111, v53, v113);
        v114 = 257;
        v17 = sub_15FB440(27, v52, (__int64)v55, (__int64)v113, 0);
      }
      goto LABEL_25;
    }
    v65 = sub_159C540(*(__int64 **)(a1[1] + 24));
  }
  v17 = sub_170E100(a1, a4, v65, a6, a7, a8, a9, v66, v67, a12, a13);
LABEL_25:
  if ( v115 != (unsigned int *)v117 )
    _libc_free((unsigned __int64)v115);
  return v17;
}
