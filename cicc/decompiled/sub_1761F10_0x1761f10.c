// Function: sub_1761F10
// Address: 0x1761f10
//
__int64 __fastcall sub_1761F10(
        __int64 *a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __m128 a5,
        double a6,
        double a7,
        double a8,
        double a9,
        double a10,
        double a11,
        __m128 a12)
{
  int v14; // r12d
  _BYTE *v15; // rdi
  unsigned __int8 v16; // al
  __int64 v17; // rbx
  unsigned int v18; // r15d
  unsigned __int64 v19; // rax
  bool v20; // al
  char v21; // bl
  bool v22; // al
  unsigned int v23; // eax
  const char *v24; // rdx
  bool v25; // bl
  int v26; // eax
  __int64 v27; // rax
  unsigned int v28; // esi
  unsigned int v29; // ebx
  __int64 v30; // rbx
  __int64 v31; // rdx
  __int64 v32; // rax
  __int64 v33; // r14
  _QWORD *v34; // rax
  _QWORD *v35; // r13
  double v37; // xmm4_8
  double v38; // xmm5_8
  __int64 v39; // rax
  int v40; // eax
  _BYTE *v41; // rdi
  unsigned __int8 v42; // al
  __int64 v43; // r8
  __int64 v44; // rax
  __int64 v45; // rax
  int v46; // eax
  const char *v47; // rdx
  unsigned __int64 v48; // rdx
  __int64 v49; // rsi
  __int64 v50; // rdx
  __int64 v51; // rsi
  unsigned __int64 v52; // rdx
  __int64 v53; // rax
  __int64 v54; // r14
  __int64 v55; // rax
  unsigned int v56; // eax
  __int64 v57; // rdx
  __int64 v58; // rcx
  unsigned int v59; // r8d
  __int64 v60; // r14
  _QWORD *v61; // rax
  unsigned int v62; // eax
  int v63; // eax
  int v64; // eax
  __int64 v65; // rax
  __int64 v66; // r12
  _QWORD *v67; // rax
  int v68; // eax
  __int64 v69; // rdx
  __int64 v70; // rcx
  unsigned int v71; // r8d
  int v72; // eax
  int v73; // eax
  __int64 v74; // rsi
  __int64 v75; // rax
  __int64 v76; // rsi
  __int64 v77; // rax
  __int64 v78; // r12
  _QWORD *v79; // rax
  char v80; // [rsp+18h] [rbp-128h]
  bool v81; // [rsp+38h] [rbp-108h]
  __int64 v82; // [rsp+40h] [rbp-100h]
  char v83; // [rsp+4Bh] [rbp-F5h]
  unsigned int v84; // [rsp+4Ch] [rbp-F4h]
  unsigned __int64 v85; // [rsp+50h] [rbp-F0h]
  unsigned int v86; // [rsp+50h] [rbp-F0h]
  unsigned __int8 *v88; // [rsp+60h] [rbp-E0h]
  __int64 v90; // [rsp+68h] [rbp-D8h]
  unsigned __int8 *v91; // [rsp+68h] [rbp-D8h]
  unsigned __int64 v92; // [rsp+70h] [rbp-D0h] BYREF
  int v93; // [rsp+78h] [rbp-C8h]
  __int64 v94; // [rsp+80h] [rbp-C0h] BYREF
  int v95; // [rsp+88h] [rbp-B8h]
  __int64 v96; // [rsp+90h] [rbp-B0h] BYREF
  int v97; // [rsp+98h] [rbp-A8h]
  unsigned __int64 v98; // [rsp+A0h] [rbp-A0h] BYREF
  int v99; // [rsp+A8h] [rbp-98h]
  __int64 v100; // [rsp+B0h] [rbp-90h] BYREF
  int v101; // [rsp+B8h] [rbp-88h]
  __int64 v102; // [rsp+C0h] [rbp-80h] BYREF
  int v103; // [rsp+C8h] [rbp-78h]
  unsigned __int64 v104; // [rsp+D0h] [rbp-70h] BYREF
  unsigned int v105; // [rsp+D8h] [rbp-68h]
  unsigned __int64 v106; // [rsp+E0h] [rbp-60h] BYREF
  __int64 v107; // [rsp+E8h] [rbp-58h]
  unsigned __int64 v108; // [rsp+F0h] [rbp-50h] BYREF
  char *v109; // [rsp+F8h] [rbp-48h]
  __int16 v110; // [rsp+100h] [rbp-40h]

  v14 = *(_WORD *)(a2 + 18) & 0x7FFF;
  v88 = *(unsigned __int8 **)(a3 - 48);
  if ( (unsigned int)(v14 - 32) > 1 )
    goto LABEL_2;
  if ( sub_15F23D0(a3) )
  {
    v39 = *(_QWORD *)(a3 + 8);
    if ( v39 )
    {
      if ( !*(_QWORD *)(v39 + 8) && sub_13D01C0(a4) )
      {
        v110 = 257;
        v60 = *(_QWORD *)(a2 - 24);
        v61 = sub_1648A60(56, 2u);
        v35 = v61;
        if ( v61 )
          sub_17582E0((__int64)v61, v14, (__int64)v88, v60, (__int64)&v108);
        return (__int64)v35;
      }
    }
  }
  v40 = *(unsigned __int16 *)(a2 + 18);
  BYTE1(v40) &= ~0x80u;
  if ( (unsigned int)(v40 - 32) > 1
    || (v41 = *(_BYTE **)(a3 - 48), v42 = v41[16], v43 = (__int64)(v41 + 24), v42 != 13)
    && (*(_BYTE *)(*(_QWORD *)v41 + 8LL) != 16
     || v42 > 0x10u
     || (v45 = sub_15A1020(v41, a2, *(_QWORD *)v41, a4)) == 0
     || (v43 = v45 + 24, *(_BYTE *)(v45 + 16) != 13)) )
  {
LABEL_2:
    v15 = *(_BYTE **)(a3 - 24);
    v16 = v15[16];
    v17 = (__int64)(v15 + 24);
    if ( v16 != 13 )
    {
      if ( *(_BYTE *)(*(_QWORD *)v15 + 8LL) != 16 )
        return 0;
      if ( v16 > 0x10u )
        return 0;
      v44 = sub_15A1020(v15, a2, *(_QWORD *)v15, a4);
      if ( !v44 || *(_BYTE *)(v44 + 16) != 13 )
        return 0;
      v17 = v44 + 24;
    }
    v18 = *(_DWORD *)(v17 + 8);
    v84 = *(_DWORD *)(a4 + 8);
    if ( v18 <= 0x40 )
    {
      if ( (unsigned __int64)*(unsigned int *)(a4 + 8) < *(_QWORD *)v17 )
        return 0;
      v86 = *(_QWORD *)v17;
      v20 = v84 <= v86 || v86 == 0;
    }
    else
    {
      v85 = *(unsigned int *)(a4 + 8);
      if ( v18 - (unsigned int)sub_16A57B0(v17) > 0x40 )
        return 0;
      v19 = **(_QWORD **)v17;
      if ( v85 < v19 )
        return 0;
      v86 = **(_QWORD **)v17;
      v20 = (_DWORD)v19 == 0 || v84 <= (unsigned int)v19;
    }
    if ( v20 )
      return 0;
    v21 = *(_BYTE *)(a3 + 16);
    v22 = sub_15F23D0(a3);
    v82 = *(_QWORD *)a3;
    if ( v21 != 49 )
    {
      if ( v14 != 36 )
      {
        if ( v14 != 34 )
          goto LABEL_24;
        if ( !v22 )
          goto LABEL_23;
      }
      v23 = *(_DWORD *)(a4 + 8);
      LODWORD(v107) = v23;
      if ( v23 > 0x40 )
      {
        sub_16A4FD0((__int64)&v106, (const void **)a4);
        v23 = v107;
        if ( (unsigned int)v107 > 0x40 )
        {
          sub_16A7DC0((__int64 *)&v106, v86);
          v23 = v107;
          LODWORD(v109) = v107;
          if ( (unsigned int)v107 > 0x40 )
          {
            sub_16A4FD0((__int64)&v108, (const void **)&v106);
            v23 = (unsigned int)v109;
            if ( (unsigned int)v109 > 0x40 )
            {
              sub_16A8110((__int64)&v108, v86);
              if ( (unsigned int)v109 > 0x40 )
              {
                v25 = sub_16A5220((__int64)&v108, (const void **)a4);
                if ( v108 )
                  j_j___libc_free_0_0(v108);
                goto LABEL_21;
              }
LABEL_20:
              v25 = v108 == *(_QWORD *)a4;
LABEL_21:
              if ( v25 )
                goto LABEL_74;
              sub_135E100((__int64 *)&v106);
LABEL_23:
              if ( v14 == 34 )
              {
                sub_13A38D0((__int64)&v104, a4);
                sub_16A7490((__int64)&v104, 1);
                v62 = v105;
                v105 = 0;
                LODWORD(v107) = v62;
                v106 = v104;
                sub_13A38D0((__int64)&v108, (__int64)&v106);
                sub_1757210((__int64 *)&v108, v86);
                sub_16A7800((__int64)&v108, 1u);
                v99 = (int)v109;
                v98 = v108;
                sub_135E100((__int64 *)&v106);
                sub_135E100((__int64 *)&v104);
                sub_13A38D0((__int64)&v106, a4);
                sub_16A7490((__int64)&v106, 1);
                v63 = v107;
                LODWORD(v107) = 0;
                LODWORD(v109) = v63;
                v108 = v106;
                sub_13A38D0((__int64)&v100, (__int64)&v98);
                sub_16A7490((__int64)&v100, 1);
                v64 = v101;
                v101 = 0;
                v103 = v64;
                v102 = v100;
                sub_13A38D0((__int64)&v104, (__int64)&v102);
                if ( v105 > 0x40 )
                {
                  sub_16A8110((__int64)&v104, v86);
                }
                else if ( v86 == v105 )
                {
                  v104 = 0;
                }
                else
                {
                  v104 >>= v86;
                }
                v80 = sub_1455820((__int64)&v104, &v108);
                sub_135E100((__int64 *)&v104);
                sub_135E100(&v102);
                sub_135E100(&v100);
                sub_135E100((__int64 *)&v108);
                sub_135E100((__int64 *)&v106);
                if ( v80 )
                {
                  v65 = sub_15A1070(v82, (__int64)&v98);
                  v110 = 257;
                  v66 = v65;
                  v67 = sub_1648A60(56, 2u);
                  v35 = v67;
                  if ( v67 )
                    sub_17582E0((__int64)v67, 34, (__int64)v88, v66, (__int64)&v108);
                  sub_135E100((__int64 *)&v98);
                  return (__int64)v35;
                }
                sub_135E100((__int64 *)&v98);
              }
LABEL_24:
              v26 = *(unsigned __int16 *)(a2 + 18);
              BYTE1(v26) &= ~0x80u;
              if ( (unsigned int)(v26 - 32) <= 1 )
              {
                if ( !sub_15F23D0(a3) )
                {
                  v27 = *(_QWORD *)(a3 + 8);
                  if ( v27 && !*(_QWORD *)(v27 + 8) )
                  {
                    v28 = v86;
                    v105 = v84;
                    v29 = v86 - v84;
                    if ( v84 <= 0x40 )
                    {
                      v104 = 0;
                    }
                    else
                    {
                      sub_16A4EF0((__int64)&v104, 0, 0);
                      v84 = v105;
                      v28 = v29 + v105;
                    }
                    if ( v84 != v28 )
                    {
                      if ( v28 > 0x3F || v84 > 0x40 )
                        sub_16A5260(&v104, v28, v84);
                      else
                        v104 |= 0xFFFFFFFFFFFFFFFFLL >> ((unsigned __int8)v29 + 64) << v28;
                    }
                    v30 = sub_15A1070(v82, (__int64)&v104);
                    v90 = a1[1];
                    v106 = (unsigned __int64)sub_1649960(a3);
                    v107 = v31;
                    v110 = 773;
                    v108 = (unsigned __int64)&v106;
                    v109 = ".mask";
                    v91 = sub_1729500(v90, v88, v30, (__int64 *)&v108, *(double *)a5.m128_u64, a6, a7);
                    sub_13A38D0((__int64)&v106, a4);
                    sub_1757210((__int64 *)&v106, v86);
                    v32 = sub_15A1070(v82, (__int64)&v106);
                    v110 = 257;
                    v33 = v32;
                    v34 = sub_1648A60(56, 2u);
                    v35 = v34;
                    if ( v34 )
                      sub_17582E0((__int64)v34, v14, (__int64)v91, v33, (__int64)&v108);
                    sub_135E100((__int64 *)&v106);
                    sub_135E100((__int64 *)&v104);
                    return (__int64)v35;
                  }
                  return 0;
                }
                sub_13A38D0((__int64)&v106, a4);
                sub_1757210((__int64 *)&v106, v86);
LABEL_74:
                v55 = sub_15A1070(v82, (__int64)&v106);
                v110 = 257;
                v54 = v55;
                v35 = sub_1648A60(56, 2u);
                if ( v35 )
LABEL_75:
                  sub_17582E0((__int64)v35, v14, (__int64)v88, v54, (__int64)&v108);
LABEL_76:
                sub_135E100((__int64 *)&v106);
                return (__int64)v35;
              }
              return 0;
            }
LABEL_18:
            if ( v86 == v23 )
              v108 = 0;
            else
              v108 >>= v86;
            goto LABEL_20;
          }
LABEL_17:
          v108 = v106;
          goto LABEL_18;
        }
      }
      else
      {
        v106 = *(_QWORD *)a4;
      }
      v24 = 0;
      if ( v86 != v23 )
        v24 = (const char *)((v106 << v86) & (0xFFFFFFFFFFFFFFFFLL >> -(char)v23));
      v106 = (unsigned __int64)v24;
      LODWORD(v109) = v23;
      goto LABEL_17;
    }
    if ( v14 != 40 )
    {
      if ( v14 != 38 )
        goto LABEL_24;
      if ( !v22 )
      {
LABEL_78:
        if ( v14 == 38 )
        {
          sub_13A38D0((__int64)&v104, a4);
          sub_16A7490((__int64)&v104, 1);
          v56 = v105;
          v105 = 0;
          LODWORD(v107) = v56;
          v106 = v104;
          sub_13A38D0((__int64)&v108, (__int64)&v106);
          sub_1757210((__int64 *)&v108, v86);
          sub_16A7800((__int64)&v108, 1u);
          v93 = (int)v109;
          v92 = v108;
          sub_135E100((__int64 *)&v106);
          sub_135E100((__int64 *)&v104);
          if ( !(unsigned __int8)sub_1758270(a4, 1, v57, v58, v59) )
          {
            sub_13A38D0((__int64)&v94, a4);
            sub_16A7490((__int64)&v94, 1);
            v68 = v95;
            v95 = 0;
            v97 = v68;
            v96 = v94;
            sub_13A38D0((__int64)&v98, (__int64)&v96);
            sub_1757210((__int64 *)&v98, v86);
            if ( (unsigned __int8)sub_13CFF40((__int64 *)&v98, v86, v69, v70, v71) )
            {
              sub_135E100((__int64 *)&v98);
              sub_135E100(&v96);
              sub_135E100(&v94);
            }
            else
            {
              sub_13A38D0((__int64)&v106, a4);
              sub_16A7490((__int64)&v106, 1);
              v72 = v107;
              LODWORD(v107) = 0;
              LODWORD(v109) = v72;
              v108 = v106;
              sub_13A38D0((__int64)&v100, (__int64)&v92);
              sub_16A7490((__int64)&v100, 1);
              v73 = v101;
              v101 = 0;
              v103 = v73;
              v102 = v100;
              sub_13A38D0((__int64)&v104, (__int64)&v102);
              if ( v105 > 0x40 )
              {
                sub_16A5E70((__int64)&v104, v86);
              }
              else
              {
                v74 = (__int64)(v104 << (64 - (unsigned __int8)v105)) >> (64 - (unsigned __int8)v105);
                v75 = v74 >> v86;
                v76 = v74 >> 63;
                if ( v86 == v105 )
                  v75 = v76;
                v104 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v105) & v75;
              }
              v83 = sub_1455820((__int64)&v104, &v108);
              sub_135E100((__int64 *)&v104);
              sub_135E100(&v102);
              sub_135E100(&v100);
              sub_135E100((__int64 *)&v108);
              sub_135E100((__int64 *)&v106);
              sub_135E100((__int64 *)&v98);
              sub_135E100(&v96);
              sub_135E100(&v94);
              if ( v83 )
              {
                v77 = sub_15A1070(v82, (__int64)&v92);
                v110 = 257;
                v78 = v77;
                v79 = sub_1648A60(56, 2u);
                v35 = v79;
                if ( v79 )
                  sub_17582E0((__int64)v79, 38, (__int64)v88, v78, (__int64)&v108);
                sub_135E100((__int64 *)&v92);
                return (__int64)v35;
              }
            }
          }
          sub_135E100((__int64 *)&v92);
        }
        goto LABEL_24;
      }
    }
    sub_13A38D0((__int64)&v106, a4);
    v46 = v107;
    if ( (unsigned int)v107 > 0x40 )
    {
      sub_16A7DC0((__int64 *)&v106, v86);
      v46 = v107;
      LODWORD(v109) = v107;
      if ( (unsigned int)v107 > 0x40 )
      {
        sub_16A4FD0((__int64)&v108, (const void **)&v106);
        v46 = (int)v109;
        if ( (unsigned int)v109 > 0x40 )
        {
          sub_16A5E70((__int64)&v108, v86);
          if ( (unsigned int)v109 > 0x40 )
          {
            v81 = sub_16A5220((__int64)&v108, (const void **)a4);
            goto LABEL_70;
          }
          v52 = v108;
LABEL_69:
          v81 = *(_QWORD *)a4 == v52;
LABEL_70:
          sub_135E100((__int64 *)&v108);
          if ( v81 )
          {
            v53 = sub_15A1070(v82, (__int64)&v106);
            v110 = 257;
            v54 = v53;
            v35 = sub_1648A60(56, 2u);
            if ( v35 )
              goto LABEL_75;
            goto LABEL_76;
          }
          sub_135E100((__int64 *)&v106);
          goto LABEL_78;
        }
        v48 = v108;
LABEL_66:
        v49 = (__int64)(v48 << (64 - (unsigned __int8)v46)) >> (64 - (unsigned __int8)v46);
        v50 = v49 >> v86;
        v51 = v49 >> 63;
        if ( v86 == v46 )
          v50 = v51;
        v52 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v46) & v50;
        v108 = v52;
        goto LABEL_69;
      }
    }
    else
    {
      v47 = 0;
      if ( v86 != (_DWORD)v107 )
        v47 = (const char *)((v106 << v86) & (0xFFFFFFFFFFFFFFFFLL >> -(char)v107));
      v106 = (unsigned __int64)v47;
      LODWORD(v109) = v107;
    }
    v48 = v106;
    goto LABEL_66;
  }
  return sub_17617A0(a1, a2, *(_QWORD *)(a3 - 24), a4, v43, a5, a6, a7, a8, v37, v38, a11, a12);
}
