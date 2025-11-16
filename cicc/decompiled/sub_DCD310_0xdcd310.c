// Function: sub_DCD310
// Address: 0xdcd310
//
unsigned __int64 __fastcall sub_DCD310(__int64 *a1, unsigned __int16 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v5; // r12
  __int64 v6; // rcx
  unsigned __int64 *v7; // rax
  __int64 v8; // r15
  unsigned __int16 v10; // r13
  __int64 v11; // rdx
  unsigned int v12; // ebx
  __int64 v13; // r14
  _QWORD *v14; // r12
  __int64 v15; // r8
  unsigned __int64 *v16; // r12
  int v17; // eax
  __int64 v18; // rsi
  __int64 v19; // rdi
  __int64 v20; // rax
  __int64 v21; // rdx
  unsigned int v22; // eax
  __int64 v23; // r15
  __int64 v24; // rax
  int v25; // eax
  __int64 v26; // rax
  unsigned int v27; // ebx
  __int64 v28; // rdi
  __int64 v29; // rdx
  unsigned int v30; // r14d
  __int64 v31; // r9
  __int64 v32; // rax
  __int64 v33; // rdi
  unsigned int v34; // edx
  unsigned int v35; // ebx
  __int64 v36; // rax
  unsigned __int64 v37; // rcx
  unsigned __int64 v38; // rdx
  __int64 v39; // rax
  char *v40; // rbx
  char *v41; // rdi
  _QWORD *v42; // rax
  __int64 *v43; // rsi
  __int64 v44; // rcx
  __int64 v45; // rdx
  unsigned __int64 *v46; // rdx
  unsigned int v47; // ebx
  __int64 v48; // rsi
  __int64 v49; // r14
  unsigned __int64 *v50; // rdi
  unsigned __int16 v51; // r15
  unsigned __int64 v52; // r13
  unsigned __int64 *v53; // rax
  unsigned int v54; // ebx
  __int64 v55; // rdx
  unsigned int v56; // r14d
  unsigned int v57; // edx
  unsigned int v58; // ebx
  bool v59; // al
  unsigned int v60; // ebx
  unsigned int v61; // ebx
  unsigned __int64 v62; // r13
  unsigned int v63; // r14d
  __int64 v64; // rdi
  unsigned __int64 *v65; // rcx
  unsigned __int64 *v66; // rsi
  __int64 v67; // r15
  unsigned __int64 *v68; // rdx
  size_t v69; // r15
  __int64 v70; // rdx
  __int64 v71; // rcx
  __int64 v72; // r15
  __int64 v73; // r8
  __int64 v74; // r9
  __int64 v75; // rax
  __int64 v76; // rbx
  unsigned __int64 *v77; // r15
  unsigned __int64 v78; // r13
  unsigned __int64 v79; // r13
  __int64 v80; // rax
  _QWORD *v81; // rsi
  unsigned int v82; // ebx
  bool v83; // al
  unsigned __int64 *v84; // rdx
  __int64 v85; // rdi
  const void *v86; // rsi
  void *v87; // r15
  __int64 v88; // rcx
  const void *v89; // rax
  size_t v90; // rcx
  __int64 v91; // rdx
  __int64 v92; // rsi
  unsigned __int64 *v93; // r15
  void *v94; // rcx
  size_t v95; // rdx
  void *v96; // rax
  __int64 v97; // r14
  __int64 v98; // rdx
  __int16 v99; // ax
  unsigned __int64 v100; // [rsp+0h] [rbp-120h]
  unsigned __int64 v101; // [rsp+8h] [rbp-118h]
  __int64 v102; // [rsp+10h] [rbp-110h]
  unsigned int v103; // [rsp+10h] [rbp-110h]
  __int64 v104; // [rsp+18h] [rbp-108h]
  unsigned __int64 *v105; // [rsp+20h] [rbp-100h]
  unsigned __int64 *v106; // [rsp+20h] [rbp-100h]
  unsigned int v107; // [rsp+20h] [rbp-100h]
  unsigned __int16 v108; // [rsp+28h] [rbp-F8h]
  __int64 *v109; // [rsp+28h] [rbp-F8h]
  __int16 v110; // [rsp+30h] [rbp-F0h]
  __int64 v111; // [rsp+30h] [rbp-F0h]
  unsigned __int16 v112; // [rsp+3Ch] [rbp-E4h]
  size_t v113; // [rsp+40h] [rbp-E0h]
  void *v114; // [rsp+40h] [rbp-E0h]
  __int64 *v115; // [rsp+48h] [rbp-D8h]
  __int64 *v116; // [rsp+58h] [rbp-C8h] BYREF
  _BYTE *v117; // [rsp+60h] [rbp-C0h] BYREF
  __int64 v118; // [rsp+68h] [rbp-B8h]
  _BYTE v119[176]; // [rsp+70h] [rbp-B0h] BYREF

  while ( 2 )
  {
    v5 = a3;
    v6 = *(unsigned int *)(a3 + 8);
    v115 = a1;
    v112 = a2;
    if ( (_DWORD)v6 == 1 )
      goto LABEL_2;
    v10 = a2;
    v11 = 0;
    v108 = a2 - 9;
    v110 = (a2 - 10) & 0xFFFD;
    v102 = a1[6];
    v104 = a1[5];
    v12 = 0;
    if ( !(_DWORD)v6 )
      goto LABEL_58;
    v13 = v5;
    v14 = 0;
    do
    {
      while ( 1 )
      {
        v18 = *(_QWORD *)v13;
        v23 = 8 * v11;
        v19 = *(_QWORD *)v13 + 8 * v11;
        v24 = *(_QWORD *)v19;
        if ( !*(_WORD *)(*(_QWORD *)v19 + 24LL) )
          break;
        v22 = *(_DWORD *)(v13 + 8);
        v11 = ++v12;
        if ( v12 >= v22 )
          goto LABEL_34;
      }
      if ( !v14 )
      {
        v14 = *(_QWORD **)v19;
        goto LABEL_19;
      }
      v15 = v14[4];
      v16 = (unsigned __int64 *)(v15 + 24);
      if ( v10 == 11 )
      {
        v106 = (unsigned __int64 *)(*(_QWORD *)(v24 + 32) + 24LL);
        v25 = sub_C49970(v15 + 24, v106);
        goto LABEL_27;
      }
      if ( v10 > 0xBu )
      {
        if ( v10 != 12 )
          goto LABEL_149;
        v106 = (unsigned __int64 *)(*(_QWORD *)(v24 + 32) + 24LL);
        v25 = sub_C4C880(v15 + 24, (__int64)v106);
LABEL_27:
        if ( v25 >= 0 )
          v16 = v106;
        LODWORD(v118) = *((_DWORD *)v16 + 2);
        if ( (unsigned int)v118 <= 0x40 )
          goto LABEL_14;
        goto LABEL_30;
      }
      if ( v10 == 9 )
      {
        v105 = (unsigned __int64 *)(*(_QWORD *)(v24 + 32) + 24LL);
        v17 = sub_C49970(v15 + 24, v105);
      }
      else
      {
        if ( v10 != 10 )
LABEL_149:
          BUG();
        v105 = (unsigned __int64 *)(*(_QWORD *)(v24 + 32) + 24LL);
        v17 = sub_C4C880(v15 + 24, (__int64)v105);
      }
      if ( v17 <= 0 )
        v16 = v105;
      LODWORD(v118) = *((_DWORD *)v16 + 2);
      if ( (unsigned int)v118 <= 0x40 )
      {
LABEL_14:
        v117 = (_BYTE *)*v16;
        goto LABEL_15;
      }
LABEL_30:
      sub_C43780((__int64)&v117, (const void **)v16);
LABEL_15:
      v14 = sub_DA26C0(v115, (__int64)&v117);
      if ( (unsigned int)v118 > 0x40 && v117 )
        j_j___libc_free_0_0(v117);
      v18 = *(_QWORD *)v13;
      v19 = *(_QWORD *)v13 + v23;
LABEL_19:
      v20 = *(unsigned int *)(v13 + 8);
      v21 = v18 + 8 * v20;
      if ( v21 != v19 + 8 )
      {
        memmove((void *)v19, (const void *)(v19 + 8), v21 - (v19 + 8));
        LODWORD(v20) = *(_DWORD *)(v13 + 8);
      }
      v22 = v20 - 1;
      v11 = v12;
      *(_DWORD *)(v13 + 8) = v22;
    }
    while ( v12 < v22 );
LABEL_34:
    v8 = (__int64)v14;
    v5 = v13;
    if ( !v22 )
      goto LABEL_56;
    if ( !v8 )
    {
      sub_DA5930(v13, v102, v104);
      v6 = *(unsigned int *)(v13 + 8);
      goto LABEL_54;
    }
    v26 = *(_QWORD *)(v8 + 32);
    v27 = *(_DWORD *)(v26 + 32);
    v28 = v26 + 24;
    if ( v108 > 1u )
    {
      if ( v110 )
      {
        if ( v27 <= 0x40 )
          v59 = *(_QWORD *)(v26 + 24) == 0;
        else
          v59 = (unsigned int)sub_C444A0(v28) == v27;
      }
      else
      {
        v55 = *(_QWORD *)(v26 + 24);
        v56 = v27 - 1;
        if ( v27 > 0x40 )
        {
          if ( (*(_QWORD *)(v55 + 8LL * (v56 >> 6)) & (1LL << v56)) != 0 && (unsigned int)sub_C44590(v28) == v56 )
            return v8;
          sub_DA5930(v5, v102, v104);
          v32 = *(_QWORD *)(v8 + 32);
          v33 = v32 + 24;
          goto LABEL_75;
        }
        v59 = 1LL << v56 == v55;
      }
LABEL_84:
      if ( v59 )
        return v8;
      sub_DA5930(v5, v102, v104);
      v32 = *(_QWORD *)(v8 + 32);
      v33 = v32 + 24;
      if ( v108 <= 1u )
      {
        if ( !v110 )
          goto LABEL_42;
        v82 = *(_DWORD *)(v32 + 32);
        if ( v82 <= 0x40 )
          v83 = *(_QWORD *)(v32 + 24) == 0;
        else
          v83 = v82 == (unsigned int)sub_C444A0(v33);
LABEL_123:
        if ( !v83 )
          goto LABEL_45;
        goto LABEL_90;
      }
      if ( v110 )
      {
        v60 = *(_DWORD *)(v32 + 32);
        if ( v60 )
        {
          if ( v60 <= 0x40 )
          {
            if ( *(_QWORD *)(v32 + 24) != 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v60) )
              goto LABEL_45;
          }
          else if ( v60 != (unsigned int)sub_C445E0(v33) )
          {
            goto LABEL_45;
          }
        }
        goto LABEL_90;
      }
LABEL_75:
      v57 = *(_DWORD *)(v32 + 32);
      v58 = v57 - 1;
      if ( v57 <= 0x40 )
      {
        if ( *(_QWORD *)(v32 + 24) != (1LL << v58) - 1 )
          goto LABEL_45;
        goto LABEL_90;
      }
      if ( (*(_QWORD *)(*(_QWORD *)(v32 + 24) + 8LL * (v58 >> 6)) & (1LL << v58)) == 0
        && (unsigned int)sub_C445E0(v33) == v58 )
      {
        goto LABEL_90;
      }
LABEL_45:
      v36 = *(unsigned int *)(v5 + 8);
      v37 = *(unsigned int *)(v5 + 12);
      v38 = v36 + 1;
      v39 = 8 * v36;
      if ( !v39 )
      {
        if ( v37 < v38 )
        {
          sub_C8D5F0(v5, (const void *)(v5 + 16), v38, 8u, a5, v31);
          v39 = 8LL * *(unsigned int *)(v5 + 8);
        }
        *(_QWORD *)(*(_QWORD *)v5 + v39) = v8;
        v6 = (unsigned int)(*(_DWORD *)(v5 + 8) + 1);
        *(_DWORD *)(v5 + 8) = v6;
        goto LABEL_54;
      }
      if ( v37 < v38 )
      {
        sub_C8D5F0(v5, (const void *)(v5 + 16), v38, 8u, a5, v31);
        v43 = *(__int64 **)v5;
        v44 = *(unsigned int *)(v5 + 8);
        v45 = 8 * v44;
        v40 = *(char **)v5;
        v41 = (char *)(*(_QWORD *)v5 + 8 * v44 - 8);
        v42 = (_QWORD *)(8 * v44 + *(_QWORD *)v5);
        if ( !v42 )
          goto LABEL_49;
      }
      else
      {
        v40 = *(char **)v5;
        v41 = (char *)(*(_QWORD *)v5 + v39 - 8);
        v42 = (_QWORD *)(*(_QWORD *)v5 + v39);
      }
      *v42 = *(_QWORD *)v41;
      v43 = *(__int64 **)v5;
      v44 = *(unsigned int *)(v5 + 8);
      v45 = 8 * v44;
      v41 = (char *)(*(_QWORD *)v5 + 8 * v44 - 8);
LABEL_49:
      if ( v40 != v41 )
      {
        memmove((char *)v43 + v45 - (v41 - v40), v40, v41 - v40);
        LODWORD(v44) = *(_DWORD *)(v5 + 8);
      }
      *(_DWORD *)(v5 + 8) = v44 + 1;
      *(_QWORD *)v40 = v8;
      goto LABEL_90;
    }
    if ( v110 )
    {
      if ( !v27 )
        return v8;
      if ( v27 <= 0x40 )
        v59 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v27) == *(_QWORD *)(v26 + 24);
      else
        v59 = (unsigned int)sub_C445E0(v28) == v27;
      goto LABEL_84;
    }
    v29 = *(_QWORD *)(v26 + 24);
    v30 = v27 - 1;
    if ( v27 <= 0x40 )
    {
      if ( (1LL << v30) - 1 == v29 )
        return v8;
    }
    else if ( (*(_QWORD *)(v29 + 8LL * (v30 >> 6)) & (1LL << v30)) == 0 && (unsigned int)sub_C445E0(v28) == v30 )
    {
      return v8;
    }
    sub_DA5930(v5, v102, v104);
    v32 = *(_QWORD *)(v8 + 32);
    v33 = v32 + 24;
LABEL_42:
    v34 = *(_DWORD *)(v32 + 32);
    v35 = v34 - 1;
    if ( v34 <= 0x40 )
    {
      v83 = *(_QWORD *)(v32 + 24) == 1LL << v35;
      goto LABEL_123;
    }
    if ( (*(_QWORD *)(*(_QWORD *)(v32 + 24) + 8LL * (v35 >> 6)) & (1LL << v35)) == 0
      || (unsigned int)sub_C44590(v33) != v35 )
    {
      goto LABEL_45;
    }
LABEL_90:
    v6 = *(unsigned int *)(v5 + 8);
LABEL_54:
    if ( (_DWORD)v6 == 1 )
    {
      v8 = **(_QWORD **)v5;
LABEL_56:
      if ( v8 )
        return v8;
      v6 = *(unsigned int *)(v5 + 8);
    }
LABEL_58:
    v8 = (__int64)sub_D96EA0((__int64)v115, v112, *(unsigned __int64 **)v5, v6, a5);
    if ( v8 )
      return v8;
    v46 = *(unsigned __int64 **)v5;
    v47 = *(_DWORD *)(v5 + 8);
    v7 = *(unsigned __int64 **)v5;
    if ( v47 )
    {
      v48 = v47;
      v49 = 0;
      while ( 1 )
      {
        v50 = &v46[v49];
        v51 = *(_WORD *)(v46[v49] + 24);
        if ( v10 <= v51 )
          break;
        if ( v47 == ++v49 )
          goto LABEL_91;
      }
      if ( v112 == v51 )
      {
        v52 = v46[v49];
        while ( 1 )
        {
          v53 = &v46[v48];
          if ( v53 != v50 + 1 )
          {
            memmove(v50, v50 + 1, (char *)v53 - (char *)(v50 + 1));
            v47 = *(_DWORD *)(v5 + 8);
            v46 = *(unsigned __int64 **)v5;
          }
          v54 = v47 - 1;
          *(_DWORD *)(v5 + 8) = v54;
          sub_D932D0(
            v5,
            (char *)&v46[v54],
            *(char **)(v52 + 32),
            (char *)(*(_QWORD *)(v52 + 32) + 8LL * *(_QWORD *)(v52 + 40)));
          v46 = *(unsigned __int64 **)v5;
          v50 = (unsigned __int64 *)(*(_QWORD *)v5 + v49 * 8);
          v52 = *v50;
          if ( *(_WORD *)(*v50 + 24) != v51 )
            break;
          v48 = *(unsigned int *)(v5 + 8);
          v47 = *(_DWORD *)(v5 + 8);
        }
        a2 = v112;
        a1 = v115;
        a3 = v5;
        continue;
      }
    }
    break;
  }
LABEL_91:
  v103 = v110 == 0 ? 39 : 35;
  v107 = v110 == 0 ? 41 : 37;
  if ( v108 <= 1u )
  {
    v103 = v110 == 0 ? 41 : 37;
    v107 = v110 == 0 ? 39 : 35;
  }
  v61 = v47 - 1;
  if ( !v61 )
    return *v7;
  v62 = v100;
  v63 = 0;
  while ( 2 )
  {
    v70 = v7[v63];
    v71 = v7[v63 + 1];
    v72 = 8LL * v63;
    if ( v70 != v71 )
    {
      v62 = v107 | v62 & 0xFFFFFF0000000000LL;
      if ( !(unsigned __int8)sub_DCD020(v115, v62, v70, v71) )
      {
        v101 = v103 | v101 & 0xFFFFFF0000000000LL;
        if ( (unsigned __int8)sub_DCD020(
                                v115,
                                v101,
                                *(_QWORD *)(*(_QWORD *)v5 + 8LL * v63),
                                *(_QWORD *)(*(_QWORD *)v5 + 8LL * (v63 + 1))) )
        {
          v84 = *(unsigned __int64 **)v5;
          v85 = v72 + 8;
          v86 = (const void *)(*(_QWORD *)v5 + v72 + 8);
          v87 = (void *)(*(_QWORD *)v5 + v72);
          v88 = 8LL * *(unsigned int *)(v5 + 8);
          v89 = (const void *)(*(_QWORD *)v5 + v88);
          v90 = v88 - v85;
          if ( v86 != v89 )
          {
            v113 = v90;
            memmove(v87, v86, v90);
            v84 = *(unsigned __int64 **)v5;
            v90 = v113;
          }
          --v61;
          *(_DWORD *)(v5 + 8) = ((_BYTE *)v87 + v90 - (_BYTE *)v84) >> 3;
        }
        else
        {
          ++v63;
        }
        goto LABEL_99;
      }
      v7 = *(unsigned __int64 **)v5;
    }
    v64 = v72 + 16;
    v65 = &v7[(unsigned __int64)v72 / 8 + 1];
    v66 = &v7[(unsigned __int64)v72 / 8 + 2];
    v67 = *(unsigned int *)(v5 + 8);
    v68 = &v7[v67];
    v69 = v67 * 8 - v64;
    if ( v66 != v68 )
    {
      v65 = (unsigned __int64 *)memmove(v65, v66, v69);
      v7 = *(unsigned __int64 **)v5;
    }
    --v61;
    *(_DWORD *)(v5 + 8) = ((char *)v65 + v69 - (char *)v7) >> 3;
LABEL_99:
    if ( v61 != v63 )
    {
      v7 = *(unsigned __int64 **)v5;
      continue;
    }
    break;
  }
  if ( *(_DWORD *)(v5 + 8) == 1 )
  {
LABEL_2:
    v7 = *(unsigned __int64 **)v5;
    return *v7;
  }
  v118 = 0x2000000000LL;
  v117 = v119;
  sub_9C8C60((__int64)&v117, v112);
  v75 = (unsigned int)v118;
  if ( *(_QWORD *)v5 + 8LL * *(unsigned int *)(v5 + 8) != *(_QWORD *)v5 )
  {
    v76 = *(_QWORD *)v5 + 8LL * *(unsigned int *)(v5 + 8);
    v77 = *(unsigned __int64 **)v5;
    do
    {
      v78 = *v77;
      if ( v75 + 1 > (unsigned __int64)HIDWORD(v118) )
      {
        sub_C8D5F0((__int64)&v117, v119, v75 + 1, 4u, v73, v74);
        v75 = (unsigned int)v118;
      }
      *(_DWORD *)&v117[4 * v75] = v78;
      v79 = HIDWORD(v78);
      LODWORD(v118) = v118 + 1;
      v80 = (unsigned int)v118;
      if ( (unsigned __int64)(unsigned int)v118 + 1 > HIDWORD(v118) )
      {
        sub_C8D5F0((__int64)&v117, v119, (unsigned int)v118 + 1LL, 4u, v73, v74);
        v80 = (unsigned int)v118;
      }
      ++v77;
      *(_DWORD *)&v117[4 * v80] = v79;
      v75 = (unsigned int)(v118 + 1);
      LODWORD(v118) = v118 + 1;
    }
    while ( (unsigned __int64 *)v76 != v77 );
  }
  v81 = &v117;
  v116 = 0;
  v8 = (__int64)sub_C65B40((__int64)(v115 + 129), (__int64)&v117, (__int64 *)&v116, (__int64)off_49DEA80);
  if ( !v8 )
  {
    v91 = v115[133];
    v92 = 8LL * *(unsigned int *)(v5 + 8);
    v93 = (unsigned __int64 *)(v115 + 133);
    v115[143] += v92;
    v94 = (void *)((v91 + 7) & 0xFFFFFFFFFFFFFFF8LL);
    if ( v115[134] >= (unsigned __int64)v94 + v92 && v91 )
      v115[133] = (__int64)v94 + v92;
    else
      v94 = (void *)sub_9D1E70((__int64)v93, v92, v92, 3);
    v95 = 8LL * *(unsigned int *)(v5 + 8);
    if ( v95 )
      v94 = memmove(v94, *(const void **)v5, v95);
    v109 = (__int64 *)v94;
    v96 = sub_C65D30((__int64)&v117, v93);
    v97 = *(unsigned int *)(v5 + 8);
    v114 = v96;
    v111 = v98;
    v8 = sub_A777F0(0x30u, (__int64 *)v93);
    if ( v8 )
    {
      v99 = sub_D95470(v109, v97);
      *(_QWORD *)v8 = 0;
      *(_WORD *)(v8 + 26) = v99;
      *(_QWORD *)(v8 + 8) = v114;
      *(_QWORD *)(v8 + 40) = v97;
      *(_QWORD *)(v8 + 16) = v111;
      *(_WORD *)(v8 + 28) = 6;
      *(_WORD *)(v8 + 24) = v112;
      *(_QWORD *)(v8 + 32) = v109;
    }
    sub_C657C0(v115 + 129, (__int64 *)v8, v116, (__int64)off_49DEA80);
    v81 = (_QWORD *)v8;
    sub_DAEE00((__int64)v115, v8, *(__int64 **)v5, *(unsigned int *)(v5 + 8));
  }
  if ( v117 != v119 )
    _libc_free(v117, v81);
  return v8;
}
