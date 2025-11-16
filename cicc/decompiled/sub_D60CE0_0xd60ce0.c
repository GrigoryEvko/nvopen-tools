// Function: sub_D60CE0
// Address: 0xd60ce0
//
__int64 __fastcall sub_D60CE0(
        __int64 a1,
        __int64 *a2,
        __int64 a3,
        __int64 a4,
        _QWORD *a5,
        __int64 a6,
        __int64 a7,
        unsigned int *a8)
{
  char v11; // di
  __int64 v12; // rcx
  int v13; // eax
  unsigned int v14; // edx
  __int64 *v15; // r15
  __int64 v16; // rsi
  __int64 v17; // rax
  unsigned int v18; // eax
  unsigned int v19; // eax
  __int64 v21; // rdx
  __int64 v22; // rax
  unsigned __int8 *v23; // r13
  unsigned int v24; // eax
  int v25; // eax
  __int64 v26; // rdi
  const void **v27; // rdx
  const void *v28; // rax
  char v29; // al
  unsigned __int8 *v30; // r15
  unsigned __int64 v31; // rax
  __int64 v32; // rcx
  __int64 v33; // rsi
  unsigned int v34; // eax
  const void **v35; // rdi
  __int64 v36; // r15
  int v37; // r9d
  __int64 *v38; // rax
  _QWORD *v39; // rdi
  __int64 v40; // rdi
  const void *v41; // rax
  char v42; // al
  __int64 v43; // r15
  __int64 v44; // rbx
  unsigned __int8 *v45; // rax
  unsigned __int16 v46; // ax
  __int64 v47; // rax
  unsigned int v48; // edx
  __int64 v49; // rax
  __int64 v50; // r15
  __int64 v51; // rcx
  __int64 v52; // rcx
  unsigned __int64 v53; // r8
  __int64 v54; // r9
  __int64 v55; // rax
  unsigned __int64 v56; // rsi
  int v57; // ecx
  const void **v58; // rdx
  const void **v59; // rsi
  const void **v60; // rax
  int v61; // edx
  int v62; // edx
  const void **v63; // rax
  __int64 v64; // rsi
  const void **v65; // rbx
  const void **v66; // r12
  const void *v67; // rdi
  const void **v68; // rbx
  const void **v69; // rbx
  unsigned int v70; // eax
  unsigned int v71; // eax
  unsigned int v72; // eax
  __int64 v73; // [rsp-8h] [rbp-188h]
  const void **v74; // [rsp+30h] [rbp-150h]
  __int64 v75; // [rsp+38h] [rbp-148h]
  const void **v77; // [rsp+40h] [rbp-140h]
  const void *v79; // [rsp+50h] [rbp-130h] BYREF
  unsigned int v80; // [rsp+58h] [rbp-128h]
  const void *v81; // [rsp+60h] [rbp-120h] BYREF
  unsigned int v82; // [rsp+68h] [rbp-118h]
  const void *v83; // [rsp+70h] [rbp-110h] BYREF
  unsigned int v84; // [rsp+78h] [rbp-108h]
  __int64 v85; // [rsp+80h] [rbp-100h] BYREF
  unsigned int v86; // [rsp+88h] [rbp-F8h]
  const void **v87; // [rsp+90h] [rbp-F0h] BYREF
  unsigned int v88; // [rsp+98h] [rbp-E8h]
  const void *v89; // [rsp+A0h] [rbp-E0h] BYREF
  unsigned int v90; // [rsp+A8h] [rbp-D8h]
  unsigned __int64 v91; // [rsp+B0h] [rbp-D0h] BYREF
  unsigned int v92; // [rsp+B8h] [rbp-C8h]
  const void *v93; // [rsp+C0h] [rbp-C0h]
  unsigned int v94; // [rsp+C8h] [rbp-B8h]
  const void **v95; // [rsp+D0h] [rbp-B0h] BYREF
  unsigned int v96; // [rsp+D8h] [rbp-A8h]
  const void *v97; // [rsp+E0h] [rbp-A0h] BYREF
  unsigned int v98; // [rsp+E8h] [rbp-98h]
  const void *v99; // [rsp+F0h] [rbp-90h] BYREF
  __int64 v100; // [rsp+F8h] [rbp-88h]
  __int64 v101; // [rsp+100h] [rbp-80h] BYREF
  __int64 v102; // [rsp+108h] [rbp-78h]
  __int64 v103; // [rsp+110h] [rbp-70h]
  __int64 v104; // [rsp+118h] [rbp-68h]
  const void **v105; // [rsp+120h] [rbp-60h] BYREF
  __int64 v106; // [rsp+128h] [rbp-58h]
  const void *v107; // [rsp+130h] [rbp-50h] BYREF
  __int64 v108; // [rsp+138h] [rbp-48h]
  __int64 v109; // [rsp+140h] [rbp-40h]
  __int64 v110; // [rsp+148h] [rbp-38h]

  v11 = *(_BYTE *)(a7 + 8) & 1;
  if ( v11 )
  {
    v12 = a7 + 16;
    v13 = 7;
  }
  else
  {
    v21 = *(unsigned int *)(a7 + 24);
    v12 = *(_QWORD *)(a7 + 16);
    if ( !(_DWORD)v21 )
      goto LABEL_41;
    v13 = v21 - 1;
  }
  v14 = v13 & (((unsigned int)a4 >> 9) ^ ((unsigned int)a4 >> 4));
  v15 = (__int64 *)(v12 + 40LL * v14);
  v16 = *v15;
  if ( a4 != *v15 )
  {
    v37 = 1;
    while ( v16 != -4096 )
    {
      v14 = v13 & (v37 + v14);
      v15 = (__int64 *)(v12 + 40LL * v14);
      v16 = *v15;
      if ( a4 == *v15 )
        goto LABEL_4;
      ++v37;
    }
    if ( v11 )
    {
      v36 = 320;
      goto LABEL_42;
    }
    v21 = *(unsigned int *)(a7 + 24);
LABEL_41:
    v36 = 40 * v21;
LABEL_42:
    v15 = (__int64 *)(v12 + v36);
  }
LABEL_4:
  v17 = 320;
  if ( !v11 )
    v17 = 40LL * *(unsigned int *)(a7 + 24);
  if ( v15 != (__int64 *)(v12 + v17) )
  {
    v18 = *((_DWORD *)v15 + 4);
    *(_DWORD *)(a1 + 8) = v18;
    if ( v18 > 0x40 )
    {
      sub_C43780(a1, (const void **)v15 + 1);
      v34 = *((_DWORD *)v15 + 8);
      *(_DWORD *)(a1 + 24) = v34;
      if ( v34 <= 0x40 )
        goto LABEL_9;
    }
    else
    {
      *(_QWORD *)a1 = v15[1];
      v19 = *((_DWORD *)v15 + 8);
      *(_DWORD *)(a1 + 24) = v19;
      if ( v19 <= 0x40 )
      {
LABEL_9:
        *(_QWORD *)(a1 + 16) = v15[3];
        return a1;
      }
    }
    sub_C43780(a1 + 16, (const void **)v15 + 3);
    return a1;
  }
  v75 = a1;
  while ( 1 )
  {
    v22 = 0;
    if ( a5 )
      v22 = (__int64)(a5 - 3);
    v23 = (unsigned __int8 *)v22;
    if ( sub_B46AA0(v22) )
      goto LABEL_23;
    v24 = *a8 + 1;
    *a8 = v24;
    if ( v24 > 0x80 )
      goto LABEL_30;
    if ( !(unsigned __int8)sub_B46490((__int64)v23) )
      goto LABEL_23;
    v25 = *v23;
    if ( (_BYTE)v25 != 62 )
      break;
    v26 = a2[3];
    v27 = *(const void ***)(a3 - 32);
    v28 = (const void *)*((_QWORD *)v23 - 4);
    v106 = -1;
    v107 = 0;
    v105 = v27;
    v108 = 0;
    v109 = 0;
    v110 = 0;
    v99 = v28;
    v100 = -1;
    v101 = 0;
    v102 = 0;
    v103 = 0;
    v104 = 0;
    v29 = sub_CF4E00(v26, (__int64)&v99, (__int64)&v105);
    if ( v29 )
    {
      v30 = v23;
      a1 = v75;
      if ( v29 == 3 && *(_BYTE *)(*(_QWORD *)(*((_QWORD *)v30 - 8) + 8LL) + 8LL) == 14 )
      {
        sub_D62600(&v105, a2);
        sub_D60530(v75, a4, a7, (__int64)&v105);
        if ( (unsigned int)v108 > 0x40 && v107 )
          j_j___libc_free_0_0(v107);
        if ( (unsigned int)v106 > 0x40 )
        {
          v35 = v105;
          if ( v105 )
            goto LABEL_81;
        }
        return a1;
      }
LABEL_22:
      sub_D60910(a1, a4, a7);
      return a1;
    }
LABEL_23:
    if ( *(_QWORD **)(a4 + 56) == a5 )
    {
      v50 = *(_QWORD *)(a4 + 16);
      a1 = v75;
      v105 = &v107;
      v106 = 0x100000000LL;
      if ( !v50 )
        goto LABEL_102;
      while ( 1 )
      {
        v51 = *(_QWORD *)(v50 + 24);
        if ( (unsigned __int8)(*(_BYTE *)v51 - 30) <= 0xAu )
          break;
        v50 = *(_QWORD *)(v50 + 8);
        if ( !v50 )
          goto LABEL_102;
      }
LABEL_88:
      v52 = *(_QWORD *)(v51 + 40);
      v53 = *(_QWORD *)(v52 + 48) & 0xFFFFFFFFFFFFFFF8LL;
      if ( v53 == v52 + 48 )
        goto LABEL_118;
      if ( !v53 )
        BUG();
      if ( (unsigned int)*(unsigned __int8 *)(v53 - 24) - 30 > 0xA )
LABEL_118:
        LODWORD(v53) = 0;
      sub_D60CE0((unsigned int)&v99, (_DWORD)a2, a3, v52, v53, 0, a7, (__int64)a8);
      v55 = (unsigned int)v106;
      v56 = (unsigned int)v106 + 1LL;
      v57 = v106;
      if ( v56 > HIDWORD(v106) )
      {
        if ( v105 > &v99 || (v74 = v105, &v99 >= &v105[4 * (unsigned int)v106]) )
        {
          sub_D5F640((__int64)&v105, v56, HIDWORD(v106), (__int64)v105, v73, v54);
          v55 = (unsigned int)v106;
          v58 = v105;
          v59 = &v99;
          v57 = v106;
        }
        else
        {
          sub_D5F640((__int64)&v105, v56, HIDWORD(v106), (__int64)v105, v73, v54);
          v58 = v105;
          v55 = (unsigned int)v106;
          v57 = v106;
          v59 = (const void **)((char *)v105 + (char *)&v99 - (char *)v74);
        }
      }
      else
      {
        v58 = v105;
        v59 = &v99;
      }
      v60 = &v58[4 * v55];
      if ( v60 )
      {
        v61 = *((_DWORD *)v59 + 2);
        *((_DWORD *)v59 + 2) = 0;
        *((_DWORD *)v60 + 2) = v61;
        *v60 = *v59;
        v62 = *((_DWORD *)v59 + 6);
        *((_DWORD *)v59 + 6) = 0;
        *((_DWORD *)v60 + 6) = v62;
        v60[2] = v59[2];
        v57 = v106;
      }
      LODWORD(v106) = v57 + 1;
      if ( (unsigned int)v102 > 0x40 && v101 )
        j_j___libc_free_0_0(v101);
      if ( (unsigned int)v100 > 0x40 && v99 )
        j_j___libc_free_0_0(v99);
      v63 = &v105[4 * (unsigned int)v106 - 4];
      if ( *((_DWORD *)v63 + 2) <= 1u || *((_DWORD *)v63 + 6) <= 1u )
        goto LABEL_102;
      while ( 1 )
      {
        v50 = *(_QWORD *)(v50 + 8);
        if ( !v50 )
          break;
        v51 = *(_QWORD *)(v50 + 24);
        if ( (unsigned __int8)(*(_BYTE *)v51 - 30) <= 0xAu )
          goto LABEL_88;
      }
      v68 = v105;
      if ( (_DWORD)v106 )
      {
        v80 = *((_DWORD *)v105 + 2);
        if ( v80 > 0x40 )
          sub_C43780((__int64)&v79, v105);
        else
          v79 = *v105;
        v82 = *((_DWORD *)v68 + 6);
        if ( v82 > 0x40 )
          sub_C43780((__int64)&v81, v68 + 2);
        else
          v81 = v68[2];
        v69 = v105 + 4;
        v77 = &v105[4 * (unsigned int)v106];
        if ( v105 + 4 != v77 )
        {
          do
          {
            v88 = *((_DWORD *)v69 + 2);
            if ( v88 > 0x40 )
              sub_C43780((__int64)&v87, v69);
            else
              v87 = (const void **)*v69;
            v90 = *((_DWORD *)v69 + 6);
            if ( v90 > 0x40 )
              sub_C43780((__int64)&v89, v69 + 2);
            else
              v89 = v69[2];
            v84 = v80;
            if ( v80 > 0x40 )
              sub_C43780((__int64)&v83, &v79);
            else
              v83 = v79;
            v86 = v82;
            if ( v82 > 0x40 )
              sub_C43780((__int64)&v85, &v81);
            else
              v85 = (__int64)v81;
            v96 = v88;
            if ( v88 > 0x40 )
              sub_C43780((__int64)&v95, (const void **)&v87);
            else
              v95 = v87;
            v98 = v90;
            if ( v90 > 0x40 )
              sub_C43780((__int64)&v97, &v89);
            else
              v97 = v89;
            LODWORD(v100) = v84;
            if ( v84 > 0x40 )
              sub_C43780((__int64)&v99, &v83);
            else
              v99 = v83;
            LODWORD(v102) = v86;
            if ( v86 > 0x40 )
              sub_C43780((__int64)&v101, (const void **)&v85);
            else
              v101 = v85;
            sub_D5E640((__int64)&v91, (__int64)a2, (__int64)&v99, (__int64)&v95);
            if ( (unsigned int)v102 > 0x40 && v101 )
              j_j___libc_free_0_0(v101);
            if ( (unsigned int)v100 > 0x40 && v99 )
              j_j___libc_free_0_0(v99);
            if ( v98 > 0x40 && v97 )
              j_j___libc_free_0_0(v97);
            if ( v96 > 0x40 && v95 )
              j_j___libc_free_0_0(v95);
            if ( v80 > 0x40 && v79 )
              j_j___libc_free_0_0(v79);
            v79 = (const void *)v91;
            v70 = v92;
            v92 = 0;
            v80 = v70;
            if ( v82 > 0x40 && v81 )
            {
              j_j___libc_free_0_0(v81);
              v81 = v93;
              v82 = v94;
              if ( v92 > 0x40 && v91 )
                j_j___libc_free_0_0(v91);
            }
            else
            {
              v81 = v93;
              v82 = v94;
            }
            if ( v86 > 0x40 && v85 )
              j_j___libc_free_0_0(v85);
            if ( v84 > 0x40 && v83 )
              j_j___libc_free_0_0(v83);
            if ( v90 > 0x40 && v89 )
              j_j___libc_free_0_0(v89);
            if ( v88 > 0x40 && v87 )
              j_j___libc_free_0_0(v87);
            v69 += 4;
          }
          while ( v77 != v69 );
          a1 = v75;
        }
        v71 = v80;
        v64 = a4;
        v80 = 0;
        LODWORD(v100) = v71;
        v99 = v79;
        v72 = v82;
        v82 = 0;
        LODWORD(v102) = v72;
        v101 = (__int64)v81;
        sub_D60530(a1, a4, a7, (__int64)&v99);
        sub_969240(&v101);
        sub_969240((__int64 *)&v99);
        sub_969240((__int64 *)&v81);
        sub_969240((__int64 *)&v79);
      }
      else
      {
LABEL_102:
        v64 = a4;
        sub_D60910(v75, a4, a7);
      }
      v65 = v105;
      v66 = &v105[4 * (unsigned int)v106];
      if ( v105 != v66 )
      {
        do
        {
          v66 -= 4;
          if ( *((_DWORD *)v66 + 6) > 0x40u )
          {
            v67 = v66[2];
            if ( v67 )
              j_j___libc_free_0_0(v67);
          }
          if ( *((_DWORD *)v66 + 2) > 0x40u && *v66 )
            j_j___libc_free_0_0(*v66);
        }
        while ( v65 != v66 );
        v66 = v105;
      }
      if ( v66 != &v107 )
        _libc_free(v66, v64);
      return a1;
    }
    a5 = (_QWORD *)(*a5 & 0xFFFFFFFFFFFFFFF8LL);
  }
  v31 = (unsigned int)(v25 - 34);
  if ( (unsigned __int8)v31 > 0x33u
    || (v32 = 0x8000000000041LL, !_bittest64(&v32, v31))
    || (v33 = *((_QWORD *)v23 - 4)) == 0
    || *(_BYTE *)v33
    || *((_QWORD *)v23 + 10) != *(_QWORD *)(v33 + 24)
    || (v38 = (__int64 *)a2[1]) == 0
    || !sub_981210(*v38, v33, (unsigned int *)&v87)
    || (v39 = (_QWORD *)a2[1], (v39[((unsigned __int64)(unsigned int)v87 >> 6) + 1] & (1LL << (char)v87)) != 0)
    || (((int)*(unsigned __int8 *)(*v39 + ((unsigned int)v87 >> 2)) >> (2 * ((unsigned __int8)v87 & 3))) & 3) == 0
    || (_DWORD)v87 != 385 )
  {
LABEL_30:
    a1 = v75;
    sub_D60910(v75, a4, a7);
    return a1;
  }
  v40 = a2[3];
  v41 = *(const void **)&v23[-32 * (*((_DWORD *)v23 + 1) & 0x7FFFFFF)];
  v105 = *(const void ***)(a3 - 32);
  v106 = -1;
  v107 = 0;
  v108 = 0;
  v109 = 0;
  v110 = 0;
  v99 = v41;
  v100 = -1;
  v101 = 0;
  v102 = 0;
  v103 = 0;
  v104 = 0;
  v42 = sub_CF4E00(v40, (__int64)&v99, (__int64)&v105);
  if ( !v42 )
    goto LABEL_23;
  v43 = (__int64)v23;
  a1 = v75;
  if ( v42 != 3 )
    goto LABEL_22;
  v44 = *a2;
  v45 = (unsigned __int8 *)sub_AD64C0(*(_QWORD *)(v43 + 8), 0, 0);
  v46 = sub_9A1D50(0x20u, v43, v45, a3, v44);
  if ( !HIBYTE(v46) )
    goto LABEL_22;
  if ( !(_BYTE)v46 )
    goto LABEL_22;
  v47 = *(_QWORD *)(v43 + 32 * (2LL - (*(_DWORD *)(v43 + 4) & 0x7FFFFFF)));
  if ( *(_BYTE *)v47 != 17 )
    goto LABEL_22;
  v92 = *(_DWORD *)(v47 + 32);
  v48 = v92;
  if ( v92 <= 0x40 )
  {
    v91 = *(_QWORD *)(v47 + 24);
    v49 = 1LL << ((unsigned __int8)v92 - 1);
    goto LABEL_59;
  }
  sub_C43780((__int64)&v91, (const void **)(v47 + 24));
  v48 = v92;
  v49 = 1LL << ((unsigned __int8)v92 - 1);
  if ( v92 <= 0x40 )
  {
LABEL_59:
    if ( (v91 & v49) == 0 )
    {
      v96 = v48;
      v95 = 0;
      goto LABEL_61;
    }
    goto LABEL_189;
  }
  if ( (*(_QWORD *)(v91 + 8LL * ((v92 - 1) >> 6)) & v49) != 0 )
  {
LABEL_189:
    sub_D60910(v75, a4, a7);
    goto LABEL_79;
  }
  v96 = v92;
  sub_C43690((__int64)&v95, 0, 0);
LABEL_61:
  LODWORD(v100) = v92;
  if ( v92 > 0x40 )
    sub_C43780((__int64)&v99, (const void **)&v91);
  else
    v99 = (const void *)v91;
  LODWORD(v106) = v96;
  if ( v96 > 0x40 )
    sub_C43780((__int64)&v105, (const void **)&v95);
  else
    v105 = v95;
  LODWORD(v108) = v100;
  if ( (unsigned int)v100 > 0x40 )
    sub_C43780((__int64)&v107, &v99);
  else
    v107 = v99;
  sub_D60530(v75, a4, a7, (__int64)&v105);
  if ( (unsigned int)v108 > 0x40 && v107 )
    j_j___libc_free_0_0(v107);
  if ( (unsigned int)v106 > 0x40 && v105 )
    j_j___libc_free_0_0(v105);
  if ( (unsigned int)v100 > 0x40 && v99 )
    j_j___libc_free_0_0(v99);
  if ( v96 > 0x40 && v95 )
    j_j___libc_free_0_0(v95);
LABEL_79:
  if ( v92 > 0x40 )
  {
    v35 = (const void **)v91;
    if ( v91 )
LABEL_81:
      j_j___libc_free_0_0(v35);
  }
  return a1;
}
