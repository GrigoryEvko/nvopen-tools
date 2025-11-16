// Function: sub_386C880
// Address: 0x386c880
//
__int64 __fastcall sub_386C880(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __m128 a4,
        double a5,
        double a6,
        double a7,
        double a8,
        double a9,
        double a10,
        __m128 a11)
{
  __int64 v12; // r12
  __int64 v14; // rax
  __int64 v15; // rsi
  unsigned int v16; // ecx
  __int64 *v17; // rdx
  __int64 v18; // r8
  __int64 v19; // r15
  int v21; // edx
  __int64 v22; // rax
  _QWORD *v23; // r14
  __int64 v24; // rax
  __int64 *v25; // rax
  __int64 v26; // r15
  _QWORD *v27; // rax
  __int64 v28; // rax
  int v29; // eax
  unsigned __int64 *v30; // rdi
  __int64 v31; // rax
  bool v32; // zf
  unsigned __int64 v33; // rax
  double v34; // xmm4_8
  double v35; // xmm5_8
  _QWORD *v36; // rax
  __int64 v37; // rax
  __int64 v38; // rbx
  _QWORD *v39; // r12
  __int64 v40; // rax
  __int64 v41; // rax
  int v42; // r9d
  __int64 *v43; // rcx
  unsigned int v44; // esi
  _QWORD *v45; // rdx
  __int64 v46; // rdx
  __int64 v47; // rcx
  char v48; // r10
  __int64 v49; // rdi
  __int64 v50; // rax
  _QWORD *v51; // rcx
  _QWORD *v52; // rsi
  __int64 v53; // rdx
  _QWORD *v54; // rax
  _BYTE *v55; // r9
  __int64 v56; // rax
  __int64 v57; // rsi
  unsigned __int64 v58; // rdx
  __int64 v59; // rdx
  __int64 v60; // rdx
  __int64 v61; // rbx
  _QWORD *v62; // rax
  __int64 v63; // r15
  __int64 v64; // rbx
  _QWORD *v65; // rax
  int v66; // edx
  __int64 v67; // r14
  __int64 v68; // r12
  unsigned int v69; // ebx
  __int64 v70; // rsi
  _QWORD *v71; // rdx
  __int64 v72; // rdi
  unsigned __int64 v73; // rsi
  __int64 v74; // rsi
  __int64 v75; // rdx
  __int64 v76; // rsi
  __int64 v77; // r9
  unsigned int v78; // edx
  __int64 v79; // rax
  __int64 v80; // rax
  __int64 v81; // rdx
  int v82; // esi
  unsigned int v83; // eax
  unsigned __int64 *v84; // rdi
  __int64 v85; // rax
  unsigned int v86; // esi
  __int64 v87; // [rsp+0h] [rbp-180h]
  __int64 v88; // [rsp+8h] [rbp-178h]
  int v89; // [rsp+8h] [rbp-178h]
  __int64 v90; // [rsp+18h] [rbp-168h]
  __int64 v91; // [rsp+18h] [rbp-168h]
  __int64 v92; // [rsp+20h] [rbp-160h]
  unsigned __int64 v93; // [rsp+28h] [rbp-158h]
  __int64 v94; // [rsp+28h] [rbp-158h]
  __int64 v95; // [rsp+28h] [rbp-158h]
  __int64 v96; // [rsp+28h] [rbp-158h]
  __int64 v97; // [rsp+30h] [rbp-150h] BYREF
  _QWORD v98[2]; // [rsp+38h] [rbp-148h] BYREF
  __int64 v99; // [rsp+48h] [rbp-138h]
  __int64 v100; // [rsp+50h] [rbp-130h] BYREF
  __int64 v101; // [rsp+58h] [rbp-128h] BYREF
  __int64 v102; // [rsp+60h] [rbp-120h]
  __int64 v103; // [rsp+68h] [rbp-118h]
  _BYTE *v104; // [rsp+80h] [rbp-100h] BYREF
  __int64 v105; // [rsp+88h] [rbp-F8h]
  _BYTE v106[240]; // [rsp+90h] [rbp-F0h] BYREF

  v12 = a3;
  v14 = *(unsigned int *)(a3 + 24);
  if ( (_DWORD)v14 )
  {
    v15 = *(_QWORD *)(a3 + 8);
    v16 = (v14 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v17 = (__int64 *)(v15 + 32LL * v16);
    v18 = *v17;
    if ( a2 == *v17 )
    {
LABEL_3:
      if ( v17 != (__int64 *)(v15 + 32 * v14) )
        return v17[3];
    }
    else
    {
      v21 = 1;
      while ( v18 != -8 )
      {
        v42 = v21 + 1;
        v16 = (v14 - 1) & (v21 + v16);
        v17 = (__int64 *)(v15 + 32LL * v16);
        v18 = *v17;
        if ( a2 == *v17 )
          goto LABEL_3;
        v21 = v42;
      }
    }
  }
  v22 = sub_157F0B0(a2);
  v23 = (_QWORD *)v22;
  if ( v22 )
  {
    v24 = sub_386E810(a1, v22, v12);
    v100 = a2;
    v101 = 6;
    v19 = v24;
    v102 = 0;
    if ( v24 )
    {
      v103 = v24;
      if ( v24 == -8 || v24 == -16 )
        goto LABEL_12;
      goto LABEL_60;
    }
    goto LABEL_66;
  }
  v92 = a1 + 408;
  if ( sub_183E920(a1 + 408, a2) )
  {
    v41 = sub_1426C80(*(_QWORD *)a1, a2);
    v100 = a2;
    v101 = 6;
    v19 = v41;
    v102 = 0;
    if ( v41 )
    {
      v103 = v41;
      if ( v41 == -16 || v41 == -8 )
      {
LABEL_12:
        sub_386BE80((__int64)&v104, v12, &v100, &v101);
        if ( v103 != 0 && v103 != -8 && v103 != -16 )
          sub_1649B30(&v101);
        return v19;
      }
LABEL_60:
      sub_164C220((__int64)&v101);
      goto LABEL_12;
    }
LABEL_66:
    v103 = 0;
    goto LABEL_12;
  }
  v25 = *(__int64 **)(a1 + 416);
  if ( *(__int64 **)(a1 + 424) != v25 )
    goto LABEL_17;
  v43 = &v25[*(unsigned int *)(a1 + 436)];
  v44 = *(_DWORD *)(a1 + 436);
  if ( v25 == v43 )
  {
LABEL_160:
    if ( v44 >= *(_DWORD *)(a1 + 432) )
    {
LABEL_17:
      sub_16CCBA0(v92, a2);
      goto LABEL_18;
    }
    *(_DWORD *)(a1 + 436) = v44 + 1;
    *v43 = a2;
    ++*(_QWORD *)(a1 + 408);
  }
  else
  {
    while ( a2 != *v25 )
    {
      if ( *v25 == -2 )
        v23 = v25;
      if ( v43 == ++v25 )
      {
        if ( !v23 )
          goto LABEL_160;
        *v23 = a2;
        --*(_DWORD *)(a1 + 440);
        ++*(_QWORD *)(a1 + 408);
        break;
      }
    }
  }
LABEL_18:
  v26 = *(_QWORD *)(a2 + 8);
  v104 = v106;
  v105 = 0x800000000LL;
  if ( v26 )
  {
    while ( 1 )
    {
      v27 = sub_1648700(v26);
      if ( (unsigned __int8)(*((_BYTE *)v27 + 16) - 25) <= 9u )
        break;
      v26 = *(_QWORD *)(v26 + 8);
      if ( !v26 )
        goto LABEL_37;
    }
LABEL_22:
    v28 = sub_386E810(a1, v27[5], v12);
    v100 = 6;
    v101 = 0;
    if ( v28 )
    {
      v102 = v28;
      if ( v28 != -16 && v28 != -8 )
        sub_164C220((__int64)&v100);
    }
    else
    {
      v102 = 0;
    }
    v29 = v105;
    if ( (unsigned int)v105 >= HIDWORD(v105) )
    {
      sub_386B6A0((__int64)&v104, 0);
      v29 = v105;
    }
    v30 = (unsigned __int64 *)&v104[24 * v29];
    if ( v30 )
    {
      *v30 = 6;
      v30[1] = 0;
      v31 = v102;
      v32 = v102 == -8;
      v30[2] = v102;
      if ( v31 != 0 && !v32 && v31 != -16 )
        sub_1649AC0(v30, v100 & 0xFFFFFFFFFFFFFFF8LL);
      v29 = v105;
    }
    LODWORD(v105) = v29 + 1;
    if ( v102 != -8 && v102 != 0 && v102 != -16 )
      sub_1649B30(&v100);
    while ( 1 )
    {
      v26 = *(_QWORD *)(v26 + 8);
      if ( !v26 )
        break;
      v27 = sub_1648700(v26);
      if ( (unsigned __int8)(*((_BYTE *)v27 + 16) - 25) <= 9u )
        goto LABEL_22;
    }
  }
LABEL_37:
  v33 = sub_14228C0(*(_QWORD *)a1, a2);
  if ( v33 )
  {
    v93 = v33;
    v19 = sub_386C720((__int64 *)a1, v33, (__int64 *)&v104, a4, a5, a6, a7, v34, v35, a10, a11);
    if ( v93 != v19 )
      goto LABEL_39;
  }
  else
  {
    v19 = sub_386C720((__int64 *)a1, 0, (__int64 *)&v104, a4, a5, a6, a7, v34, v35, a10, a11);
    if ( v19 )
      goto LABEL_39;
    v19 = sub_1426C80(*(_QWORD *)a1, a2);
  }
  if ( (*(_DWORD *)(v19 + 20) & 0xFFFFFFF) != 0 )
  {
    v48 = *(_BYTE *)(v19 + 23);
    v49 = (__int64)v104;
    v50 = 3LL * (*(_DWORD *)(v19 + 20) & 0xFFFFFFF);
    if ( (v48 & 0x40) != 0 )
    {
      v51 = *(_QWORD **)(v19 - 8);
      v52 = &v51[v50];
    }
    else
    {
      v52 = (_QWORD *)v19;
      v51 = (_QWORD *)(v19 - v50 * 8);
    }
    v53 = (__int64)v104;
    v54 = v51;
    while ( *v54 == *(_QWORD *)(v53 + 16) )
    {
      v54 += 3;
      v53 += 24;
      if ( v54 == v52 )
        goto LABEL_39;
    }
    if ( 24LL * (unsigned int)v105 )
    {
      v55 = &v104[24 * (unsigned int)v105];
      do
      {
        v56 = *(_QWORD *)(v49 + 16);
        if ( *v51 )
        {
          v57 = v51[1];
          v58 = v51[2] & 0xFFFFFFFFFFFFFFFCLL;
          *(_QWORD *)v58 = v57;
          if ( v57 )
            *(_QWORD *)(v57 + 16) = *(_QWORD *)(v57 + 16) & 3LL | v58;
        }
        *v51 = v56;
        if ( v56 )
        {
          v59 = *(_QWORD *)(v56 + 8);
          v51[1] = v59;
          if ( v59 )
            *(_QWORD *)(v59 + 16) = (unsigned __int64)(v51 + 1) | *(_QWORD *)(v59 + 16) & 3LL;
          v51[2] = (v56 + 8) | v51[2] & 3LL;
          *(_QWORD *)(v56 + 8) = v51;
        }
        v49 += 24;
        v51 += 3;
      }
      while ( v55 != (_BYTE *)v49 );
      v48 = *(_BYTE *)(v19 + 23);
    }
    if ( (v48 & 0x40) != 0 )
      v60 = *(_QWORD *)(v19 - 8);
    else
      v60 = v19 - 24LL * (*(_DWORD *)(v19 + 20) & 0xFFFFFFF);
    v88 = v60 + 24LL * *(unsigned int *)(v19 + 76) + 8;
    if ( *(_QWORD *)(a2 + 8) )
    {
      v90 = a2;
      v61 = *(_QWORD *)(a2 + 8);
      while ( 1 )
      {
        v62 = sub_1648700(v61);
        if ( (unsigned __int8)(*((_BYTE *)v62 + 16) - 25) <= 9u )
          break;
        v61 = *(_QWORD *)(v61 + 8);
        if ( !v61 )
          goto LABEL_112;
      }
      v94 = v19;
      v63 = v88;
LABEL_109:
      v63 += 8;
      *(_QWORD *)(v63 - 8) = v62[5];
      while ( 1 )
      {
        v61 = *(_QWORD *)(v61 + 8);
        if ( !v61 )
          break;
        v62 = sub_1648700(v61);
        if ( (unsigned __int8)(*((_BYTE *)v62 + 16) - 25) <= 9u )
          goto LABEL_109;
      }
      v19 = v94;
LABEL_112:
      a2 = v90;
    }
  }
  else
  {
    if ( *(_QWORD *)(a2 + 8) )
    {
      v89 = *(_DWORD *)(v19 + 20);
      v95 = a2;
      v64 = *(_QWORD *)(a2 + 8);
      while ( 1 )
      {
        v65 = sub_1648700(v64);
        if ( (unsigned __int8)(*((_BYTE *)v65 + 16) - 25) <= 9u )
          break;
        v64 = *(_QWORD *)(v64 + 8);
        if ( !v64 )
        {
          a2 = v95;
          goto LABEL_143;
        }
      }
      v66 = v89;
      v67 = v12;
      v68 = v64;
      v87 = v95;
      v69 = 0;
LABEL_137:
      v77 = v65[5];
      v78 = v66 & 0xFFFFFFF;
      v79 = 3LL * v69++;
      v80 = *(_QWORD *)&v104[8 * v79 + 16];
      if ( v78 == *(_DWORD *)(v19 + 76) )
      {
        v91 = v80;
        v96 = v77;
        v86 = v78 + (v78 >> 1);
        if ( v86 < 2 )
          v86 = 2;
        *(_DWORD *)(v19 + 76) = v86;
        sub_16488D0(v19, v86, 1);
        v80 = v91;
        v77 = v96;
        v78 = *(_DWORD *)(v19 + 20) & 0xFFFFFFF;
      }
      v81 = (v78 + 1) & 0xFFFFFFF;
      v82 = v81 | *(_DWORD *)(v19 + 20) & 0xF0000000;
      *(_DWORD *)(v19 + 20) = v82;
      if ( (v82 & 0x40000000) != 0 )
        v70 = *(_QWORD *)(v19 - 8);
      else
        v70 = v19 - 24 * v81;
      v71 = (_QWORD *)(v70 + 24LL * (unsigned int)(v81 - 1));
      if ( *v71 )
      {
        v72 = v71[1];
        v73 = v71[2] & 0xFFFFFFFFFFFFFFFCLL;
        *(_QWORD *)v73 = v72;
        if ( v72 )
          *(_QWORD *)(v72 + 16) = *(_QWORD *)(v72 + 16) & 3LL | v73;
      }
      *v71 = v80;
      if ( v80 )
      {
        v74 = *(_QWORD *)(v80 + 8);
        v71[1] = v74;
        if ( v74 )
          *(_QWORD *)(v74 + 16) = (unsigned __int64)(v71 + 1) | *(_QWORD *)(v74 + 16) & 3LL;
        v71[2] = (v80 + 8) | v71[2] & 3LL;
        *(_QWORD *)(v80 + 8) = v71;
      }
      v75 = *(_DWORD *)(v19 + 20) & 0xFFFFFFF;
      if ( (*(_BYTE *)(v19 + 23) & 0x40) != 0 )
        v76 = *(_QWORD *)(v19 - 8);
      else
        v76 = v19 - 24 * v75;
      *(_QWORD *)(v76 + 8LL * (unsigned int)(v75 - 1) + 24LL * *(unsigned int *)(v19 + 76) + 8) = v77;
      while ( 1 )
      {
        v68 = *(_QWORD *)(v68 + 8);
        if ( !v68 )
          break;
        v65 = sub_1648700(v68);
        if ( (unsigned __int8)(*((_BYTE *)v65 + 16) - 25) <= 9u )
        {
          v66 = *(_DWORD *)(v19 + 20);
          goto LABEL_137;
        }
      }
      v12 = v67;
      a2 = v87;
    }
LABEL_143:
    v100 = 4;
    v101 = 0;
    v102 = v19;
    if ( v19 != -16 && v19 != -8 )
      sub_164C220((__int64)&v100);
    v83 = *(_DWORD *)(a1 + 16);
    if ( v83 >= *(_DWORD *)(a1 + 20) )
    {
      sub_1BC1F40(a1 + 8, 0);
      v83 = *(_DWORD *)(a1 + 16);
    }
    v84 = (unsigned __int64 *)(*(_QWORD *)(a1 + 8) + 24LL * v83);
    if ( v84 )
    {
      *v84 = 4;
      v84[1] = 0;
      v85 = v102;
      v32 = v102 == -8;
      v84[2] = v102;
      if ( v85 != 0 && !v32 && v85 != -16 )
        sub_1649AC0(v84, v100 & 0xFFFFFFFFFFFFFFF8LL);
      v83 = *(_DWORD *)(a1 + 16);
    }
    *(_DWORD *)(a1 + 16) = v83 + 1;
    if ( v102 != 0 && v102 != -8 && v102 != -16 )
      sub_1649B30(&v100);
  }
LABEL_39:
  v36 = *(_QWORD **)(a1 + 416);
  if ( *(_QWORD **)(a1 + 424) == v36 )
  {
    v45 = &v36[*(unsigned int *)(a1 + 436)];
    if ( v36 == v45 )
    {
LABEL_113:
      v36 = v45;
    }
    else
    {
      while ( a2 != *v36 )
      {
        if ( v45 == ++v36 )
          goto LABEL_113;
      }
    }
    goto LABEL_78;
  }
  v36 = sub_16CC9F0(v92, a2);
  if ( a2 == *v36 )
  {
    v46 = *(_QWORD *)(a1 + 424);
    if ( v46 == *(_QWORD *)(a1 + 416) )
      v47 = *(unsigned int *)(a1 + 436);
    else
      v47 = *(unsigned int *)(a1 + 432);
    v45 = (_QWORD *)(v46 + 8 * v47);
LABEL_78:
    if ( v36 != v45 )
    {
      *v36 = -2;
      ++*(_DWORD *)(a1 + 440);
    }
    goto LABEL_42;
  }
  v37 = *(_QWORD *)(a1 + 424);
  if ( v37 == *(_QWORD *)(a1 + 416) )
  {
    v36 = (_QWORD *)(v37 + 8LL * *(unsigned int *)(a1 + 436));
    v45 = v36;
    goto LABEL_78;
  }
LABEL_42:
  v97 = a2;
  v98[0] = 6;
  v98[1] = 0;
  if ( v19 )
  {
    v99 = v19;
    if ( v19 != -8 && v19 != -16 )
      sub_164C220((__int64)v98);
  }
  else
  {
    v99 = 0;
  }
  sub_386BE80((__int64)&v100, v12, &v97, v98);
  if ( v99 != 0 && v99 != -8 && v99 != -16 )
    sub_1649B30(v98);
  v38 = (__int64)v104;
  v39 = &v104[24 * (unsigned int)v105];
  if ( v104 != (_BYTE *)v39 )
  {
    do
    {
      v40 = *(v39 - 1);
      v39 -= 3;
      if ( v40 != 0 && v40 != -8 && v40 != -16 )
        sub_1649B30(v39);
    }
    while ( (_QWORD *)v38 != v39 );
    v39 = v104;
  }
  if ( v39 != (_QWORD *)v106 )
    _libc_free((unsigned __int64)v39);
  return v19;
}
