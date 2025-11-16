// Function: sub_1D81290
// Address: 0x1d81290
//
__int64 __fastcall sub_1D81290(
        _QWORD *a1,
        __int64 a2,
        __m128 a3,
        __m128i a4,
        __m128i a5,
        __m128i a6,
        double a7,
        double a8,
        double a9,
        __m128 a10)
{
  __int64 v10; // rbx
  __int64 v11; // r13
  __int64 v13; // rax
  int v14; // r8d
  int v15; // r9d
  __int64 v16; // r14
  unsigned __int64 v17; // rax
  int v18; // r8d
  int v19; // r9d
  __int64 v20; // rdx
  __int64 v21; // rdx
  __int64 v22; // rax
  int v23; // eax
  unsigned int v24; // ebx
  unsigned __int64 v25; // r12
  __int64 v26; // r14
  __int64 *v27; // r13
  __int64 v28; // r12
  __int64 *v29; // rbx
  __int64 *v30; // r14
  unsigned int v31; // esi
  _QWORD *v32; // rax
  __int64 *v33; // rdx
  __int64 v34; // rax
  __int64 v35; // rdx
  __int64 v36; // rax
  __int64 v37; // rax
  int v38; // r8d
  int v39; // r9d
  __int64 v40; // r13
  unsigned __int64 v41; // rbx
  unsigned __int64 v42; // r14
  __int64 v43; // rax
  _QWORD *v44; // r12
  _QWORD *v45; // rdi
  __int64 *v46; // rdx
  __int64 v47; // rax
  __int64 v48; // rdx
  __int64 v49; // rax
  __int64 *v50; // rdx
  __int64 v51; // rcx
  __int64 v52; // rax
  __int64 v53; // rdx
  __int64 v54; // rax
  double v55; // xmm4_8
  double v56; // xmm5_8
  __int64 *v57; // rdi
  unsigned int v58; // r12d
  __int64 v60; // r12
  __int64 v61; // rax
  __int64 v62; // r14
  unsigned __int64 v63; // rbx
  __int64 v64; // rdx
  __int64 *v65; // rax
  __int64 v66; // rsi
  unsigned __int64 v67; // rdx
  __int64 v68; // rdx
  __int64 v69; // rdx
  __int64 v70; // rsi
  _QWORD *v71; // r13
  __int64 v72; // rsi
  __int64 v73; // r12
  _QWORD *v74; // rdi
  __int64 v75; // rdx
  __int64 v76; // rcx
  __int64 v77; // r8
  __int64 v78; // r9
  __int64 v79; // r13
  int v80; // eax
  __int64 v81; // rax
  int v82; // edx
  __int64 v83; // r12
  _QWORD *v84; // rax
  __int64 v85; // rbx
  _QWORD *v86; // rdi
  __int64 v87; // rax
  __int64 v88; // r12
  __int64 v89; // r13
  _QWORD *v90; // rax
  __int64 v91; // rbx
  _QWORD *v92; // rdi
  __int64 *v93; // rax
  __int64 *i; // rdx
  __int64 *v95; // rax
  __int64 v96; // rax
  size_t v97; // rdx
  __int64 v98; // r14
  __int64 v99; // r12
  __int64 v100; // r13
  __int64 v101; // rax
  __int64 **v102; // [rsp+8h] [rbp-1C8h]
  _QWORD *v103; // [rsp+10h] [rbp-1C0h]
  unsigned int v104; // [rsp+18h] [rbp-1B8h]
  __int64 v105; // [rsp+18h] [rbp-1B8h]
  __int64 *v107; // [rsp+30h] [rbp-1A0h]
  __int64 v108; // [rsp+30h] [rbp-1A0h]
  char *s; // [rsp+40h] [rbp-190h]
  _QWORD *sa; // [rsp+40h] [rbp-190h]
  unsigned int v111; // [rsp+48h] [rbp-188h]
  __int64 v112; // [rsp+48h] [rbp-188h]
  unsigned __int64 v113; // [rsp+48h] [rbp-188h]
  __int64 v114; // [rsp+48h] [rbp-188h]
  __int64 *v115; // [rsp+48h] [rbp-188h]
  __int64 v116; // [rsp+58h] [rbp-178h] BYREF
  __int64 v117[2]; // [rsp+60h] [rbp-170h] BYREF
  __int64 v118; // [rsp+70h] [rbp-160h]
  __int64 *v119; // [rsp+80h] [rbp-150h] BYREF
  __int64 v120; // [rsp+88h] [rbp-148h]
  _BYTE v121[128]; // [rsp+90h] [rbp-140h] BYREF
  __int64 *v122; // [rsp+110h] [rbp-C0h] BYREF
  __int64 v123; // [rsp+118h] [rbp-B8h]
  _BYTE v124[176]; // [rsp+120h] [rbp-B0h] BYREF

  v10 = a2 + 72;
  v11 = *(_QWORD *)(a2 + 80);
  v119 = (__int64 *)v121;
  v120 = 0x1000000000LL;
  v122 = (__int64 *)v124;
  v123 = 0x1000000000LL;
  if ( v11 == a2 + 72 )
  {
    v58 = 0;
    goto LABEL_57;
  }
  do
  {
    while ( 1 )
    {
      v16 = v11 - 24;
      if ( !v11 )
        v16 = 0;
      v17 = sub_157EBA0(v16);
      if ( *(_BYTE *)(v17 + 16) == 30 )
      {
        v20 = (unsigned int)v120;
        if ( (unsigned int)v120 >= HIDWORD(v120) )
        {
          v113 = v17;
          sub_16CD150((__int64)&v119, v121, 0, 8, v18, v19);
          v20 = (unsigned int)v120;
          v17 = v113;
        }
        v119[v20] = v17;
        LODWORD(v120) = v120 + 1;
      }
      v13 = sub_157F7B0(v16);
      if ( v13 )
      {
        if ( (*(_BYTE *)(v13 + 18) & 1) != 0 )
          break;
      }
      v11 = *(_QWORD *)(v11 + 8);
      if ( v10 == v11 )
        goto LABEL_15;
    }
    v21 = (unsigned int)v123;
    if ( (unsigned int)v123 >= HIDWORD(v123) )
    {
      v114 = v13;
      sub_16CD150((__int64)&v122, v124, 0, 8, v14, v15);
      v21 = (unsigned int)v123;
      v13 = v114;
    }
    v122[v21] = v13;
    LODWORD(v123) = v123 + 1;
    v11 = *(_QWORD *)(v11 + 8);
  }
  while ( v10 != v11 );
LABEL_15:
  if ( !(_DWORD)v120 )
    goto LABEL_54;
  v22 = sub_15E38F0(a2);
  v23 = sub_14DD7D0(v22);
  if ( v23 <= 10 )
  {
    if ( v23 <= 6 )
      goto LABEL_18;
LABEL_54:
    v57 = v122;
    v58 = 0;
    goto LABEL_55;
  }
  if ( v23 == 12 )
    goto LABEL_54;
LABEL_18:
  v103 = (_QWORD *)sub_15E0530(a2);
  v104 = v120;
  v24 = (unsigned int)(v120 + 63) >> 6;
  v25 = 8LL * v24;
  s = (char *)malloc(v25);
  if ( s )
  {
    v26 = v104;
  }
  else if ( v25 || (v101 = malloc(1u)) == 0 )
  {
    sub_16BD1C0("Allocation failed", 1u);
    v26 = (unsigned int)v120;
  }
  else
  {
    s = (char *)v101;
    v26 = v104;
  }
  if ( v24 )
    memset(s, 0, v25);
  v27 = v119;
  v111 = 0;
  v107 = &v119[v26];
  if ( v119 != v107 )
  {
    do
    {
      v28 = *v27;
      v29 = &v122[(unsigned int)v123];
      v30 = v122;
      if ( v122 != v29 )
      {
        while ( !(unsigned __int8)sub_137E580(*v30, v28, a1[21], 0) )
        {
          if ( v29 == ++v30 )
            goto LABEL_28;
        }
        *(_QWORD *)&s[8 * (v111 >> 6)] |= 1LL << v111;
      }
LABEL_28:
      ++v111;
      ++v27;
    }
    while ( v107 != v27 );
  }
  v31 = v104 >> 6;
  if ( v104 >> 6 )
  {
    v32 = s;
    while ( *v32 == -1 )
    {
      if ( &s[8 * v31] == (char *)++v32 )
        goto LABEL_60;
    }
  }
  else
  {
LABEL_60:
    if ( (v104 & 0x3F) == 0 || *(_QWORD *)&s[8 * v31] == (1LL << (v104 & 0x3F)) - 1 )
    {
      v41 = (unsigned int)v120;
      goto LABEL_63;
    }
  }
  v33 = (__int64 *)a1[1];
  v34 = *v33;
  v35 = v33[1];
  if ( v34 == v35 )
LABEL_122:
    BUG();
  while ( *(_UNKNOWN **)v34 != &unk_4F9D3C0 )
  {
    v34 += 16;
    if ( v35 == v34 )
      goto LABEL_122;
  }
  v36 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v34 + 8) + 104LL))(*(_QWORD *)(v34 + 8), &unk_4F9D3C0);
  v102 = (__int64 **)sub_14A4050(v36, a2);
  v37 = sub_15E0530(a2);
  v40 = (unsigned int)v120;
  v108 = v37;
  if ( !(_DWORD)v120 )
  {
    _libc_free((unsigned __int64)s);
    goto LABEL_94;
  }
  v41 = 0;
  v42 = 0;
  do
  {
    v43 = *(_QWORD *)&s[8 * ((unsigned int)v42 >> 6)];
    v44 = (_QWORD *)v119[v42];
    if ( _bittest64(&v43, v42) )
    {
      v119[v41++] = (__int64)v44;
    }
    else
    {
      v112 = v44[5];
      v45 = sub_1648A60(56, 0);
      if ( v45 )
        sub_15F82A0((__int64)v45, v108, (__int64)v44);
      sub_15F20C0(v44);
      v46 = (__int64 *)a1[1];
      v118 = 0;
      v117[0] = 0x1000000000001LL;
      v47 = *v46;
      v48 = v46[1];
      if ( v47 == v48 )
LABEL_124:
        BUG();
      while ( *(_UNKNOWN **)v47 != &unk_4FB9E3C )
      {
        v47 += 16;
        if ( v48 == v47 )
          goto LABEL_124;
      }
      v49 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v47 + 8) + 104LL))(
              *(_QWORD *)(v47 + 8),
              &unk_4FB9E3C);
      v50 = (__int64 *)a1[1];
      v51 = v49;
      v52 = *v50;
      v53 = v50[1];
      if ( v52 == v53 )
LABEL_123:
        BUG();
      while ( *(_UNKNOWN **)v52 != &unk_4FBA0D1 )
      {
        v52 += 16;
        if ( v53 == v52 )
          goto LABEL_123;
      }
      v105 = v51;
      v54 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v52 + 8) + 104LL))(
              *(_QWORD *)(v52 + 8),
              &unk_4FBA0D1);
      sub_1B5E140(v112, v102, *(_QWORD **)(v54 + 160), v105, (unsigned int *)v117, 0, a3, a4, a5, a6, v55, v56, a9, a10);
    }
    ++v42;
  }
  while ( v40 != v42 );
  v87 = (unsigned int)v120;
  if ( v41 >= (unsigned int)v120 )
  {
    if ( v41 > (unsigned int)v120 )
    {
      if ( v41 > HIDWORD(v120) )
      {
        sub_16CD150((__int64)&v119, v121, v41, 8, v38, v39);
        v87 = (unsigned int)v120;
      }
      v93 = &v119[v87];
      for ( i = &v119[v41]; i != v93; ++v93 )
      {
        if ( v93 )
          *v93 = 0;
      }
      LODWORD(v120) = v41;
      _libc_free((unsigned __int64)s);
      goto LABEL_64;
    }
  }
  else
  {
    LODWORD(v120) = v41;
  }
LABEL_63:
  _libc_free((unsigned __int64)s);
  if ( v41 )
  {
LABEL_64:
    if ( !a1[20] )
    {
      v117[0] = sub_16471D0(v103, 0);
      v95 = (__int64 *)sub_1643270(v103);
      v96 = sub_1644EA0(v95, v117, 1, 0);
      v97 = 0;
      v98 = v96;
      v99 = *(_QWORD *)(a1[22] + 76856LL);
      v100 = *(_QWORD *)(a2 + 40);
      if ( v99 )
        v97 = strlen(*(const char **)(a1[22] + 76856LL));
      a1[20] = sub_1632190(v100, v99, v97, v98);
    }
    if ( v41 == 1 )
    {
      v88 = *(_QWORD *)(*v119 + 40);
      v116 = sub_1D81020((_QWORD *)*v119);
      v89 = a1[20];
      LOWORD(v118) = 257;
      v90 = sub_1648A60(72, 2u);
      v91 = (__int64)v90;
      if ( v90 )
      {
        sub_15F1F50(
          (__int64)v90,
          **(_QWORD **)(*(_QWORD *)(*(_QWORD *)v89 + 24LL) + 16LL),
          54,
          (__int64)(v90 - 6),
          2,
          v88);
        *(_QWORD *)(v91 + 56) = 0;
        sub_15F5B40(v91, *(_QWORD *)(*(_QWORD *)v89 + 24LL), v89, &v116, 1, (__int64)v117, 0, 0);
      }
      *(_WORD *)(v91 + 18) = *(_WORD *)(v91 + 18) & 0x8000
                           | *(_WORD *)(v91 + 18) & 3
                           | (4 * *(_WORD *)(a1[22] + 81028LL));
      v92 = sub_1648A60(56, 0);
      if ( v92 )
        sub_15F82E0((__int64)v92, (__int64)v103, v88);
    }
    else
    {
      v117[0] = (__int64)"unwind_resume";
      LOWORD(v118) = 259;
      sa = (_QWORD *)sub_22077B0(64);
      if ( sa )
        sub_157FB60(sa, (__int64)v103, (__int64)v117, a2, 0);
      v117[0] = (__int64)"exn.obj";
      LOWORD(v118) = 259;
      v60 = sub_16471D0(v103, 0);
      v61 = sub_1648B60(64);
      v62 = v61;
      if ( v61 )
      {
        sub_15F1F50(v61, v60, 53, 0, 0, (__int64)sa);
        *(_DWORD *)(v62 + 56) = v41;
        sub_164B780(v62, v117);
        sub_1648880(v62, *(_DWORD *)(v62 + 56), 1);
      }
      v63 = (unsigned __int64)v119;
      v115 = &v119[(unsigned int)v120];
      if ( v119 != v115 )
      {
        do
        {
          v71 = *(_QWORD **)v63;
          v72 = 1;
          v73 = *(_QWORD *)(*(_QWORD *)v63 + 40LL);
          v74 = sub_1648A60(56, 1u);
          if ( v74 )
          {
            v72 = (__int64)sa;
            sub_15F8590((__int64)v74, (__int64)sa, v73);
          }
          v79 = sub_1D81020(v71);
          v80 = *(_DWORD *)(v62 + 20) & 0xFFFFFFF;
          if ( v80 == *(_DWORD *)(v62 + 56) )
          {
            sub_15F55D0(v62, v72, v75, v76, v77, v78);
            v80 = *(_DWORD *)(v62 + 20) & 0xFFFFFFF;
          }
          v81 = (v80 + 1) & 0xFFFFFFF;
          v82 = v81 | *(_DWORD *)(v62 + 20) & 0xF0000000;
          *(_DWORD *)(v62 + 20) = v82;
          if ( (v82 & 0x40000000) != 0 )
            v64 = *(_QWORD *)(v62 - 8);
          else
            v64 = v62 - 24 * v81;
          v65 = (__int64 *)(v64 + 24LL * (unsigned int)(v81 - 1));
          if ( *v65 )
          {
            v66 = v65[1];
            v67 = v65[2] & 0xFFFFFFFFFFFFFFFCLL;
            *(_QWORD *)v67 = v66;
            if ( v66 )
              *(_QWORD *)(v66 + 16) = *(_QWORD *)(v66 + 16) & 3LL | v67;
          }
          *v65 = v79;
          if ( v79 )
          {
            v68 = *(_QWORD *)(v79 + 8);
            v65[1] = v68;
            if ( v68 )
              *(_QWORD *)(v68 + 16) = (unsigned __int64)(v65 + 1) | *(_QWORD *)(v68 + 16) & 3LL;
            v65[2] = (v79 + 8) | v65[2] & 3;
            *(_QWORD *)(v79 + 8) = v65;
          }
          v69 = *(_DWORD *)(v62 + 20) & 0xFFFFFFF;
          if ( (*(_BYTE *)(v62 + 23) & 0x40) != 0 )
            v70 = *(_QWORD *)(v62 - 8);
          else
            v70 = v62 - 24 * v69;
          v63 += 8LL;
          *(_QWORD *)(v70 + 8LL * (unsigned int)(v69 - 1) + 24LL * *(unsigned int *)(v62 + 56) + 8) = v73;
        }
        while ( v115 != (__int64 *)v63 );
      }
      v116 = v62;
      LOWORD(v118) = 257;
      v83 = a1[20];
      v84 = sub_1648A60(72, 2u);
      v85 = (__int64)v84;
      if ( v84 )
      {
        sub_15F1F50(
          (__int64)v84,
          **(_QWORD **)(*(_QWORD *)(*(_QWORD *)v83 + 24LL) + 16LL),
          54,
          (__int64)(v84 - 6),
          2,
          (__int64)sa);
        *(_QWORD *)(v85 + 56) = 0;
        sub_15F5B40(v85, *(_QWORD *)(*(_QWORD *)v83 + 24LL), v83, &v116, 1, (__int64)v117, 0, 0);
      }
      *(_WORD *)(v85 + 18) = *(_WORD *)(v85 + 18) & 0x8000
                           | *(_WORD *)(v85 + 18) & 3
                           | (4 * *(_WORD *)(a1[22] + 81028LL));
      v86 = sub_1648A60(56, 0);
      if ( v86 )
        sub_15F82E0((__int64)v86, (__int64)v103, (__int64)sa);
    }
  }
LABEL_94:
  v57 = v122;
  v58 = 1;
LABEL_55:
  if ( v57 != (__int64 *)v124 )
    _libc_free((unsigned __int64)v57);
LABEL_57:
  if ( v119 != (__int64 *)v121 )
    _libc_free((unsigned __int64)v119);
  return v58;
}
