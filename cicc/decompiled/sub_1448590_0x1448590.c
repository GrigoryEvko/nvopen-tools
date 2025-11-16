// Function: sub_1448590
// Address: 0x1448590
//
void __fastcall sub_1448590(_QWORD *a1, _QWORD *a2, char a3)
{
  char *v6; // rsi
  char *v7; // rdi
  __int64 v8; // rdx
  __int64 v9; // rsi
  __int64 v10; // r8
  unsigned __int64 v11; // r15
  __int64 v12; // rax
  __int64 v13; // rdi
  __int64 v14; // rdx
  __int64 v15; // rax
  char v16; // cl
  __int64 v17; // r8
  unsigned __int64 v18; // r15
  __int64 v19; // rax
  unsigned __int64 v20; // rdi
  unsigned __int64 v21; // rdx
  __int64 v22; // rax
  char v23; // cl
  unsigned __int64 v24; // rax
  __int64 v25; // rsi
  __int64 v26; // rdx
  unsigned __int64 v27; // rax
  char v28; // si
  char v29; // r8
  bool v30; // si
  unsigned __int64 v31; // rsi
  __int64 *v32; // r15
  __int64 *v33; // r13
  char *v34; // rsi
  _QWORD *v35; // r14
  char *v36; // rsi
  __int64 *v37; // r14
  __int64 *v38; // rbx
  __int64 v39; // r13
  char *v40; // r13
  char *v41; // rdx
  __int64 *v42; // rdi
  __int64 v43; // r8
  __int64 v44; // rsi
  unsigned __int64 v45; // r11
  unsigned __int64 v46; // rbx
  unsigned __int64 v47; // r10
  unsigned __int64 v48; // r14
  _QWORD *v49; // rax
  char *v50; // rsi
  _QWORD *v51; // rcx
  __int64 v52; // r12
  char *v53; // rax
  __int64 *v54; // r14
  __int64 v55; // rcx
  __int64 v56; // r13
  __int64 v57; // rax
  __int64 v58; // r12
  char *v59; // rbx
  __int64 v60; // r12
  char *v61; // r14
  bool v62; // cf
  unsigned __int64 v63; // r10
  __int64 v64; // rbx
  __int64 *v65; // rax
  __int64 *v66; // rsi
  __int64 *v67; // rcx
  __int64 *v68; // rcx
  __int64 *v69; // r13
  __int64 *v70; // rdx
  __int64 *v71; // r13
  __int64 *v72; // rbx
  __int64 v73; // r14
  char *v74; // rcx
  char *v75; // rax
  char *v76; // rbx
  __int64 v77; // rax
  __int64 *v78; // rax
  __int64 *v79; // r8
  __int64 *v80; // rcx
  __int64 v81; // rax
  __int64 v82; // r12
  char *v83; // [rsp+8h] [rbp-248h]
  char *v84; // [rsp+10h] [rbp-240h]
  char *v85; // [rsp+10h] [rbp-240h]
  __int64 *v86; // [rsp+10h] [rbp-240h]
  __int64 v87; // [rsp+18h] [rbp-238h]
  char *v88; // [rsp+18h] [rbp-238h]
  __int64 *v89; // [rsp+18h] [rbp-238h]
  char *v90; // [rsp+18h] [rbp-238h]
  char v91[8]; // [rsp+20h] [rbp-230h] BYREF
  __int64 v92; // [rsp+28h] [rbp-228h]
  unsigned __int64 v93; // [rsp+30h] [rbp-220h]
  char v94[64]; // [rsp+48h] [rbp-208h] BYREF
  __int64 v95; // [rsp+88h] [rbp-1C8h]
  __int64 v96; // [rsp+90h] [rbp-1C0h]
  unsigned __int64 v97; // [rsp+98h] [rbp-1B8h]
  char v98[8]; // [rsp+A0h] [rbp-1B0h] BYREF
  __int64 v99; // [rsp+A8h] [rbp-1A8h]
  unsigned __int64 v100; // [rsp+B0h] [rbp-1A0h]
  char v101[64]; // [rsp+C8h] [rbp-188h] BYREF
  unsigned __int64 v102; // [rsp+108h] [rbp-148h]
  unsigned __int64 i; // [rsp+110h] [rbp-140h]
  unsigned __int64 v104; // [rsp+118h] [rbp-138h]
  char *v105; // [rsp+120h] [rbp-130h] BYREF
  char *v106; // [rsp+128h] [rbp-128h]
  char *v107; // [rsp+130h] [rbp-120h]
  __int64 v108; // [rsp+188h] [rbp-C8h]
  __int64 v109; // [rsp+190h] [rbp-C0h]
  __int64 v110; // [rsp+198h] [rbp-B8h]
  char v111[8]; // [rsp+1A0h] [rbp-B0h] BYREF
  __int64 v112; // [rsp+1A8h] [rbp-A8h]
  unsigned __int64 v113; // [rsp+1B0h] [rbp-A0h]
  __int64 v114; // [rsp+208h] [rbp-48h]
  __int64 v115; // [rsp+210h] [rbp-40h]
  __int64 v116; // [rsp+218h] [rbp-38h]

  a2[1] = a1;
  v105 = (char *)a2;
  v6 = (char *)a1[6];
  if ( v6 != (char *)a1[7] )
  {
    if ( v6 )
    {
      *(_QWORD *)v6 = a2;
      a1[6] += 8LL;
      goto LABEL_4;
    }
    a1[6] = 8;
    v61 = (char *)a2;
    goto LABEL_103;
  }
  sub_1444F30(a1 + 5, v6, (__int64 *)&v105);
  v61 = v105;
  if ( v105 )
  {
LABEL_103:
    sub_1444060(v61);
    j_j___libc_free_0(v61, 112);
  }
LABEL_4:
  if ( !a3 )
    return;
  sub_14456E0(&v105, a1);
  v7 = v91;
  sub_16CCCB0(v91, v94, &v105);
  v9 = v109;
  v10 = v108;
  v95 = 0;
  v96 = 0;
  v97 = 0;
  v11 = v109 - v108;
  if ( v109 == v108 )
  {
    v13 = 0;
  }
  else
  {
    if ( v11 > 0x7FFFFFFFFFFFFFF8LL )
      goto LABEL_170;
    v12 = sub_22077B0(v109 - v108);
    v9 = v109;
    v10 = v108;
    v13 = v12;
  }
  v95 = v13;
  v96 = v13;
  v97 = v13 + v11;
  if ( v10 != v9 )
  {
    v14 = v13;
    v15 = v10;
    do
    {
      if ( v14 )
      {
        *(_QWORD *)v14 = *(_QWORD *)v15;
        v16 = *(_BYTE *)(v15 + 32);
        *(_BYTE *)(v14 + 32) = v16;
        if ( v16 )
        {
          *(__m128i *)(v14 + 8) = _mm_loadu_si128((const __m128i *)(v15 + 8));
          *(_QWORD *)(v14 + 24) = *(_QWORD *)(v15 + 24);
        }
      }
      v15 += 40;
      v14 += 40;
    }
    while ( v15 != v9 );
    v13 += 8 * ((unsigned __int64)(v15 - 40 - v10) >> 3) + 40;
  }
  v96 = v13;
  v7 = v98;
  sub_16CCCB0(v98, v101, v111);
  v9 = v115;
  v17 = v114;
  v102 = 0;
  i = 0;
  v104 = 0;
  v18 = v115 - v114;
  if ( v115 == v114 )
  {
    v20 = 0;
    goto LABEL_19;
  }
  if ( v18 > 0x7FFFFFFFFFFFFFF8LL )
LABEL_170:
    sub_4261EA(v7, v9, v8);
  v19 = sub_22077B0(v115 - v114);
  v9 = v115;
  v17 = v114;
  v20 = v19;
LABEL_19:
  v102 = v20;
  i = v20;
  v104 = v20 + v18;
  if ( v17 == v9 )
  {
    v24 = v20;
  }
  else
  {
    v21 = v20;
    v22 = v17;
    do
    {
      if ( v21 )
      {
        *(_QWORD *)v21 = *(_QWORD *)v22;
        v23 = *(_BYTE *)(v22 + 32);
        *(_BYTE *)(v21 + 32) = v23;
        if ( v23 )
        {
          *(__m128i *)(v21 + 8) = _mm_loadu_si128((const __m128i *)(v22 + 8));
          *(_QWORD *)(v21 + 24) = *(_QWORD *)(v22 + 24);
        }
      }
      v22 += 40;
      v21 += 40LL;
    }
    while ( v22 != v9 );
    v24 = v20 + 8 * ((unsigned __int64)(v22 - 40 - v17) >> 3) + 40;
  }
  for ( i = v24; ; v24 = i )
  {
    v26 = v95;
    if ( v96 - v95 != v24 - v20 )
      goto LABEL_27;
    if ( v95 == v96 )
      break;
    v27 = v20;
    while ( *(_QWORD *)v26 == *(_QWORD *)v27 )
    {
      v28 = *(_BYTE *)(v26 + 32);
      v29 = *(_BYTE *)(v27 + 32);
      if ( v28 && v29 )
      {
        if ( ((*(__int64 *)(v26 + 8) >> 1) & 3) != 0 )
          v30 = ((*(__int64 *)(v27 + 8) >> 1) & 3) == ((*(__int64 *)(v26 + 8) >> 1) & 3);
        else
          v30 = *(_DWORD *)(v26 + 24) == *(_DWORD *)(v27 + 24);
        if ( !v30 )
        {
          v25 = **(_QWORD **)(v96 - 40);
          if ( (v25 & 4) != 0 )
            goto LABEL_28;
          goto LABEL_39;
        }
      }
      else if ( v28 != v29 )
      {
        break;
      }
      v26 += 40;
      v27 += 40LL;
      if ( v96 == v26 )
        goto LABEL_43;
    }
LABEL_27:
    v25 = **(_QWORD **)(v96 - 40);
    if ( (v25 & 4) == 0 )
    {
LABEL_39:
      v31 = v25 & 0xFFFFFFFFFFFFFFF8LL;
      if ( (unsigned __int8)sub_1443560(a2, v31) )
        sub_1448350(a1[2], v31, (__int64)a2);
    }
LABEL_28:
    sub_1445340((__int64)v91);
    v20 = v102;
  }
LABEL_43:
  if ( v20 )
    j_j___libc_free_0(v20, v104 - v20);
  if ( v100 != v99 )
    _libc_free(v100);
  if ( v95 )
    j_j___libc_free_0(v95, v97 - v95);
  if ( v93 != v92 )
    _libc_free(v93);
  if ( v114 )
    j_j___libc_free_0(v114, v116 - v114);
  if ( v113 != v112 )
    _libc_free(v113);
  if ( v108 )
    j_j___libc_free_0(v108, v110 - v108);
  if ( v107 != v106 )
    _libc_free((unsigned __int64)v107);
  v32 = (__int64 *)a1[5];
  v33 = (__int64 *)a1[6];
  v105 = 0;
  v106 = 0;
  v107 = 0;
  if ( v33 != v32 )
  {
    while ( 1 )
    {
      v35 = (_QWORD *)*v32;
      if ( a2[4] )
      {
        if ( !(unsigned __int8)sub_1443560(a2, *v35 & 0xFFFFFFFFFFFFFFF8LL)
          || !(unsigned __int8)sub_1443560(a2, v35[4]) && v35[4] != a2[4] )
        {
LABEL_62:
          v34 = v106;
          if ( v106 == v107 )
          {
            sub_1444F30((__int64 *)&v105, v106, v32);
          }
          else
          {
            if ( v106 )
            {
              *(_QWORD *)v106 = *v32;
              *v32 = 0;
              v34 = v106;
            }
            v106 = v34 + 8;
          }
          goto LABEL_66;
        }
        v35 = (_QWORD *)*v32;
      }
      if ( a2 == v35 )
        goto LABEL_62;
      v35[1] = a2;
      v36 = (char *)a2[6];
      if ( v36 == (char *)a2[7] )
      {
        sub_1444F30(a2 + 5, v36, v32);
LABEL_66:
        if ( v33 == ++v32 )
          goto LABEL_73;
      }
      else
      {
        if ( v36 )
        {
          *(_QWORD *)v36 = *v32;
          *v32 = 0;
          v36 = (char *)a2[6];
        }
        ++v32;
        a2[6] = v36 + 8;
        if ( v33 == v32 )
        {
LABEL_73:
          v32 = (__int64 *)a1[5];
          v37 = (__int64 *)a1[6];
          if ( v32 != v37 )
          {
            v38 = (__int64 *)a1[5];
            do
            {
              v39 = *v38;
              if ( *v38 )
              {
                sub_1444060(*v38);
                j_j___libc_free_0(v39, 112);
              }
              ++v38;
            }
            while ( v38 != v37 );
            a1[6] = v32;
            v32 = (__int64 *)a1[5];
          }
          break;
        }
      }
    }
  }
  v40 = v106;
  v41 = v105;
  if ( v105 != v106 )
  {
    v42 = (__int64 *)a1[6];
    v43 = v106 - v105;
    v44 = (char *)v42 - (char *)v32;
    v45 = (v106 - v105) >> 3;
    v46 = v45;
    v47 = v42 - v32;
    v48 = v47;
    if ( a1[7] - (_QWORD)v42 >= (unsigned __int64)(v106 - v105) )
    {
      if ( v43 >= (unsigned __int64)v44 )
      {
        v74 = &v105[v44];
        if ( v106 == &v105[v44] )
        {
          v77 = a1[6];
        }
        else
        {
          v75 = (char *)a1[6];
          v76 = (char *)v42 + v106 - v74;
          do
          {
            if ( v75 )
            {
              *(_QWORD *)v75 = *(_QWORD *)v74;
              *(_QWORD *)v74 = 0;
            }
            v75 += 8;
            v74 += 8;
          }
          while ( v76 != v75 );
          v77 = a1[6];
        }
        v78 = (__int64 *)(v77 + 8 * (v45 - v47));
        a1[6] = v78;
        if ( v42 == v32 )
        {
          a1[6] = (char *)v78 + v44;
        }
        else
        {
          v79 = v32;
          v80 = (__int64 *)((char *)v78 + (char *)v42 - (char *)v32);
          do
          {
            if ( v78 )
            {
              *v78 = *v79;
              *v79 = 0;
            }
            ++v78;
            ++v79;
          }
          while ( v80 != v78 );
          a1[6] += v44;
          if ( v44 > 0 )
          {
            do
            {
              v81 = *(_QWORD *)v41;
              *(_QWORD *)v41 = 0;
              v82 = *v32;
              *v32 = v81;
              if ( v82 )
              {
                v90 = v41;
                sub_1444060(v82);
                j_j___libc_free_0(v82, 112);
                v41 = v90;
              }
              v41 += 8;
              ++v32;
              --v48;
            }
            while ( v48 );
          }
        }
      }
      else
      {
        v49 = (_QWORD *)a1[6];
        v50 = (char *)v42 - v43;
        v51 = (__int64 *)((char *)v42 - v43);
        do
        {
          if ( v49 )
          {
            *v49 = *v51;
            *v51 = 0;
          }
          ++v49;
          ++v51;
        }
        while ( v49 != (__int64 *)((char *)v42 + v43) );
        a1[6] += v43;
        v52 = (v50 - (char *)v32) >> 3;
        if ( v50 - (char *)v32 > 0 )
        {
          v53 = &v50[-8 * v52];
          v54 = &v42[-v52];
          do
          {
            v55 = *(_QWORD *)&v53[8 * v52 - 8];
            *(_QWORD *)&v53[8 * v52 - 8] = 0;
            v56 = v54[v52 - 1];
            v54[v52 - 1] = v55;
            if ( v56 )
            {
              v83 = v41;
              v84 = v53;
              v87 = v43;
              sub_1444060(v56);
              j_j___libc_free_0(v56, 112);
              v41 = v83;
              v53 = v84;
              v43 = v87;
            }
            --v52;
          }
          while ( v52 );
        }
        if ( v43 > 0 )
        {
          do
          {
            v57 = *(_QWORD *)v41;
            *(_QWORD *)v41 = 0;
            v58 = *v32;
            *v32 = v57;
            if ( v58 )
            {
              v88 = v41;
              sub_1444060(v58);
              j_j___libc_free_0(v58, 112);
              v41 = v88;
            }
            v41 += 8;
            ++v32;
            --v46;
          }
          while ( v46 );
        }
      }
LABEL_95:
      v59 = v106;
      v40 = v105;
      if ( v106 != v105 )
      {
        do
        {
          v60 = *(_QWORD *)v40;
          if ( *(_QWORD *)v40 )
          {
            sub_1444060(*(_QWORD *)v40);
            j_j___libc_free_0(v60, 112);
          }
          v40 += 8;
        }
        while ( v59 != v40 );
        v40 = v105;
      }
      goto LABEL_100;
    }
    if ( v45 > 0xFFFFFFFFFFFFFFFLL - v47 )
      sub_4262D8((__int64)"vector::_M_range_insert");
    if ( v47 >= v45 )
      v45 = v42 - v32;
    v62 = __CFADD__(v45, v47);
    v63 = v45 + v47;
    if ( v62 )
    {
      v64 = 0xFFFFFFFFFFFFFFFLL;
    }
    else
    {
      if ( !v63 )
      {
        v86 = 0;
        v65 = 0;
        v89 = 0;
LABEL_127:
        v68 = (__int64 *)((char *)v65 + v40 - v41);
        do
        {
          if ( v65 )
          {
            *v65 = *(_QWORD *)v41;
            *(_QWORD *)v41 = 0;
          }
          ++v65;
          v41 += 8;
        }
        while ( v65 != v68 );
        v69 = (__int64 *)a1[6];
        if ( v69 == v32 )
        {
          v71 = v65;
        }
        else
        {
          v70 = v32;
          v71 = (__int64 *)((char *)v68 + (char *)v69 - (char *)v32);
          do
          {
            if ( v68 )
            {
              *v68 = *v70;
              *v70 = 0;
            }
            ++v68;
            ++v70;
          }
          while ( v71 != v68 );
          v32 = (__int64 *)a1[6];
        }
        v72 = (__int64 *)a1[5];
        if ( v72 != v32 )
        {
          do
          {
            v73 = *v72;
            if ( *v72 )
            {
              sub_1444060(*v72);
              j_j___libc_free_0(v73, 112);
            }
            ++v72;
          }
          while ( v72 != v32 );
          v32 = (__int64 *)a1[5];
        }
        if ( v32 )
          j_j___libc_free_0(v32, a1[7] - (_QWORD)v32);
        a1[6] = v71;
        a1[5] = v89;
        a1[7] = v86;
        goto LABEL_95;
      }
      if ( v63 > 0xFFFFFFFFFFFFFFFLL )
        v63 = 0xFFFFFFFFFFFFFFFLL;
      v64 = v63;
    }
    v85 = v105;
    v65 = (__int64 *)sub_22077B0(v64 * 8);
    v66 = (__int64 *)a1[5];
    v41 = v85;
    v89 = v65;
    if ( v66 == v32 )
    {
      v86 = &v65[v64];
    }
    else
    {
      v67 = v65;
      v65 = (__int64 *)((char *)v65 + (char *)v32 - (char *)v66);
      do
      {
        if ( v67 )
        {
          *v67 = *v66;
          *v66 = 0;
        }
        ++v67;
        ++v66;
      }
      while ( v67 != v65 );
      v86 = &v89[v64];
    }
    goto LABEL_127;
  }
LABEL_100:
  if ( v40 )
    j_j___libc_free_0(v40, v107 - v40);
}
