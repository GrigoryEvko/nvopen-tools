// Function: sub_1E67460
// Address: 0x1e67460
//
void __fastcall sub_1E67460(_QWORD *a1, _QWORD *a2, char a3)
{
  char *v6; // rsi
  _BYTE *v7; // rsi
  _QWORD *v8; // rdi
  __int64 v9; // rcx
  __int64 v10; // rdx
  unsigned __int64 v11; // r15
  __int64 v12; // rax
  __int64 v13; // rcx
  char v14; // si
  __int64 v15; // rax
  __int64 v16; // rcx
  unsigned __int64 v17; // r15
  __int64 v18; // rax
  __int64 v19; // rdi
  __int64 v20; // rdx
  __int64 v21; // rax
  char v22; // si
  __int64 v23; // rsi
  __int64 v24; // rdx
  __int64 v25; // rax
  char v26; // si
  char v27; // r8
  bool v28; // si
  unsigned __int64 v29; // rsi
  __int64 *v30; // r15
  __int64 *v31; // r13
  char *v32; // rsi
  _QWORD *v33; // r14
  char *v34; // rsi
  __int64 *v35; // r14
  __int64 *v36; // rbx
  __int64 v37; // r13
  char *v38; // r13
  char *v39; // rdx
  __int64 *v40; // rdi
  __int64 v41; // r8
  __int64 v42; // rsi
  unsigned __int64 v43; // r11
  unsigned __int64 v44; // rbx
  unsigned __int64 v45; // r10
  unsigned __int64 v46; // r14
  _QWORD *v47; // rax
  char *v48; // rsi
  _QWORD *v49; // rcx
  __int64 v50; // r12
  char *v51; // rax
  __int64 *v52; // r14
  __int64 v53; // rcx
  __int64 v54; // r13
  __int64 v55; // rax
  __int64 v56; // r12
  char *v57; // rbx
  __int64 v58; // r12
  char *v59; // r14
  bool v60; // cf
  unsigned __int64 v61; // r10
  __int64 v62; // rbx
  __int64 *v63; // rax
  __int64 *v64; // rsi
  __int64 *v65; // rcx
  __int64 *v66; // rax
  __int64 *v67; // r13
  __int64 *v68; // rdx
  __int64 *v69; // r13
  __int64 *v70; // rbx
  __int64 v71; // r14
  char *v72; // rcx
  char *v73; // rax
  char *v74; // rbx
  __int64 v75; // rax
  __int64 *v76; // rax
  __int64 *v77; // r8
  __int64 *v78; // rcx
  __int64 v79; // rax
  __int64 v80; // r12
  char *v81; // [rsp+8h] [rbp-248h]
  char *v82; // [rsp+10h] [rbp-240h]
  char *v83; // [rsp+10h] [rbp-240h]
  __int64 *v84; // [rsp+10h] [rbp-240h]
  __int64 v85; // [rsp+18h] [rbp-238h]
  char *v86; // [rsp+18h] [rbp-238h]
  __int64 *v87; // [rsp+18h] [rbp-238h]
  char *v88; // [rsp+18h] [rbp-238h]
  _QWORD v89[2]; // [rsp+20h] [rbp-230h] BYREF
  unsigned __int64 v90; // [rsp+30h] [rbp-220h]
  _BYTE v91[64]; // [rsp+48h] [rbp-208h] BYREF
  __int64 v92; // [rsp+88h] [rbp-1C8h]
  __int64 v93; // [rsp+90h] [rbp-1C0h]
  unsigned __int64 v94; // [rsp+98h] [rbp-1B8h]
  _QWORD v95[2]; // [rsp+A0h] [rbp-1B0h] BYREF
  unsigned __int64 v96; // [rsp+B0h] [rbp-1A0h]
  _BYTE v97[64]; // [rsp+C8h] [rbp-188h] BYREF
  __int64 v98; // [rsp+108h] [rbp-148h]
  __int64 i; // [rsp+110h] [rbp-140h]
  unsigned __int64 v100; // [rsp+118h] [rbp-138h]
  char *v101; // [rsp+120h] [rbp-130h] BYREF
  char *v102; // [rsp+128h] [rbp-128h]
  char *v103; // [rsp+130h] [rbp-120h]
  __int64 v104; // [rsp+188h] [rbp-C8h]
  __int64 v105; // [rsp+190h] [rbp-C0h]
  __int64 v106; // [rsp+198h] [rbp-B8h]
  char v107[8]; // [rsp+1A0h] [rbp-B0h] BYREF
  __int64 v108; // [rsp+1A8h] [rbp-A8h]
  unsigned __int64 v109; // [rsp+1B0h] [rbp-A0h]
  __int64 v110; // [rsp+208h] [rbp-48h]
  __int64 v111; // [rsp+210h] [rbp-40h]
  __int64 v112; // [rsp+218h] [rbp-38h]

  a2[1] = a1;
  v101 = (char *)a2;
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
    v59 = (char *)a2;
    goto LABEL_101;
  }
  sub_1E64200(a1 + 5, v6, (__int64 *)&v101);
  v59 = v101;
  if ( v101 )
  {
LABEL_101:
    sub_1E63510(v59);
    j_j___libc_free_0(v59, 112);
  }
LABEL_4:
  if ( !a3 )
    return;
  sub_1E648D0(&v101, a1);
  v7 = v91;
  v8 = v89;
  sub_16CCCB0(v89, (__int64)v91, (__int64)&v101);
  v9 = v105;
  v10 = v104;
  v92 = 0;
  v93 = 0;
  v94 = 0;
  v11 = v105 - v104;
  if ( v105 == v104 )
  {
    v12 = 0;
  }
  else
  {
    if ( v11 > 0x7FFFFFFFFFFFFFE0LL )
      goto LABEL_169;
    v12 = sub_22077B0(v105 - v104);
    v9 = v105;
    v10 = v104;
  }
  v92 = v12;
  v93 = v12;
  v94 = v12 + v11;
  if ( v10 == v9 )
  {
    v13 = v12;
  }
  else
  {
    v13 = v12 + v9 - v10;
    do
    {
      if ( v12 )
      {
        *(_QWORD *)v12 = *(_QWORD *)v10;
        v14 = *(_BYTE *)(v10 + 24);
        *(_BYTE *)(v12 + 24) = v14;
        if ( v14 )
          *(__m128i *)(v12 + 8) = _mm_loadu_si128((const __m128i *)(v10 + 8));
      }
      v12 += 32;
      v10 += 32;
    }
    while ( v12 != v13 );
  }
  v8 = v95;
  v93 = v13;
  v7 = v97;
  sub_16CCCB0(v95, (__int64)v97, (__int64)v107);
  v15 = v111;
  v16 = v110;
  v98 = 0;
  i = 0;
  v100 = 0;
  v17 = v111 - v110;
  if ( v111 == v110 )
  {
    v19 = 0;
    goto LABEL_18;
  }
  if ( v17 > 0x7FFFFFFFFFFFFFE0LL )
LABEL_169:
    sub_4261EA(v8, v7, v10);
  v18 = sub_22077B0(v111 - v110);
  v16 = v110;
  v19 = v18;
  v15 = v111;
LABEL_18:
  v98 = v19;
  i = v19;
  v100 = v19 + v17;
  if ( v16 == v15 )
  {
    v21 = v19;
  }
  else
  {
    v20 = v19;
    v21 = v19 + v15 - v16;
    do
    {
      if ( v20 )
      {
        *(_QWORD *)v20 = *(_QWORD *)v16;
        v22 = *(_BYTE *)(v16 + 24);
        *(_BYTE *)(v20 + 24) = v22;
        if ( v22 )
          *(__m128i *)(v20 + 8) = _mm_loadu_si128((const __m128i *)(v16 + 8));
      }
      v20 += 32;
      v16 += 32;
    }
    while ( v20 != v21 );
  }
  for ( i = v21; ; v21 = i )
  {
    v24 = v92;
    if ( v93 - v92 != v21 - v19 )
      goto LABEL_25;
    if ( v92 == v93 )
      break;
    v25 = v19;
    while ( *(_QWORD *)v24 == *(_QWORD *)v25 )
    {
      v26 = *(_BYTE *)(v24 + 24);
      v27 = *(_BYTE *)(v25 + 24);
      if ( v26 && v27 )
      {
        if ( ((*(__int64 *)(v24 + 8) >> 1) & 3) != 0 )
          v28 = ((*(__int64 *)(v25 + 8) >> 1) & 3) == ((*(__int64 *)(v24 + 8) >> 1) & 3);
        else
          v28 = *(_QWORD *)(v24 + 16) == *(_QWORD *)(v25 + 16);
        if ( !v28 )
        {
          v23 = **(_QWORD **)(v93 - 32);
          if ( (v23 & 4) != 0 )
            goto LABEL_26;
          goto LABEL_37;
        }
      }
      else if ( v26 != v27 )
      {
        break;
      }
      v24 += 32;
      v25 += 32;
      if ( v93 == v24 )
        goto LABEL_41;
    }
LABEL_25:
    v23 = **(_QWORD **)(v93 - 32);
    if ( (v23 & 4) == 0 )
    {
LABEL_37:
      v29 = v23 & 0xFFFFFFFFFFFFFFF8LL;
      if ( (unsigned __int8)sub_1E62AD0(a2, v29) )
        sub_1E67220(a1[2], v29, (__int64)a2);
    }
LABEL_26:
    sub_1E645F0((__int64)v89);
    v19 = v98;
  }
LABEL_41:
  if ( v19 )
    j_j___libc_free_0(v19, v100 - v19);
  if ( v96 != v95[1] )
    _libc_free(v96);
  if ( v92 )
    j_j___libc_free_0(v92, v94 - v92);
  if ( v90 != v89[1] )
    _libc_free(v90);
  if ( v110 )
    j_j___libc_free_0(v110, v112 - v110);
  if ( v109 != v108 )
    _libc_free(v109);
  if ( v104 )
    j_j___libc_free_0(v104, v106 - v104);
  if ( v103 != v102 )
    _libc_free((unsigned __int64)v103);
  v30 = (__int64 *)a1[5];
  v31 = (__int64 *)a1[6];
  v101 = 0;
  v102 = 0;
  v103 = 0;
  if ( v31 != v30 )
  {
    while ( 1 )
    {
      v33 = (_QWORD *)*v30;
      if ( a2[4] )
      {
        if ( !(unsigned __int8)sub_1E62AD0(a2, *v33 & 0xFFFFFFFFFFFFFFF8LL)
          || !(unsigned __int8)sub_1E62AD0(a2, v33[4]) && v33[4] != a2[4] )
        {
LABEL_60:
          v32 = v102;
          if ( v102 == v103 )
          {
            sub_1E64200((__int64 *)&v101, v102, v30);
          }
          else
          {
            if ( v102 )
            {
              *(_QWORD *)v102 = *v30;
              *v30 = 0;
              v32 = v102;
            }
            v102 = v32 + 8;
          }
          goto LABEL_64;
        }
        v33 = (_QWORD *)*v30;
      }
      if ( a2 == v33 )
        goto LABEL_60;
      v33[1] = a2;
      v34 = (char *)a2[6];
      if ( v34 == (char *)a2[7] )
      {
        sub_1E64200(a2 + 5, v34, v30);
LABEL_64:
        if ( v31 == ++v30 )
          goto LABEL_71;
      }
      else
      {
        if ( v34 )
        {
          *(_QWORD *)v34 = *v30;
          *v30 = 0;
          v34 = (char *)a2[6];
        }
        ++v30;
        a2[6] = v34 + 8;
        if ( v31 == v30 )
        {
LABEL_71:
          v30 = (__int64 *)a1[5];
          v35 = (__int64 *)a1[6];
          if ( v30 != v35 )
          {
            v36 = (__int64 *)a1[5];
            do
            {
              v37 = *v36;
              if ( *v36 )
              {
                sub_1E63510(*v36);
                j_j___libc_free_0(v37, 112);
              }
              ++v36;
            }
            while ( v36 != v35 );
            a1[6] = v30;
            v30 = (__int64 *)a1[5];
          }
          break;
        }
      }
    }
  }
  v38 = v102;
  v39 = v101;
  if ( v101 != v102 )
  {
    v40 = (__int64 *)a1[6];
    v41 = v102 - v101;
    v42 = (char *)v40 - (char *)v30;
    v43 = (v102 - v101) >> 3;
    v44 = v43;
    v45 = v40 - v30;
    v46 = v45;
    if ( a1[7] - (_QWORD)v40 >= (unsigned __int64)(v102 - v101) )
    {
      if ( v41 >= (unsigned __int64)v42 )
      {
        v72 = &v101[v42];
        if ( v102 == &v101[v42] )
        {
          v75 = a1[6];
        }
        else
        {
          v73 = (char *)a1[6];
          v74 = (char *)v40 + v102 - v72;
          do
          {
            if ( v73 )
            {
              *(_QWORD *)v73 = *(_QWORD *)v72;
              *(_QWORD *)v72 = 0;
            }
            v73 += 8;
            v72 += 8;
          }
          while ( v73 != v74 );
          v75 = a1[6];
        }
        v76 = (__int64 *)(v75 + 8 * (v43 - v45));
        a1[6] = v76;
        if ( v40 == v30 )
        {
          a1[6] = (char *)v76 + v42;
        }
        else
        {
          v77 = v30;
          v78 = (__int64 *)((char *)v76 + (char *)v40 - (char *)v30);
          do
          {
            if ( v76 )
            {
              *v76 = *v77;
              *v77 = 0;
            }
            ++v76;
            ++v77;
          }
          while ( v78 != v76 );
          a1[6] += v42;
          if ( v42 > 0 )
          {
            do
            {
              v79 = *(_QWORD *)v39;
              *(_QWORD *)v39 = 0;
              v80 = *v30;
              *v30 = v79;
              if ( v80 )
              {
                v88 = v39;
                sub_1E63510(v80);
                j_j___libc_free_0(v80, 112);
                v39 = v88;
              }
              v39 += 8;
              ++v30;
              --v46;
            }
            while ( v46 );
          }
        }
      }
      else
      {
        v47 = (_QWORD *)a1[6];
        v48 = (char *)v40 - v41;
        v49 = (__int64 *)((char *)v40 - v41);
        do
        {
          if ( v47 )
          {
            *v47 = *v49;
            *v49 = 0;
          }
          ++v47;
          ++v49;
        }
        while ( v47 != (__int64 *)((char *)v40 + v41) );
        a1[6] += v41;
        v50 = (v48 - (char *)v30) >> 3;
        if ( v48 - (char *)v30 > 0 )
        {
          v51 = &v48[-8 * v50];
          v52 = &v40[-v50];
          do
          {
            v53 = *(_QWORD *)&v51[8 * v50 - 8];
            *(_QWORD *)&v51[8 * v50 - 8] = 0;
            v54 = v52[v50 - 1];
            v52[v50 - 1] = v53;
            if ( v54 )
            {
              v81 = v39;
              v82 = v51;
              v85 = v41;
              sub_1E63510(v54);
              j_j___libc_free_0(v54, 112);
              v39 = v81;
              v51 = v82;
              v41 = v85;
            }
            --v50;
          }
          while ( v50 );
        }
        if ( v41 > 0 )
        {
          do
          {
            v55 = *(_QWORD *)v39;
            *(_QWORD *)v39 = 0;
            v56 = *v30;
            *v30 = v55;
            if ( v56 )
            {
              v86 = v39;
              sub_1E63510(v56);
              j_j___libc_free_0(v56, 112);
              v39 = v86;
            }
            v39 += 8;
            ++v30;
            --v44;
          }
          while ( v44 );
        }
      }
LABEL_93:
      v57 = v102;
      v38 = v101;
      if ( v102 != v101 )
      {
        do
        {
          v58 = *(_QWORD *)v38;
          if ( *(_QWORD *)v38 )
          {
            sub_1E63510(*(_QWORD *)v38);
            j_j___libc_free_0(v58, 112);
          }
          v38 += 8;
        }
        while ( v57 != v38 );
        v38 = v101;
      }
      goto LABEL_98;
    }
    if ( v43 > 0xFFFFFFFFFFFFFFFLL - v45 )
      sub_4262D8((__int64)"vector::_M_range_insert");
    if ( v45 >= v43 )
      v43 = v40 - v30;
    v60 = __CFADD__(v43, v45);
    v61 = v43 + v45;
    if ( v60 )
    {
      v62 = 0xFFFFFFFFFFFFFFFLL;
    }
    else
    {
      if ( !v61 )
      {
        v84 = 0;
        v65 = 0;
        v87 = 0;
LABEL_125:
        v66 = (__int64 *)((char *)v65 + v38 - v39);
        do
        {
          if ( v65 )
          {
            *v65 = *(_QWORD *)v39;
            *(_QWORD *)v39 = 0;
          }
          ++v65;
          v39 += 8;
        }
        while ( v66 != v65 );
        v67 = (__int64 *)a1[6];
        if ( v67 == v30 )
        {
          v69 = v66;
        }
        else
        {
          v68 = v30;
          v69 = (__int64 *)((char *)v66 + (char *)v67 - (char *)v30);
          do
          {
            if ( v66 )
            {
              *v66 = *v68;
              *v68 = 0;
            }
            ++v66;
            ++v68;
          }
          while ( v69 != v66 );
          v30 = (__int64 *)a1[6];
        }
        v70 = (__int64 *)a1[5];
        if ( v70 != v30 )
        {
          do
          {
            v71 = *v70;
            if ( *v70 )
            {
              sub_1E63510(*v70);
              j_j___libc_free_0(v71, 112);
            }
            ++v70;
          }
          while ( v70 != v30 );
          v30 = (__int64 *)a1[5];
        }
        if ( v30 )
          j_j___libc_free_0(v30, a1[7] - (_QWORD)v30);
        a1[6] = v69;
        a1[5] = v87;
        a1[7] = v84;
        goto LABEL_93;
      }
      if ( v61 > 0xFFFFFFFFFFFFFFFLL )
        v61 = 0xFFFFFFFFFFFFFFFLL;
      v62 = v61;
    }
    v83 = v101;
    v63 = (__int64 *)sub_22077B0(v62 * 8);
    v64 = (__int64 *)a1[5];
    v39 = v83;
    v87 = v63;
    if ( v64 == v30 )
    {
      v65 = v63;
      v84 = &v63[v62];
    }
    else
    {
      v65 = (__int64 *)((char *)v63 + (char *)v30 - (char *)v64);
      do
      {
        if ( v63 )
        {
          *v63 = *v64;
          *v64 = 0;
        }
        ++v63;
        ++v64;
      }
      while ( v63 != v65 );
      v84 = &v87[v62];
    }
    goto LABEL_125;
  }
LABEL_98:
  if ( v38 )
    j_j___libc_free_0(v38, v103 - v38);
}
