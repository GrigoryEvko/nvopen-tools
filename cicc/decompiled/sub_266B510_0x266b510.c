// Function: sub_266B510
// Address: 0x266b510
//
__int64 __fastcall sub_266B510(__int64 a1)
{
  _QWORD *v2; // r12
  _QWORD *v3; // r13
  __int64 v4; // rax
  __int64 v5; // rsi
  unsigned __int64 v6; // rsi
  unsigned __int64 *v7; // rdx
  __int64 v8; // rcx
  __int64 v9; // r8
  __int64 v10; // r9
  unsigned __int64 *v11; // r13
  char v12; // di
  unsigned __int64 *v13; // r14
  _QWORD *v14; // rax
  __int64 v15; // r13
  __int64 v16; // rbx
  __int64 v17; // r14
  __int64 v18; // rax
  unsigned __int64 v19; // rbx
  __m128i *v20; // r14
  __m128i *v21; // r13
  __int64 v22; // rax
  unsigned __int64 v23; // r12
  _QWORD *v24; // rbx
  unsigned __int64 v25; // r14
  __int64 v26; // r13
  _QWORD *v27; // r14
  __int64 v28; // rdx
  __int64 v29; // rax
  _QWORD *v30; // rbx
  unsigned int v31; // edx
  _QWORD *v32; // r14
  __int64 v33; // rax
  __int64 v34; // rdx
  unsigned int v35; // eax
  unsigned int v37; // eax
  _QWORD *v38; // rbx
  _QWORD *v39; // r12
  __int64 v40; // rsi
  void *v41; // r8
  __int64 v42; // rdx
  unsigned int v43; // edx
  unsigned int v44; // eax
  _QWORD *v45; // r14
  unsigned __int64 v46; // rax
  unsigned __int64 v47; // rax
  __int64 v48; // rax
  __int64 v49; // rax
  _QWORD *v50; // rbx
  void *v51; // rcx
  __int64 v52; // rax
  _QWORD *v53; // r14
  __int64 v54; // rdx
  void *v55; // rdx
  _QWORD *v56; // rbx
  __int8 v57; // al
  __int64 v58; // rax
  bool v59; // zf
  void *v60; // [rsp+8h] [rbp-228h]
  __int64 v61; // [rsp+20h] [rbp-210h]
  unsigned int v62; // [rsp+20h] [rbp-210h]
  unsigned int v63; // [rsp+20h] [rbp-210h]
  unsigned int v64; // [rsp+20h] [rbp-210h]
  int v65; // [rsp+20h] [rbp-210h]
  void *v66; // [rsp+20h] [rbp-210h]
  void *v67; // [rsp+20h] [rbp-210h]
  unsigned __int8 v68; // [rsp+28h] [rbp-208h]
  __m128i *v69; // [rsp+30h] [rbp-200h] BYREF
  __m128i *v70; // [rsp+38h] [rbp-1F8h]
  const __m128i *i; // [rsp+40h] [rbp-1F0h]
  void *v72; // [rsp+50h] [rbp-1E0h]
  _QWORD v73[2]; // [rsp+58h] [rbp-1D8h] BYREF
  __int64 v74; // [rsp+68h] [rbp-1C8h]
  __int64 v75; // [rsp+70h] [rbp-1C0h]
  __m128i v76; // [rsp+80h] [rbp-1B0h] BYREF
  __int64 v77; // [rsp+90h] [rbp-1A0h]
  __int64 v78; // [rsp+98h] [rbp-198h]
  __int64 v79; // [rsp+A0h] [rbp-190h]
  unsigned __int64 *v80; // [rsp+B0h] [rbp-180h] BYREF
  __int64 v81; // [rsp+B8h] [rbp-178h] BYREF
  __int64 v82; // [rsp+C0h] [rbp-170h] BYREF
  __int64 v83; // [rsp+C8h] [rbp-168h]
  __int64 v84; // [rsp+D0h] [rbp-160h]
  __int64 v85; // [rsp+E0h] [rbp-150h] BYREF
  _QWORD *v86; // [rsp+E8h] [rbp-148h]
  __int64 v87; // [rsp+F0h] [rbp-140h]
  unsigned int v88; // [rsp+F8h] [rbp-138h]
  _QWORD *v89; // [rsp+108h] [rbp-128h]
  unsigned int v90; // [rsp+118h] [rbp-118h]
  char v91; // [rsp+120h] [rbp-110h]
  __int64 v92; // [rsp+130h] [rbp-100h]
  unsigned __int64 v93; // [rsp+138h] [rbp-F8h] BYREF
  _QWORD *v94; // [rsp+140h] [rbp-F0h]
  __int64 v95; // [rsp+148h] [rbp-E8h]
  __int64 v96; // [rsp+150h] [rbp-E0h] BYREF
  void *s; // [rsp+158h] [rbp-D8h]
  _BYTE v98[12]; // [rsp+160h] [rbp-D0h]
  char v99; // [rsp+16Ch] [rbp-C4h]
  char v100; // [rsp+170h] [rbp-C0h] BYREF
  __int64 *v101; // [rsp+190h] [rbp-A0h]
  int v102; // [rsp+198h] [rbp-98h] BYREF
  _QWORD *v103; // [rsp+1A0h] [rbp-90h]
  int *v104; // [rsp+1A8h] [rbp-88h]
  int *v105; // [rsp+1B0h] [rbp-80h]
  __int64 v106; // [rsp+1B8h] [rbp-78h]
  _QWORD v107[3]; // [rsp+1C0h] [rbp-70h] BYREF
  int v108; // [rsp+1D8h] [rbp-58h]
  __int64 v109; // [rsp+1E0h] [rbp-50h]
  __int64 v110; // [rsp+1E8h] [rbp-48h]
  __int64 v111; // [rsp+1F0h] [rbp-40h]
  int v112; // [rsp+1F8h] [rbp-38h]

  v85 = 0;
  v88 = 128;
  v87 = 0;
  v2 = (_QWORD *)sub_C7D670(6144, 8);
  v86 = v2;
  v81 = 2;
  v82 = 0;
  v83 = -4096;
  v80 = (unsigned __int64 *)&unk_4A1FB50;
  v84 = 0;
  v3 = v2 + 768;
  if ( v2 != v2 + 768 )
  {
    v4 = -4096;
    do
    {
      if ( v2 )
      {
        v5 = v81;
        v2[3] = v4;
        v2[2] = 0;
        v2[1] = v5 & 6;
        if ( v4 != -4096 && v4 != 0 && v4 != -8192 )
        {
          sub_BD6050(v2 + 1, v5 & 0xFFFFFFFFFFFFFFF8LL);
          v4 = v83;
        }
        *v2 = &unk_4A1FB50;
        v2[4] = v84;
      }
      v2 += 6;
    }
    while ( v3 != v2 );
    v80 = (unsigned __int64 *)&unk_49DB368;
    if ( v4 != -4096 && v4 != 0 && v4 != -8192 )
      sub_BD60C0(&v81);
  }
  s = &v100;
  v104 = &v102;
  v105 = &v102;
  v80 = (unsigned __int64 *)&v82;
  v91 = 0;
  v92 = 0;
  v93 = 0;
  v94 = 0;
  v95 = 0;
  v96 = 0;
  *(_QWORD *)v98 = 4;
  *(_DWORD *)&v98[8] = 0;
  v99 = 1;
  v101 = &v85;
  v102 = 0;
  v103 = 0;
  v106 = 0;
  memset(v107, 0, sizeof(v107));
  v108 = 0;
  v109 = 0;
  v110 = 0;
  v111 = 0;
  v112 = 0;
  v81 = 0x400000000LL;
  sub_BAA9B0(a1, (__int64)&v80, 0);
  v6 = (unsigned __int64)&v80;
  sub_BAA9B0(a1, (__int64)&v80, 1);
  v11 = v80;
  v12 = v99;
  v13 = &v80[(unsigned int)v81];
  if ( v80 != v13 )
  {
    while ( 1 )
    {
      while ( 1 )
      {
        v6 = *v11;
        if ( v12 )
          break;
LABEL_88:
        ++v11;
        sub_C8CC70((__int64)&v96, v6, (__int64)v7, v8, v9, v10);
        v12 = v99;
        if ( v13 == v11 )
          goto LABEL_19;
      }
      v14 = s;
      v8 = *(unsigned int *)&v98[4];
      v7 = (unsigned __int64 *)((char *)s + 8 * *(unsigned int *)&v98[4]);
      if ( s == v7 )
      {
LABEL_90:
        if ( *(_DWORD *)&v98[4] >= *(_DWORD *)v98 )
          goto LABEL_88;
        v8 = (unsigned int)(*(_DWORD *)&v98[4] + 1);
        ++v11;
        ++*(_DWORD *)&v98[4];
        *v7 = v6;
        v12 = v99;
        ++v96;
        if ( v13 == v11 )
          break;
      }
      else
      {
        while ( v6 != *v14 )
        {
          if ( v7 == ++v14 )
            goto LABEL_90;
        }
        if ( v13 == ++v11 )
          break;
      }
    }
  }
LABEL_19:
  v15 = *(_QWORD *)(a1 + 32);
  v16 = a1 + 24;
  v69 = 0;
  v70 = 0;
  for ( i = 0; v16 != v15; v15 = *(_QWORD *)(v15 + 8) )
  {
    v17 = v15 - 56;
    if ( !v15 )
      v17 = 0;
    if ( !sub_B2FC80(v17) && (*(_BYTE *)(v17 + 32) & 0xF) != 1 && !(unsigned __int8)sub_2665490(v17) )
    {
      v18 = sub_3148040(v17, 0);
      v76.m128i_i64[1] = v17;
      v6 = (unsigned __int64)v70;
      v76.m128i_i64[0] = v18;
      if ( v70 == i )
      {
        sub_26693E0((unsigned __int64 *)&v69, v70, &v76);
      }
      else
      {
        if ( v70 )
        {
          *v70 = _mm_loadu_si128(&v76);
          v6 = (unsigned __int64)v70;
        }
        v6 += 16LL;
        v70 = (__m128i *)v6;
      }
    }
  }
  sub_266B4A0((__int64)&v69);
  v19 = (unsigned __int64)v69;
  v20 = v70;
  v21 = v69;
  while ( v20 != v21 )
  {
    do
    {
      ++v21;
      if ( (__m128i *)v19 != &v21[-1] && v21[-2].m128i_i64[0] == v21[-1].m128i_i64[0] )
        break;
      if ( v20 == v21 )
        goto LABEL_42;
    }
    while ( v21->m128i_i64[0] != v21[-1].m128i_i64[0] );
    v22 = v21[-1].m128i_i64[1];
    v76 = (__m128i)6uLL;
    v77 = v22;
    if ( v22 != 0 && v22 != -4096 && v22 != -8192 )
      sub_BD73F0((__int64)&v76);
    v6 = (unsigned __int64)&v76;
    sub_2669560((__int64)&v93, &v76);
    if ( v77 != -4096 && v77 != 0 && v77 != -8192 )
      sub_BD60C0(&v76);
  }
LABEL_42:
  v68 = 0;
  v23 = v93;
  v24 = v94;
  do
  {
    v93 = 0;
    v94 = 0;
    v61 = v95;
    v95 = 0;
    if ( v24 != (_QWORD *)v23 )
    {
      v25 = v23;
      do
      {
        v26 = *(_QWORD *)(v25 + 16);
        if ( v26 && !sub_B2FC80(*(_QWORD *)(v25 + 16)) && (*(_BYTE *)(v26 + 32) & 0xF) != 1 )
        {
          v6 = v26;
          v68 |= sub_26696B0((__int64)&v85, v26);
        }
        v25 += 24LL;
      }
      while ( (_QWORD *)v25 != v24 );
      v27 = (_QWORD *)v23;
      do
      {
        v28 = v27[2];
        LOBYTE(v6) = v28 != 0;
        if ( v28 != -4096 && v28 != 0 && v28 != -8192 )
          sub_BD60C0(v27);
        v27 += 3;
      }
      while ( v24 != v27 );
    }
    if ( v23 )
    {
      v6 = v61 - v23;
      j_j___libc_free_0(v23);
    }
    v24 = v94;
    v23 = v93;
  }
  while ( v94 != (_QWORD *)v93 );
  sub_2665870(v103);
  v103 = 0;
  v106 = 0;
  v104 = &v102;
  v105 = &v102;
  sub_2666560((__int64)v107);
  v29 = (unsigned int)v87;
  ++v85;
  if ( !v87 )
    goto LABEL_76;
  v30 = v86;
  v31 = 4 * v87;
  v6 = 48LL * v88;
  if ( (unsigned int)(4 * v87) < 0x40 )
    v31 = 64;
  v32 = (_QWORD *)((char *)v86 + v6);
  if ( v88 > v31 )
  {
    v73[0] = 2;
    v73[1] = 0;
    v41 = &unk_49DB368;
    v74 = -4096;
    v72 = &unk_4A1FB50;
    v75 = 0;
    v76.m128i_i64[1] = 2;
    v77 = 0;
    v78 = -8192;
    v76.m128i_i64[0] = (__int64)&unk_4A1FB50;
    v79 = 0;
    do
    {
      v42 = v30[3];
      *v30 = v41;
      if ( v42 != 0 && v42 != -4096 && v42 != -8192 )
      {
        v60 = v41;
        v62 = v29;
        sub_BD60C0(v30 + 1);
        v41 = v60;
        v29 = v62;
      }
      v30 += 6;
    }
    while ( v30 != v32 );
    v76.m128i_i64[0] = (__int64)&unk_49DB368;
    if ( v78 != 0 && v78 != -4096 && v78 != -8192 )
    {
      v63 = v29;
      sub_BD60C0(&v76.m128i_i64[1]);
      v29 = v63;
    }
    v72 = &unk_49DB368;
    if ( v74 != 0 && v74 != -4096 && v74 != -8192 )
    {
      v64 = v29;
      sub_BD60C0(v73);
      v29 = v64;
    }
    if ( (_DWORD)v29 )
    {
      v43 = v29 - 1;
      v29 = 64;
      if ( v43 )
      {
        _BitScanReverse(&v44, v43);
        v29 = (unsigned int)(1 << (33 - (v44 ^ 0x1F)));
        if ( (int)v29 < 64 )
          v29 = 64;
      }
    }
    v45 = v86;
    if ( v88 == (_DWORD)v29 )
    {
      v87 = 0;
      v76.m128i_i64[1] = 2;
      v55 = &unk_4A1FB50;
      v56 = &v86[6 * v29];
      v77 = 0;
      v78 = -4096;
      v76.m128i_i64[0] = (__int64)&unk_4A1FB50;
      v79 = 0;
      if ( v56 != v86 )
      {
        do
        {
          if ( v45 )
          {
            v57 = v76.m128i_i8[8];
            v45[2] = 0;
            v45[1] = v57 & 6;
            v58 = v78;
            v59 = v78 == 0;
            v45[3] = v78;
            LOBYTE(v6) = !v59;
            if ( v58 != -4096 && !v59 && v58 != -8192 )
            {
              v67 = v55;
              v6 = v76.m128i_i64[1] & 0xFFFFFFFFFFFFFFF8LL;
              sub_BD6050(v45 + 1, v76.m128i_i64[1] & 0xFFFFFFFFFFFFFFF8LL);
              v55 = v67;
            }
            *v45 = v55;
            v45[4] = v79;
          }
          v45 += 6;
        }
        while ( v56 != v45 );
        v76.m128i_i64[0] = (__int64)&unk_49DB368;
        if ( v78 != 0 && v78 != -4096 && v78 != -8192 )
          goto LABEL_75;
      }
      goto LABEL_76;
    }
    v65 = v29;
    sub_C7D6A0((__int64)v86, v6, 8);
    if ( !v65 )
    {
      v86 = 0;
      v87 = 0;
      v88 = 0;
      goto LABEL_76;
    }
    v6 = 8;
    v46 = (4 * v65 / 3u + 1) | ((unsigned __int64)(4 * v65 / 3u + 1) >> 1);
    v47 = (((v46 >> 2) | v46) >> 4) | (v46 >> 2) | v46;
    v48 = ((((v47 >> 8) | v47) >> 16) | (v47 >> 8) | v47) + 1;
    v88 = v48;
    v49 = sub_C7D670(48 * v48, 8);
    v87 = 0;
    v50 = (_QWORD *)v49;
    v86 = (_QWORD *)v49;
    v76.m128i_i64[1] = 2;
    v77 = 0;
    v51 = &unk_4A1FB50;
    v78 = -4096;
    v76.m128i_i64[0] = (__int64)&unk_4A1FB50;
    v79 = 0;
    v52 = 6LL * v88;
    v53 = &v50[v52];
    if ( v50 == &v50[v52] )
      goto LABEL_76;
    v33 = -4096;
    do
    {
      if ( v50 )
      {
        v54 = v76.m128i_i64[1];
        v50[2] = 0;
        v50[3] = v33;
        v6 = v54 & 6;
        v50[1] = v6;
        LOBYTE(v6) = v33 != 0;
        if ( v33 != 0 && v33 != -4096 && v33 != -8192 )
        {
          v66 = v51;
          v6 = v54 & 0xFFFFFFFFFFFFFFF8LL;
          sub_BD6050(v50 + 1, v54 & 0xFFFFFFFFFFFFFFF8LL);
          v33 = v78;
          v51 = v66;
        }
        *v50 = v51;
        v50[4] = v79;
      }
      v50 += 6;
    }
    while ( v53 != v50 );
  }
  else
  {
    v76.m128i_i64[1] = 2;
    v77 = 0;
    v78 = -4096;
    v76.m128i_i64[0] = (__int64)&unk_4A1FB50;
    v33 = -4096;
    v79 = 0;
    if ( v86 == v32 )
    {
      v87 = 0;
      goto LABEL_76;
    }
    do
    {
      v34 = v30[3];
      if ( v33 != v34 )
      {
        LOBYTE(v6) = v34 != 0;
        if ( v34 != -4096 && v34 != 0 && v34 != -8192 )
        {
          sub_BD60C0(v30 + 1);
          v33 = v78;
        }
        v30[3] = v33;
        if ( v33 != -4096 && v33 != 0 && v33 != -8192 )
        {
          v6 = v76.m128i_i64[1] & 0xFFFFFFFFFFFFFFF8LL;
          sub_BD6050(v30 + 1, v76.m128i_i64[1] & 0xFFFFFFFFFFFFFFF8LL);
        }
        v33 = v78;
      }
      v30 += 6;
      *(v30 - 2) = v79;
    }
    while ( v30 != v32 );
    v87 = 0;
  }
  v76.m128i_i64[0] = (__int64)&unk_49DB368;
  if ( v33 != 0 && v33 != -8192 && v33 != -4096 )
LABEL_75:
    sub_BD60C0(&v76.m128i_i64[1]);
LABEL_76:
  if ( v91 )
  {
    v37 = v90;
    v91 = 0;
    if ( v90 )
    {
      v38 = v89;
      v39 = &v89[2 * v90];
      do
      {
        if ( *v38 != -8192 && *v38 != -4096 )
        {
          v40 = v38[1];
          if ( v40 )
            sub_B91220((__int64)(v38 + 1), v40);
        }
        v38 += 2;
      }
      while ( v39 != v38 );
      v37 = v90;
    }
    v6 = 16LL * v37;
    sub_C7D6A0((__int64)v89, v6, 8);
  }
  ++v96;
  if ( v99 )
    goto LABEL_82;
  v35 = 4 * (*(_DWORD *)&v98[4] - *(_DWORD *)&v98[8]);
  if ( v35 < 0x20 )
    v35 = 32;
  if ( *(_DWORD *)v98 <= v35 )
  {
    memset(s, -1, 8LL * *(unsigned int *)v98);
LABEL_82:
    *(_QWORD *)&v98[4] = 0;
    goto LABEL_83;
  }
  sub_C8C990((__int64)&v96, v6);
LABEL_83:
  if ( v69 )
    j_j___libc_free_0((unsigned __int64)v69);
  if ( v80 != (unsigned __int64 *)&v82 )
    _libc_free((unsigned __int64)v80);
  sub_26661E0((__int64)&v85);
  return v68;
}
