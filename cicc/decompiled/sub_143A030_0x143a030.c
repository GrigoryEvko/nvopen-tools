// Function: sub_143A030
// Address: 0x143a030
//
void __fastcall sub_143A030(__int64 a1, __int64 *a2)
{
  __int64 *v2; // rax
  __int64 *v3; // rbx
  __int64 *v4; // r13
  __int64 v5; // rdx
  _QWORD *v6; // rax
  _QWORD *v7; // rdx
  __int64 *v8; // r13
  __int64 *v9; // r12
  __int64 v10; // r15
  __int64 *v11; // rbx
  __int64 *v12; // r14
  __int64 v13; // rdi
  __int64 v14; // rax
  __int64 v15; // rax
  void *v16; // rdi
  unsigned int v17; // eax
  __int64 v18; // rdx
  unsigned __int64 v19; // rdi
  __int64 v20; // rax
  __int64 v21; // rdi
  __int64 v22; // rdi
  unsigned __int64 *v23; // rbx
  unsigned __int64 *v24; // r12
  unsigned __int64 v25; // rdi
  unsigned __int64 *v26; // r12
  _QWORD *v27; // rbx
  _QWORD *v28; // r12
  __int64 v29; // r13
  __int64 v30; // rdi
  unsigned int v31; // ecx
  _QWORD *v32; // rdi
  unsigned int v33; // eax
  int v34; // eax
  unsigned __int64 v35; // rax
  unsigned __int64 v36; // rax
  int v37; // ebx
  __int64 v38; // r12
  _QWORD *v39; // rax
  _QWORD *i; // rdx
  _QWORD *v41; // rbx
  _QWORD *v42; // r14
  __int64 v43; // rax
  unsigned __int64 *v44; // rdx
  unsigned __int64 *v45; // r12
  unsigned __int64 *v46; // rbx
  unsigned __int64 v47; // rdi
  unsigned __int64 *v48; // rbx
  unsigned __int64 v49; // rdi
  _QWORD *v50; // rax
  _QWORD v51[2]; // [rsp+28h] [rbp-318h] BYREF
  __int64 v52; // [rsp+38h] [rbp-308h]
  __int64 v53; // [rsp+40h] [rbp-300h]
  void *v54; // [rsp+50h] [rbp-2F0h]
  _QWORD v55[2]; // [rsp+58h] [rbp-2E8h] BYREF
  __int64 v56; // [rsp+68h] [rbp-2D8h]
  __int64 v57; // [rsp+70h] [rbp-2D0h]
  unsigned __int64 v58[2]; // [rsp+80h] [rbp-2C0h] BYREF
  char v59; // [rsp+90h] [rbp-2B0h] BYREF
  __int64 v60; // [rsp+98h] [rbp-2A8h]
  _QWORD *v61; // [rsp+A0h] [rbp-2A0h]
  __int64 v62; // [rsp+A8h] [rbp-298h]
  unsigned int v63; // [rsp+B0h] [rbp-290h]
  __int64 *v64; // [rsp+C0h] [rbp-280h]
  char v65; // [rsp+C8h] [rbp-278h]
  int v66; // [rsp+CCh] [rbp-274h]
  __int64 v67; // [rsp+D0h] [rbp-270h] BYREF
  _QWORD *v68; // [rsp+D8h] [rbp-268h]
  __int64 v69; // [rsp+E0h] [rbp-260h]
  __int64 v70; // [rsp+E8h] [rbp-258h]
  __int64 *v71; // [rsp+F0h] [rbp-250h]
  __int64 *v72; // [rsp+F8h] [rbp-248h]
  __int64 v73; // [rsp+100h] [rbp-240h]
  unsigned __int64 v74; // [rsp+108h] [rbp-238h]
  unsigned __int64 v75; // [rsp+110h] [rbp-230h]
  unsigned __int64 *v76; // [rsp+118h] [rbp-228h]
  __int64 v77; // [rsp+120h] [rbp-220h]
  _BYTE v78[32]; // [rsp+128h] [rbp-218h] BYREF
  unsigned __int64 *v79; // [rsp+148h] [rbp-1F8h]
  __int64 v80; // [rsp+150h] [rbp-1F0h]
  _QWORD v81[3]; // [rsp+158h] [rbp-1E8h] BYREF
  __int64 v82; // [rsp+170h] [rbp-1D0h] BYREF
  _QWORD *v83; // [rsp+178h] [rbp-1C8h]
  __int64 v84; // [rsp+180h] [rbp-1C0h]
  __int64 v85; // [rsp+188h] [rbp-1B8h]
  __int64 v86; // [rsp+190h] [rbp-1B0h]
  __int64 v87; // [rsp+198h] [rbp-1A8h]
  __int64 v88; // [rsp+1A0h] [rbp-1A0h]
  int v89; // [rsp+1A8h] [rbp-198h]
  __int64 v90; // [rsp+1B8h] [rbp-188h]
  _BYTE *v91; // [rsp+1C0h] [rbp-180h]
  _BYTE *v92; // [rsp+1C8h] [rbp-178h]
  __int64 v93; // [rsp+1D0h] [rbp-170h]
  int v94; // [rsp+1D8h] [rbp-168h]
  _BYTE v95[128]; // [rsp+1E0h] [rbp-160h] BYREF
  __int64 v96; // [rsp+260h] [rbp-E0h]
  _BYTE *v97; // [rsp+268h] [rbp-D8h]
  _BYTE *v98; // [rsp+270h] [rbp-D0h]
  __int64 v99; // [rsp+278h] [rbp-C8h]
  int v100; // [rsp+280h] [rbp-C0h]
  _BYTE v101[184]; // [rsp+288h] [rbp-B8h] BYREF

  v58[0] = (unsigned __int64)&v59;
  v58[1] = 0x100000000LL;
  v64 = a2;
  v60 = 0;
  v61 = 0;
  v62 = 0;
  v63 = 0;
  v65 = 0;
  v66 = 0;
  sub_15D3930(v58);
  v76 = (unsigned __int64 *)v78;
  v77 = 0x400000000LL;
  v79 = v81;
  v67 = 0;
  v68 = 0;
  v69 = 0;
  v70 = 0;
  v71 = 0;
  v72 = 0;
  v73 = 0;
  v74 = 0;
  v75 = 0;
  v80 = 0;
  v81[0] = 0;
  v81[1] = 1;
  sub_1400540((__int64)&v67, (__int64)v58);
  v82 = 0;
  v91 = v95;
  v92 = v95;
  v97 = v101;
  v98 = v101;
  v83 = 0;
  v84 = 0;
  v85 = 0;
  v86 = 0;
  v87 = 0;
  v88 = 0;
  v89 = 0;
  v90 = 0;
  v93 = 16;
  v94 = 0;
  v96 = 0;
  v99 = 16;
  v100 = 0;
  sub_137CAE0((__int64)&v82, a2, (__int64)&v67, 0);
  v2 = (__int64 *)sub_22077B0(8);
  v3 = v2;
  if ( v2 )
    sub_13702A0(v2, a2, (__int64)&v82, (__int64)&v67);
  v4 = *(__int64 **)(a1 + 16);
  *(_QWORD *)(a1 + 16) = v3;
  if ( v4 )
  {
    sub_1368A00(v4);
    j_j___libc_free_0(v4, 8);
    v3 = *(__int64 **)(a1 + 16);
  }
  *(_QWORD *)(a1 + 8) = v3;
  if ( v98 != v97 )
    _libc_free((unsigned __int64)v98);
  if ( v92 != v91 )
    _libc_free((unsigned __int64)v92);
  j___libc_free_0(v87);
  if ( (_DWORD)v85 )
  {
    v41 = v83;
    v51[0] = 2;
    v51[1] = 0;
    v42 = &v83[5 * (unsigned int)v85];
    v52 = -8;
    v53 = 0;
    v55[0] = 2;
    v55[1] = 0;
    v56 = -16;
    v54 = &unk_49E8A80;
    v57 = 0;
    do
    {
      v43 = v41[3];
      *v41 = &unk_49EE2B0;
      if ( v43 != -8 && v43 != 0 && v43 != -16 )
        sub_1649B30(v41 + 1);
      v41 += 5;
    }
    while ( v42 != v41 );
    v54 = &unk_49EE2B0;
    if ( v56 != -8 && v56 != 0 && v56 != -16 )
      sub_1649B30(v55);
    if ( v52 != 0 && v52 != -8 && v52 != -16 )
      sub_1649B30(v51);
  }
  j___libc_free_0(v83);
  ++v67;
  if ( !(_DWORD)v69 )
  {
    if ( !HIDWORD(v69) )
      goto LABEL_16;
    v5 = (unsigned int)v70;
    if ( (unsigned int)v70 > 0x40 )
    {
      j___libc_free_0(v68);
      v68 = 0;
      v69 = 0;
      LODWORD(v70) = 0;
      goto LABEL_16;
    }
    goto LABEL_13;
  }
  v31 = 4 * v69;
  v5 = (unsigned int)v70;
  if ( (unsigned int)(4 * v69) < 0x40 )
    v31 = 64;
  if ( (unsigned int)v70 <= v31 )
  {
LABEL_13:
    v6 = v68;
    v7 = &v68[2 * v5];
    if ( v68 != v7 )
    {
      do
      {
        *v6 = -8;
        v6 += 2;
      }
      while ( v7 != v6 );
    }
    v69 = 0;
    goto LABEL_16;
  }
  v32 = v68;
  if ( (_DWORD)v69 == 1 )
  {
    v38 = 2048;
    v37 = 128;
LABEL_71:
    j___libc_free_0(v68);
    LODWORD(v70) = v37;
    v39 = (_QWORD *)sub_22077B0(v38);
    v69 = 0;
    v68 = v39;
    for ( i = &v39[2 * (unsigned int)v70]; i != v39; v39 += 2 )
    {
      if ( v39 )
        *v39 = -8;
    }
    goto LABEL_16;
  }
  _BitScanReverse(&v33, v69 - 1);
  v34 = 1 << (33 - (v33 ^ 0x1F));
  if ( v34 < 64 )
    v34 = 64;
  if ( (_DWORD)v70 != v34 )
  {
    v35 = (4 * v34 / 3u + 1) | ((unsigned __int64)(4 * v34 / 3u + 1) >> 1);
    v36 = ((v35 | (v35 >> 2)) >> 4) | v35 | (v35 >> 2) | ((((v35 | (v35 >> 2)) >> 4) | v35 | (v35 >> 2)) >> 8);
    v37 = (v36 | (v36 >> 16)) + 1;
    v38 = 16 * ((v36 | (v36 >> 16)) + 1);
    goto LABEL_71;
  }
  v69 = 0;
  v50 = &v68[2 * (unsigned int)v70];
  do
  {
    if ( v32 )
      *v32 = -8;
    v32 += 2;
  }
  while ( v50 != v32 );
LABEL_16:
  v8 = v72;
  v9 = v71;
  if ( v71 != v72 )
  {
    do
    {
      v10 = *v9;
      v11 = *(__int64 **)(*v9 + 16);
      if ( *(__int64 **)(*v9 + 8) == v11 )
      {
        *(_BYTE *)(v10 + 160) = 1;
      }
      else
      {
        v12 = *(__int64 **)(*v9 + 8);
        do
        {
          v13 = *v12++;
          sub_13FACC0(v13);
        }
        while ( v11 != v12 );
        *(_BYTE *)(v10 + 160) = 1;
        v14 = *(_QWORD *)(v10 + 8);
        if ( *(_QWORD *)(v10 + 16) != v14 )
          *(_QWORD *)(v10 + 16) = v14;
      }
      v15 = *(_QWORD *)(v10 + 32);
      if ( v15 != *(_QWORD *)(v10 + 40) )
        *(_QWORD *)(v10 + 40) = v15;
      ++*(_QWORD *)(v10 + 56);
      v16 = *(void **)(v10 + 72);
      if ( v16 == *(void **)(v10 + 64) )
      {
        *(_QWORD *)v10 = 0;
      }
      else
      {
        v17 = 4 * (*(_DWORD *)(v10 + 84) - *(_DWORD *)(v10 + 88));
        v18 = *(unsigned int *)(v10 + 80);
        if ( v17 < 0x20 )
          v17 = 32;
        if ( (unsigned int)v18 > v17 )
          sub_16CC920(v10 + 56);
        else
          memset(v16, -1, 8 * v18);
        v19 = *(_QWORD *)(v10 + 72);
        v20 = *(_QWORD *)(v10 + 64);
        *(_QWORD *)v10 = 0;
        if ( v20 != v19 )
          _libc_free(v19);
      }
      v21 = *(_QWORD *)(v10 + 32);
      if ( v21 )
        j_j___libc_free_0(v21, *(_QWORD *)(v10 + 48) - v21);
      v22 = *(_QWORD *)(v10 + 8);
      if ( v22 )
        j_j___libc_free_0(v22, *(_QWORD *)(v10 + 24) - v22);
      ++v9;
    }
    while ( v8 != v9 );
    if ( v71 != v72 )
      v72 = v71;
  }
  v23 = v79;
  v24 = &v79[2 * (unsigned int)v80];
  if ( v79 != v24 )
  {
    do
    {
      v25 = *v23;
      v23 += 2;
      _libc_free(v25);
    }
    while ( v23 != v24 );
  }
  LODWORD(v80) = 0;
  if ( !(_DWORD)v77 )
    goto LABEL_41;
  v44 = v76;
  v81[0] = 0;
  v45 = &v76[(unsigned int)v77];
  v46 = v76 + 1;
  v74 = *v76;
  v75 = v74 + 4096;
  if ( v45 != v76 + 1 )
  {
    do
    {
      v47 = *v46++;
      _libc_free(v47);
    }
    while ( v45 != v46 );
    v44 = v76;
  }
  LODWORD(v77) = 1;
  _libc_free(*v44);
  v48 = v79;
  v26 = &v79[2 * (unsigned int)v80];
  if ( v79 != v26 )
  {
    do
    {
      v49 = *v48;
      v48 += 2;
      _libc_free(v49);
    }
    while ( v48 != v26 );
LABEL_41:
    v26 = v79;
  }
  if ( v26 != v81 )
    _libc_free((unsigned __int64)v26);
  if ( v76 != (unsigned __int64 *)v78 )
    _libc_free((unsigned __int64)v76);
  if ( v71 )
    j_j___libc_free_0(v71, v73 - (_QWORD)v71);
  j___libc_free_0(v68);
  if ( v63 )
  {
    v27 = v61;
    v28 = &v61[2 * v63];
    do
    {
      if ( *v27 != -16 && *v27 != -8 )
      {
        v29 = v27[1];
        if ( v29 )
        {
          v30 = *(_QWORD *)(v29 + 24);
          if ( v30 )
            j_j___libc_free_0(v30, *(_QWORD *)(v29 + 40) - v30);
          j_j___libc_free_0(v29, 56);
        }
      }
      v27 += 2;
    }
    while ( v28 != v27 );
  }
  j___libc_free_0(v61);
  if ( (char *)v58[0] != &v59 )
    _libc_free(v58[0]);
}
