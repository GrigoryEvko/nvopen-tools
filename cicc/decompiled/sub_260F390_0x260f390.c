// Function: sub_260F390
// Address: 0x260f390
//
_QWORD *__fastcall sub_260F390(_QWORD *a1, __int64 a2, __int64 *a3, __int64 a4)
{
  __int64 v6; // rax
  __int64 *v7; // r13
  __int64 *v8; // rbx
  unsigned __int64 i; // rdx
  __int64 v10; // rdi
  unsigned int v11; // ecx
  __int64 v12; // rsi
  __int64 *v13; // rbx
  __int64 *v14; // r15
  __int64 v15; // rsi
  __int64 v16; // rdi
  __int64 *v17; // r13
  __int64 *v18; // rbx
  unsigned __int64 j; // rdx
  __int64 v20; // rdi
  unsigned int v21; // ecx
  __int64 v22; // rsi
  __int64 *v23; // rbx
  __int64 *v24; // r15
  __int64 v25; // rsi
  __int64 v26; // rdi
  __int64 *v27; // r13
  __int64 *v28; // rbx
  unsigned __int64 k; // rdx
  __int64 v30; // rdi
  unsigned int v31; // ecx
  __int64 v32; // rsi
  __int64 *v33; // rbx
  __int64 *v34; // r15
  __int64 v35; // rsi
  __int64 v36; // rdi
  _QWORD *v37; // rsi
  _QWORD *v38; // rdx
  unsigned __int64 v39; // r14
  unsigned __int64 v40; // r15
  bool v42; // [rsp+4Fh] [rbp-241h]
  unsigned __int64 v43; // [rsp+58h] [rbp-238h] BYREF
  _QWORD v44[2]; // [rsp+60h] [rbp-230h] BYREF
  __int64 (__fastcall *v45)(_QWORD *, _QWORD *, int); // [rsp+70h] [rbp-220h]
  __int64 (__fastcall *v46)(__int64 *, __int64); // [rsp+78h] [rbp-218h]
  _QWORD v47[2]; // [rsp+80h] [rbp-210h] BYREF
  __int64 (__fastcall *v48)(_QWORD *, _QWORD *, int); // [rsp+90h] [rbp-200h]
  __int64 (__fastcall *v49)(__int64 *, __int64); // [rsp+98h] [rbp-1F8h]
  _QWORD v50[2]; // [rsp+A0h] [rbp-1F0h] BYREF
  __int64 (__fastcall *v51)(_QWORD *, _QWORD *, int); // [rsp+B0h] [rbp-1E0h]
  unsigned __int64 (__fastcall *v52)(unsigned __int64 **, __int64); // [rsp+B8h] [rbp-1D8h]
  _QWORD v53[2]; // [rsp+C0h] [rbp-1D0h] BYREF
  __int64 v54; // [rsp+D0h] [rbp-1C0h]
  __int64 v55; // [rsp+D8h] [rbp-1B8h]
  __int64 v56; // [rsp+E0h] [rbp-1B0h]
  __int64 (__fastcall *v57)(__int64, __int64, __int64); // [rsp+E8h] [rbp-1A8h]
  _QWORD *v58; // [rsp+F0h] [rbp-1A0h]
  __int64 v59; // [rsp+F8h] [rbp-198h]
  __int64 v60; // [rsp+100h] [rbp-190h]
  __int64 v61; // [rsp+108h] [rbp-188h]
  unsigned int v62; // [rsp+110h] [rbp-180h]
  __int64 (__fastcall *v63)(__int64, __int64, __int64); // [rsp+118h] [rbp-178h]
  _QWORD *v64; // [rsp+120h] [rbp-170h]
  __int64 (__fastcall *v65)(); // [rsp+128h] [rbp-168h]
  _QWORD *v66; // [rsp+130h] [rbp-160h]
  _QWORD v67[2]; // [rsp+138h] [rbp-158h] BYREF
  __int64 *v68; // [rsp+148h] [rbp-148h]
  __int64 v69; // [rsp+150h] [rbp-140h]
  _BYTE v70[32]; // [rsp+158h] [rbp-138h] BYREF
  __int64 *v71; // [rsp+178h] [rbp-118h]
  __int64 v72; // [rsp+180h] [rbp-110h]
  _QWORD v73[2]; // [rsp+188h] [rbp-108h] BYREF
  _QWORD v74[2]; // [rsp+198h] [rbp-F8h] BYREF
  __int64 *v75; // [rsp+1A8h] [rbp-E8h]
  __int64 v76; // [rsp+1B0h] [rbp-E0h]
  _BYTE v77[32]; // [rsp+1B8h] [rbp-D8h] BYREF
  __int64 *v78; // [rsp+1D8h] [rbp-B8h]
  __int64 v79; // [rsp+1E0h] [rbp-B0h]
  _QWORD v80[2]; // [rsp+1E8h] [rbp-A8h] BYREF
  _QWORD v81[2]; // [rsp+1F8h] [rbp-98h] BYREF
  __int64 *v82; // [rsp+208h] [rbp-88h]
  __int64 v83; // [rsp+210h] [rbp-80h]
  _BYTE v84[32]; // [rsp+218h] [rbp-78h] BYREF
  __int64 *v85; // [rsp+238h] [rbp-58h]
  __int64 v86; // [rsp+240h] [rbp-50h]
  _QWORD v87[2]; // [rsp+248h] [rbp-48h] BYREF
  int v88; // [rsp+258h] [rbp-38h]

  v47[0] = a4;
  v6 = *(_QWORD *)(sub_BC0510(a4, &unk_4F82418, (__int64)a3) + 8);
  v66 = v50;
  v43 = 0;
  v44[0] = v6;
  v46 = sub_25F5AB0;
  v45 = sub_25F5B10;
  v49 = sub_25F5A90;
  v48 = sub_25F5B40;
  v50[0] = &v43;
  v52 = sub_25F6140;
  v51 = sub_25F5B70;
  v53[0] = 256;
  v57 = sub_25EF9D0;
  v58 = v44;
  v53[1] = 0;
  v63 = sub_25F5AD0;
  v64 = v47;
  v54 = 0;
  v65 = sub_25F5AF0;
  v55 = 0;
  v56 = 0;
  v59 = 0;
  v60 = 0;
  v61 = 0;
  v62 = 0;
  v67[0] = 0;
  v67[1] = 0;
  v71 = v73;
  v68 = (__int64 *)v70;
  v75 = (__int64 *)v77;
  v69 = 0x400000000LL;
  v76 = 0x400000000LL;
  v78 = v80;
  v83 = 0x400000000LL;
  v82 = (__int64 *)v84;
  v72 = 0;
  v73[0] = 0;
  v73[1] = 0;
  v74[0] = 0;
  v74[1] = 0;
  v79 = 0;
  v80[0] = 0;
  v80[1] = 0;
  v81[0] = 0;
  v81[1] = 0;
  v85 = v87;
  v86 = 0;
  v87[0] = 0;
  v87[1] = 0;
  v88 = 256;
  v42 = sub_260F360(v53, a3);
  sub_22B0CF0((__int64)v81);
  v7 = v82;
  v8 = &v82[(unsigned int)v83];
  if ( v82 != v8 )
  {
    for ( i = (unsigned __int64)v82; ; i = (unsigned __int64)v82 )
    {
      v10 = *v7;
      v11 = (unsigned int)((__int64)((__int64)v7 - i) >> 3) >> 7;
      v12 = 4096LL << v11;
      if ( v11 >= 0x1E )
        v12 = 0x40000000000LL;
      ++v7;
      sub_C7D6A0(v10, v12, 16);
      if ( v8 == v7 )
        break;
    }
  }
  v13 = v85;
  v14 = &v85[2 * (unsigned int)v86];
  if ( v85 != v14 )
  {
    do
    {
      v15 = v13[1];
      v16 = *v13;
      v13 += 2;
      sub_C7D6A0(v16, v15, 16);
    }
    while ( v14 != v13 );
    v14 = v85;
  }
  if ( v14 != v87 )
    _libc_free((unsigned __int64)v14);
  if ( v82 != (__int64 *)v84 )
    _libc_free((unsigned __int64)v82);
  sub_25FD180((__int64)v74);
  v17 = v75;
  v18 = &v75[(unsigned int)v76];
  if ( v75 != v18 )
  {
    for ( j = (unsigned __int64)v75; ; j = (unsigned __int64)v75 )
    {
      v20 = *v17;
      v21 = (unsigned int)((__int64)((__int64)v17 - j) >> 3) >> 7;
      v22 = 4096LL << v21;
      if ( v21 >= 0x1E )
        v22 = 0x40000000000LL;
      ++v17;
      sub_C7D6A0(v20, v22, 16);
      if ( v18 == v17 )
        break;
    }
  }
  v23 = v78;
  v24 = &v78[2 * (unsigned int)v79];
  if ( v78 != v24 )
  {
    do
    {
      v25 = v23[1];
      v26 = *v23;
      v23 += 2;
      sub_C7D6A0(v26, v25, 16);
    }
    while ( v24 != v23 );
    v24 = v78;
  }
  if ( v24 != v80 )
    _libc_free((unsigned __int64)v24);
  if ( v75 != (__int64 *)v77 )
    _libc_free((unsigned __int64)v75);
  sub_25FCE60((__int64)v67);
  v27 = v68;
  v28 = &v68[(unsigned int)v69];
  if ( v68 != v28 )
  {
    for ( k = (unsigned __int64)v68; ; k = (unsigned __int64)v68 )
    {
      v30 = *v27;
      v31 = (unsigned int)((__int64)((__int64)v27 - k) >> 3) >> 7;
      v32 = 4096LL << v31;
      if ( v31 >= 0x1E )
        v32 = 0x40000000000LL;
      ++v27;
      sub_C7D6A0(v30, v32, 16);
      if ( v28 == v27 )
        break;
    }
  }
  v33 = v71;
  v34 = &v71[2 * (unsigned int)v72];
  if ( v71 != v34 )
  {
    do
    {
      v35 = v33[1];
      v36 = *v33;
      v33 += 2;
      sub_C7D6A0(v36, v35, 16);
    }
    while ( v34 != v33 );
    v34 = v71;
  }
  if ( v34 != v73 )
    _libc_free((unsigned __int64)v34);
  if ( v68 != (__int64 *)v70 )
    _libc_free((unsigned __int64)v68);
  sub_C7D6A0(v60, 16LL * v62, 8);
  sub_C7D6A0(v54, 4LL * (unsigned int)v56, 4);
  v37 = a1 + 4;
  v38 = a1 + 10;
  if ( v42 )
  {
    memset(a1, 0, 0x60u);
    a1[1] = v37;
    *((_DWORD *)a1 + 4) = 2;
    *((_BYTE *)a1 + 28) = 1;
    a1[7] = v38;
    *((_DWORD *)a1 + 16) = 2;
    *((_BYTE *)a1 + 76) = 1;
  }
  else
  {
    a1[1] = v37;
    a1[2] = 0x100000002LL;
    a1[6] = 0;
    a1[7] = v38;
    a1[8] = 2;
    *((_DWORD *)a1 + 18) = 0;
    *((_BYTE *)a1 + 76) = 1;
    *((_DWORD *)a1 + 6) = 0;
    *((_BYTE *)a1 + 28) = 1;
    a1[4] = &qword_4F82400;
    *a1 = 1;
  }
  if ( v51 )
    v51(v50, v50, 3);
  v39 = v43;
  if ( v43 )
  {
    v40 = *(_QWORD *)(v43 + 16);
    if ( v40 )
    {
      sub_FDC110(*(__int64 **)(v43 + 16));
      j_j___libc_free_0(v40);
    }
    j_j___libc_free_0(v39);
  }
  if ( v48 )
    v48(v47, v47, 3);
  if ( v45 )
    v45(v44, v44, 3);
  return a1;
}
