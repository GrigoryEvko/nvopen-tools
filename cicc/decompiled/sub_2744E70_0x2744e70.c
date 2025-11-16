// Function: sub_2744E70
// Address: 0x2744e70
//
__int64 __fastcall sub_2744E70(unsigned int a1, __int64 a2, _BYTE *a3, __int64 a4)
{
  unsigned int v4; // r12d
  __int64 v6; // rax
  unsigned __int64 *v7; // rbx
  unsigned __int64 *v8; // r13
  __int64 v10; // rbx
  __int64 v11; // rdx
  __int64 v12; // rcx
  __int64 v13; // r8
  __int64 v14; // r9
  __int64 v15; // rbx
  unsigned __int64 *v16; // r15
  unsigned __int64 *v17; // r14
  __int64 *v18; // rsi
  __int64 v19; // rdx
  __int64 v20; // rcx
  __int64 v21; // r8
  __int64 v22; // r9
  char v23; // r13
  unsigned int v24; // eax
  __int64 v25; // rdx
  char *v26; // rdx
  bool v27; // of
  __int64 v28; // r9
  __int64 v29; // r8
  char *v30; // rdx
  int v31; // eax
  char v32; // al
  unsigned int v33; // edx
  unsigned int v34; // r13d
  __int64 v35; // rax
  unsigned __int64 *v36; // rax
  char *v37; // rcx
  __int64 v38; // rdx
  char v39; // r13
  __int64 v40; // r9
  int v41; // eax
  char *v42; // rax
  char v43; // [rsp+10h] [rbp-280h]
  char v44; // [rsp+1Eh] [rbp-272h]
  char v45; // [rsp+1Fh] [rbp-271h]
  char v46; // [rsp+1Fh] [rbp-271h]
  _BYTE *v47; // [rsp+20h] [rbp-270h] BYREF
  __int64 v48; // [rsp+28h] [rbp-268h]
  _BYTE v49[64]; // [rsp+30h] [rbp-260h] BYREF
  _BYTE *v50; // [rsp+70h] [rbp-220h] BYREF
  __int64 v51; // [rsp+78h] [rbp-218h]
  _BYTE v52[64]; // [rsp+80h] [rbp-210h] BYREF
  char *v53; // [rsp+C0h] [rbp-1D0h] BYREF
  __int64 v54; // [rsp+C8h] [rbp-1C8h]
  _BYTE v55[64]; // [rsp+D0h] [rbp-1C0h] BYREF
  char *v56; // [rsp+110h] [rbp-180h] BYREF
  __int64 v57; // [rsp+118h] [rbp-178h]
  _BYTE v58[64]; // [rsp+120h] [rbp-170h] BYREF
  char *v59; // [rsp+160h] [rbp-130h] BYREF
  unsigned int v60; // [rsp+168h] [rbp-128h]
  char v61; // [rsp+170h] [rbp-120h] BYREF
  char *v62; // [rsp+1B0h] [rbp-E0h]
  int v63; // [rsp+1B8h] [rbp-D8h]
  char v64; // [rsp+1C0h] [rbp-D0h] BYREF
  unsigned __int64 *v65; // [rsp+1F0h] [rbp-A0h]
  unsigned int v66; // [rsp+1F8h] [rbp-98h]
  _BYTE v67[81]; // [rsp+200h] [rbp-90h] BYREF
  char v68; // [rsp+251h] [rbp-3Fh]
  char v69; // [rsp+252h] [rbp-3Eh]

  sub_2742E90(&v59, a4, a1, a2, a3);
  if ( !v60 || (v10 = (__int64)&v62[24 * v63], v10 != sub_27435A0((__int64)v62, v10, a4)) )
  {
    v4 = 0;
    v6 = v66;
    goto LABEL_3;
  }
  v15 = a4 + 632;
  if ( !v67[80] )
    v15 = a4;
  v16 = v65;
  v17 = &v65[10 * v66];
  if ( v65 != v17 )
  {
    do
    {
      v18 = (__int64 *)*v16;
      v19 = *((unsigned int *)v16 + 2);
      v16 += 10;
      sub_27406A0(v15, v18, v19, v12, v13, v14);
    }
    while ( v17 != v16 );
  }
  v56 = v58;
  v57 = 0x800000000LL;
  if ( v60 )
    sub_27388F0((__int64)&v56, (__int64)&v59, v11, v60, v13, v14);
  v45 = sub_30ADCF0(v15, &v56);
  if ( v56 != v58 )
    _libc_free((unsigned __int64)v56);
  if ( !v68 && !v69 )
  {
    if ( v45 )
    {
      v44 = 1;
      v23 = 1;
LABEL_28:
      LOBYTE(v4) = v44;
      v24 = v4;
      BYTE1(v24) = v23;
      v4 = v24;
      goto LABEL_65;
    }
    v37 = (char *)v60;
    v38 = (__int64)v55;
    v53 = v55;
    v54 = 0x800000000LL;
    if ( v60 )
    {
      sub_27388F0((__int64)&v53, (__int64)&v59, (__int64)v55, v60, v21, v22);
      v38 = (__int64)v53;
    }
    v27 = __OFADD__(1, (*(_QWORD *)v38)++);
    if ( v27 )
    {
      v50 = v52;
      v51 = 0x800000000LL;
    }
    else
    {
      v38 = (unsigned int)v54;
      v56 = v58;
      v57 = 0x800000000LL;
      if ( (_DWORD)v54 )
      {
        sub_27388F0((__int64)&v56, (__int64)&v53, (unsigned int)v54, (__int64)v37, v21, v22);
        v42 = v56;
        v38 = (unsigned int)v57;
        v37 = &v56[8 * (unsigned int)v57];
        if ( v56 != v37 )
        {
          do
          {
            v38 = -*(_QWORD *)v42;
            v27 = (unsigned __int128)(-1 * (__int128)*(__int64 *)v42) >> 64 != 0;
            *(_QWORD *)v42 = v38;
            if ( v27 )
            {
              v50 = v52;
              v51 = 0x800000000LL;
              goto LABEL_114;
            }
            v42 += 8;
          }
          while ( v37 != v42 );
          v38 = (unsigned int)v57;
        }
        v50 = v52;
        v51 = 0x800000000LL;
        if ( (_DWORD)v38 )
          sub_2738790((__int64)&v50, &v56, v38, (__int64)v37, v21, v22);
LABEL_114:
        if ( v56 != v58 )
          _libc_free((unsigned __int64)v56);
      }
      else
      {
        v37 = v52;
        v51 = 0x800000000LL;
        v50 = v52;
      }
    }
    if ( v53 != v55 )
      _libc_free((unsigned __int64)v53);
    if ( (_DWORD)v51 )
    {
      v56 = v58;
      v57 = 0x800000000LL;
      sub_27388F0((__int64)&v56, (__int64)&v50, v38, (__int64)v37, v21, v22);
      v46 = sub_30ADCF0(v15, &v56);
      if ( v46 )
      {
        if ( v56 != v58 )
          _libc_free((unsigned __int64)v56);
        v23 = 1;
        goto LABEL_82;
      }
      if ( v56 != v58 )
        _libc_free((unsigned __int64)v56);
    }
    v46 = 0;
    v23 = 0;
LABEL_82:
    if ( v50 != v52 )
      _libc_free((unsigned __int64)v50);
    v44 = 0;
    goto LABEL_64;
  }
  v25 = v60;
  v56 = v58;
  v57 = 0x800000000LL;
  if ( v60 )
  {
    sub_27388F0((__int64)&v56, (__int64)&v59, v60, v20, v21, v22);
    v25 = (__int64)v56;
    v31 = v57;
    v20 = (__int64)&v56[8 * (unsigned int)v57];
    if ( v56 != (char *)v20 )
    {
      do
      {
        v27 = (unsigned __int128)(-1 * (__int128)*(__int64 *)v25) >> 64 != 0;
        *(_QWORD *)v25 = -*(_QWORD *)v25;
        if ( v27 )
        {
          v47 = v49;
          v48 = 0x800000000LL;
          goto LABEL_56;
        }
        v25 += 8;
      }
      while ( v20 != v25 );
      v31 = v57;
    }
    v20 = 0x800000000LL;
    v47 = v49;
    v48 = 0x800000000LL;
    if ( v31 )
      sub_2738790((__int64)&v47, &v56, v25, 0x800000000LL, v21, v22);
LABEL_56:
    if ( v56 != v58 )
      _libc_free((unsigned __int64)v56);
  }
  else
  {
    v48 = 0x800000000LL;
    v47 = v49;
  }
  if ( !(_DWORD)v48 )
    goto LABEL_32;
  v56 = v58;
  v57 = 0x800000000LL;
  sub_27388F0((__int64)&v56, (__int64)&v47, v25, v20, v21, v22);
  v32 = sub_30ADCF0(v15, &v56);
  if ( v56 != v58 )
  {
    v43 = v32;
    _libc_free((unsigned __int64)v56);
    v32 = v43;
  }
  v46 = v32 & v45;
  if ( v46 )
  {
    v23 = 1;
    v44 = v68;
  }
  else
  {
LABEL_32:
    v26 = v55;
    v53 = v55;
    v54 = 0x800000000LL;
    if ( v60 )
    {
      sub_27388F0((__int64)&v53, (__int64)&v59, (__int64)v55, v20, v21, v22);
      v26 = v53;
    }
    v27 = __OFADD__(1, (*(_QWORD *)v26)++);
    if ( v27 )
    {
      v50 = v52;
      v51 = 0x800000000LL;
    }
    else
    {
      v56 = v58;
      v57 = 0x800000000LL;
      if ( (_DWORD)v54 )
      {
        sub_27388F0((__int64)&v56, (__int64)&v53, (__int64)v26, v20, v21, v22);
        v26 = v56;
        v41 = v57;
        v20 = (__int64)&v56[8 * (unsigned int)v57];
        if ( v56 != (char *)v20 )
        {
          do
          {
            v27 = (unsigned __int128)(-1 * (__int128)*(__int64 *)v26) >> 64 != 0;
            *(_QWORD *)v26 = -*(_QWORD *)v26;
            if ( v27 )
            {
              v50 = v52;
              v51 = 0x800000000LL;
              goto LABEL_99;
            }
            v26 += 8;
          }
          while ( (char *)v20 != v26 );
          v41 = v57;
        }
        v50 = v52;
        v20 = 0x800000000LL;
        v51 = 0x800000000LL;
        if ( v41 )
          sub_2738790((__int64)&v50, &v56, (__int64)v26, 0x800000000LL, v21, v40);
LABEL_99:
        if ( v56 != v58 )
          _libc_free((unsigned __int64)v56);
      }
      else
      {
        v20 = (__int64)v52;
        v51 = 0x800000000LL;
        v50 = v52;
      }
    }
    if ( v53 != v55 )
      _libc_free((unsigned __int64)v53);
    v28 = (unsigned int)v51;
    v46 = 0;
    if ( (_DWORD)v51 )
    {
      v56 = v58;
      v57 = 0x800000000LL;
      sub_27388F0((__int64)&v56, (__int64)&v50, (__int64)v26, v20, v21, (unsigned int)v51);
      v46 = sub_30ADCF0(v15, &v56);
      if ( v56 != v58 )
        _libc_free((unsigned __int64)v56);
    }
    v29 = v60;
    v56 = v58;
    v30 = v58;
    v57 = 0x800000000LL;
    if ( v60 )
    {
      sub_27388F0((__int64)&v56, (__int64)&v59, (__int64)v58, v20, v60, v28);
      v30 = v56;
    }
    v27 = __OFSUB__((*(_QWORD *)v30)--, 1);
    v53 = v55;
    v54 = 0x800000000LL;
    if ( !v27 && (_DWORD)v57 )
      sub_2738790((__int64)&v53, &v56, (__int64)v30, v20, v29, v28);
    if ( v56 != v58 )
      _libc_free((unsigned __int64)v56);
    if ( (_DWORD)v54 )
    {
      v56 = v58;
      v57 = 0x800000000LL;
      sub_27388F0((__int64)&v56, (__int64)&v53, (__int64)v30, v20, v29, v28);
      v39 = sub_30ADCF0(v15, &v56);
      if ( v56 != v58 )
        _libc_free((unsigned __int64)v56);
      v46 |= v39;
    }
    v23 = 0;
    if ( v46 )
    {
      v23 = 1;
      v44 = v69;
    }
    if ( v53 != v55 )
      _libc_free((unsigned __int64)v53);
    if ( v50 != v52 )
      _libc_free((unsigned __int64)v50);
  }
  if ( v47 != v49 )
    _libc_free((unsigned __int64)v47);
LABEL_64:
  v4 = 0;
  if ( v46 )
    goto LABEL_28;
LABEL_65:
  v33 = v66;
  v6 = v66;
  if ( v66 )
  {
    v34 = 0;
    do
    {
      v35 = (unsigned int)(*(_DWORD *)(v15 + 16) - 1);
      *(_DWORD *)(v15 + 16) = v35;
      v36 = (unsigned __int64 *)(*(_QWORD *)(v15 + 8) + 144 * v35);
      if ( (unsigned __int64 *)*v36 != v36 + 2 )
      {
        _libc_free(*v36);
        v33 = v66;
      }
      ++v34;
      v6 = v33;
    }
    while ( v33 > v34 );
  }
LABEL_3:
  v7 = v65;
  v8 = &v65[10 * v6];
  if ( v65 != v8 )
  {
    do
    {
      v8 -= 10;
      if ( (unsigned __int64 *)*v8 != v8 + 2 )
        _libc_free(*v8);
    }
    while ( v7 != v8 );
    v8 = v65;
  }
  if ( v8 != (unsigned __int64 *)v67 )
    _libc_free((unsigned __int64)v8);
  if ( v62 != &v64 )
    _libc_free((unsigned __int64)v62);
  if ( v59 != &v61 )
    _libc_free((unsigned __int64)v59);
  return v4;
}
