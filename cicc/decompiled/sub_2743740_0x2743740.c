// Function: sub_2743740
// Address: 0x2743740
//
void __fastcall sub_2743740(
        __int64 a1,
        unsigned int a2,
        unsigned __int8 *a3,
        _BYTE *a4,
        int a5,
        unsigned int a6,
        __int64 a7,
        bool a8)
{
  unsigned __int64 *v8; // rbx
  unsigned __int64 *v9; // r12
  __int64 v10; // rbx
  _QWORD *v11; // rax
  __int64 v12; // rcx
  unsigned __int64 v13; // r8
  __int64 *v14; // r9
  unsigned __int64 v15; // rax
  __int64 v16; // r8
  __int64 **v17; // r9
  __int64 v18; // rbx
  __int64 **v19; // r13
  __int64 **v20; // r14
  unsigned int v21; // ecx
  __int64 **v22; // rdx
  __int64 *v23; // rdi
  __int64 v24; // rax
  unsigned __int64 v25; // rdx
  __int64 *v26; // r12
  unsigned int v27; // esi
  int v28; // eax
  __int64 *v29; // rcx
  __int64 v30; // rdx
  __int64 v31; // rcx
  __int64 v32; // r8
  __int64 v33; // r9
  __int64 *v34; // r13
  __int64 v35; // rax
  unsigned __int64 v36; // r15
  unsigned int v37; // eax
  unsigned int v38; // esi
  __int64 v39; // r9
  int v40; // r14d
  _DWORD *v41; // rdi
  __int64 v42; // rcx
  __int64 v43; // rdx
  __int64 v44; // rax
  __int64 v45; // r8
  __int64 v46; // rax
  __int64 v47; // r9
  unsigned __int64 *v48; // r15
  unsigned __int64 *v49; // r14
  unsigned __int64 v50; // r8
  __int64 v51; // r9
  _QWORD *v52; // rdx
  _QWORD *v53; // rax
  _QWORD *v54; // rax
  __int64 v55; // rcx
  __int64 *v56; // r8
  unsigned __int64 v57; // r9
  unsigned __int64 v58; // rax
  unsigned __int64 *v59; // rbx
  int v60; // r11d
  int v61; // eax
  int v62; // edx
  __int64 *v63; // [rsp+28h] [rbp-340h]
  __int64 v64; // [rsp+50h] [rbp-318h]
  unsigned __int64 *v65; // [rsp+68h] [rbp-300h]
  unsigned int v66; // [rsp+70h] [rbp-2F8h] BYREF
  int v67[3]; // [rsp+74h] [rbp-2F4h] BYREF
  __int64 v68; // [rsp+80h] [rbp-2E8h] BYREF
  _BYTE *v69; // [rsp+88h] [rbp-2E0h] BYREF
  __int64 v70; // [rsp+90h] [rbp-2D8h]
  _BYTE v71[16]; // [rsp+98h] [rbp-2D0h] BYREF
  __int64 **v72; // [rsp+A8h] [rbp-2C0h] BYREF
  __int64 v73; // [rsp+B0h] [rbp-2B8h]
  _BYTE v74[48]; // [rsp+B8h] [rbp-2B0h] BYREF
  void *s; // [rsp+E8h] [rbp-280h] BYREF
  __int64 v76; // [rsp+F0h] [rbp-278h]
  _BYTE v77[64]; // [rsp+F8h] [rbp-270h] BYREF
  unsigned __int64 v78; // [rsp+138h] [rbp-230h] BYREF
  unsigned int v79; // [rsp+140h] [rbp-228h]
  char v80; // [rsp+148h] [rbp-220h] BYREF
  char *v81; // [rsp+188h] [rbp-1E0h]
  int v82; // [rsp+190h] [rbp-1D8h]
  char v83; // [rsp+198h] [rbp-1D0h] BYREF
  unsigned __int64 *v84; // [rsp+1C8h] [rbp-1A0h]
  unsigned int v85; // [rsp+1D0h] [rbp-198h]
  char v86; // [rsp+1D8h] [rbp-190h] BYREF
  unsigned __int8 v87[16]; // [rsp+228h] [rbp-140h] BYREF
  __int64 *v88; // [rsp+238h] [rbp-130h] BYREF
  __int64 v89; // [rsp+240h] [rbp-128h]
  _BYTE v90[64]; // [rsp+248h] [rbp-120h] BYREF
  _BYTE *v91; // [rsp+288h] [rbp-E0h]
  __int64 v92; // [rsp+290h] [rbp-D8h]
  _BYTE v93[48]; // [rsp+298h] [rbp-D0h] BYREF
  unsigned __int64 *v94; // [rsp+2C8h] [rbp-A0h]
  __int64 v95; // [rsp+2D0h] [rbp-98h]
  _BYTE v96[80]; // [rsp+2D8h] [rbp-90h] BYREF
  __int16 v97; // [rsp+328h] [rbp-40h]
  char v98; // [rsp+32Ah] [rbp-3Eh]

  v73 = 0x600000000LL;
  v67[0] = a5;
  v66 = a6;
  v72 = (__int64 **)v74;
  sub_2741480((__int64 *)&v78, a1, a2, a3, a4, (__int64)&v72, a8);
  if ( !v79 )
    goto LABEL_2;
  v10 = (__int64)&v81[24 * v82];
  if ( v10 != sub_27435A0((__int64)v81, v10, a1) || v87[2] )
    goto LABEL_2;
  v65 = (unsigned __int64 *)a1;
  if ( v87[0] )
    v65 = (unsigned __int64 *)(a1 + 632);
  if ( !v79 )
    goto LABEL_2;
  v11 = sub_27381E0((_QWORD *)(v78 + 8), v78 + 8LL * v79);
  if ( (_QWORD *)v12 == v11 )
    goto LABEL_2;
  v15 = v13;
  if ( *v65 >= v13 )
    v15 = *v65;
  *v65 = v15;
  if ( !(unsigned __int8)sub_27406A0((__int64)v65, v14, v13, v12, v13, (__int64)v14) )
  {
LABEL_2:
    v8 = v84;
    v9 = &v84[10 * v85];
    if ( v84 == v9 )
      goto LABEL_7;
    do
    {
      v9 -= 10;
      if ( (unsigned __int64 *)*v9 != v9 + 2 )
        _libc_free(*v9);
    }
    while ( v8 != v9 );
    goto LABEL_6;
  }
  v18 = a1 + 600;
  v69 = v71;
  v70 = 0x200000000LL;
  if ( v87[0] )
    v18 = a1 + 1232;
  v19 = v72;
  v20 = &v72[(unsigned int)v73];
  if ( v20 != v72 )
  {
    while ( 1 )
    {
      v26 = *v19;
      v27 = *(_DWORD *)(v18 + 24);
      v28 = *(_DWORD *)(v18 + 16) + 1;
      v88 = *v19;
      LODWORD(v89) = v28;
      if ( !v27 )
        break;
      v17 = *(__int64 ***)(v18 + 8);
      v21 = (v27 - 1) & (((unsigned int)v26 >> 9) ^ ((unsigned int)v26 >> 4));
      v22 = &v17[2 * v21];
      v23 = *v22;
      if ( v26 != *v22 )
      {
        v64 = *(_QWORD *)(v18 + 8);
        v60 = 1;
        v16 = 0;
        while ( v23 != (__int64 *)-4096LL )
        {
          if ( v23 != (__int64 *)-8192LL || v16 )
            v22 = (__int64 **)v16;
          v16 = (unsigned int)(v60 + 1);
          v21 = (v27 - 1) & (v60 + v21);
          v17 = (__int64 **)(v64 + 16LL * v21);
          v23 = *v17;
          if ( v26 == *v17 )
            goto LABEL_30;
          ++v60;
          v16 = (__int64)v22;
          v22 = (__int64 **)(v64 + 16LL * v21);
        }
        if ( !v16 )
          v16 = (__int64)v22;
        ++*(_QWORD *)v18;
        s = (void *)v16;
        if ( 4 * v28 < 3 * v27 )
        {
          v29 = v26;
          if ( v27 - *(_DWORD *)(v18 + 20) - v28 <= v27 >> 3 )
          {
LABEL_36:
            sub_D39D40(v18, v27);
            sub_22B1A50(v18, (__int64 *)&v88, &s);
            v29 = v88;
            v16 = (__int64)s;
            v28 = *(_DWORD *)(v18 + 16) + 1;
          }
          *(_DWORD *)(v18 + 16) = v28;
          if ( *(_QWORD *)v16 != -4096 )
            --*(_DWORD *)(v18 + 20);
          *(_QWORD *)v16 = v29;
          *(_DWORD *)(v16 + 8) = v89;
          goto LABEL_30;
        }
LABEL_35:
        v27 *= 2;
        goto LABEL_36;
      }
LABEL_30:
      v24 = (unsigned int)v70;
      v25 = (unsigned int)v70 + 1LL;
      if ( v25 > HIDWORD(v70) )
      {
        sub_C8D5F0((__int64)&v69, v71, v25, 8u, v16, (__int64)v17);
        v24 = (unsigned int)v70;
      }
      ++v19;
      *(_QWORD *)&v69[8 * v24] = v26;
      LODWORD(v70) = v70 + 1;
      if ( v20 == v19 )
        goto LABEL_40;
    }
    ++*(_QWORD *)v18;
    s = 0;
    goto LABEL_35;
  }
LABEL_40:
  sub_27393B0(a7, v67, &v66, v87, (__int64)&v69, (__int64)v17);
  if ( !v87[0] )
  {
    v34 = (__int64 *)v72;
    v63 = (__int64 *)&v72[(unsigned int)v73];
    if ( v63 != (__int64 *)v72 )
    {
      while ( 1 )
      {
        v35 = *v34;
        s = v77;
        v68 = v35;
        v36 = (unsigned int)(*(_DWORD *)(v18 + 16) + 1);
        v76 = 0x800000000LL;
        if ( (unsigned int)v36 > 8 )
          break;
        if ( v36 )
        {
          v37 = 8 * v36;
          if ( 8 * v36 )
          {
            v30 = v37;
            *(_QWORD *)&v77[v37 - 8] = 0;
            memset(v77, 0, 8LL * ((v37 - 1) >> 3));
            v31 = 0;
          }
        }
        LODWORD(v76) = v36;
        v88 = (__int64 *)v90;
        v89 = 0x800000000LL;
        if ( (_DWORD)v36 )
          goto LABEL_85;
LABEL_47:
        v98 = 0;
        v91 = v93;
        v92 = 0x200000000LL;
        v94 = (unsigned __int64 *)v96;
        v95 = 0x100000000LL;
        v97 = 0;
        if ( s != v77 )
          _libc_free((unsigned __int64)s);
        v38 = *(_DWORD *)(v18 + 24);
        if ( !v38 )
        {
          ++*(_QWORD *)v18;
          s = 0;
          goto LABEL_107;
        }
        v39 = *(_QWORD *)(v18 + 8);
        v40 = 1;
        v41 = 0;
        v42 = v68;
        LODWORD(v43) = (v38 - 1) & (((unsigned int)v68 >> 9) ^ ((unsigned int)v68 >> 4));
        v44 = v39 + 16LL * (unsigned int)v43;
        v45 = *(_QWORD *)v44;
        if ( *(_QWORD *)v44 != v68 )
        {
          while ( v45 != -4096 )
          {
            if ( v45 == -8192 && !v41 )
              v41 = (_DWORD *)v44;
            v43 = (v38 - 1) & ((_DWORD)v43 + v40);
            v44 = v39 + 16 * v43;
            v45 = *(_QWORD *)v44;
            if ( v68 == *(_QWORD *)v44 )
              goto LABEL_51;
            ++v40;
          }
          if ( !v41 )
            v41 = (_DWORD *)v44;
          v61 = *(_DWORD *)(v18 + 16);
          ++*(_QWORD *)v18;
          v62 = v61 + 1;
          s = v41;
          if ( 4 * (v61 + 1) >= 3 * v38 )
          {
LABEL_107:
            v38 *= 2;
          }
          else
          {
            v45 = v38 >> 3;
            if ( v38 - *(_DWORD *)(v18 + 20) - v62 > (unsigned int)v45 )
            {
LABEL_103:
              *(_DWORD *)(v18 + 16) = v62;
              if ( *(_QWORD *)v41 != -4096 )
                --*(_DWORD *)(v18 + 20);
              *(_QWORD *)v41 = v42;
              v46 = 0;
              v41[2] = 0;
              goto LABEL_52;
            }
          }
          sub_D39D40(v18, v38);
          sub_22B1A50(v18, &v68, &s);
          v42 = v68;
          v41 = s;
          v62 = *(_DWORD *)(v18 + 16) + 1;
          goto LABEL_103;
        }
LABEL_51:
        v46 = *(unsigned int *)(v44 + 8);
LABEL_52:
        v88[v46] = -1;
        sub_27406A0((__int64)v65, v88, (unsigned int)v89, v42, v45, v39);
        s = v77;
        v76 = 0x200000000LL;
        sub_27393B0(a7, v67, &v66, v87, (__int64)&s, v47);
        if ( s != v77 )
          _libc_free((unsigned __int64)s);
        v48 = v94;
        v32 = 10LL * (unsigned int)v95;
        v49 = &v94[v32];
        if ( v94 != &v94[v32] )
        {
          do
          {
            v49 -= 10;
            if ( (unsigned __int64 *)*v49 != v49 + 2 )
              _libc_free(*v49);
          }
          while ( v48 != v49 );
          v49 = v94;
        }
        if ( v49 != (unsigned __int64 *)v96 )
          _libc_free((unsigned __int64)v49);
        if ( v91 != v93 )
          _libc_free((unsigned __int64)v91);
        if ( v88 != (__int64 *)v90 )
          _libc_free((unsigned __int64)v88);
        if ( v63 == ++v34 )
          goto LABEL_66;
      }
      sub_C8D5F0((__int64)&s, v77, v36, 8u, v32 * 8, v33);
      memset(s, 0, 8 * v36);
      LODWORD(v76) = v36;
      v88 = (__int64 *)v90;
      v89 = 0x800000000LL;
LABEL_85:
      sub_2738790((__int64)&v88, (char **)&s, v30, v31, v32 * 8, v33);
      goto LABEL_47;
    }
  }
LABEL_66:
  if ( v87[1] )
  {
    v50 = v78;
    v51 = v79;
    v52 = (_QWORD *)(v78 + 8LL * v79);
    if ( v52 != (_QWORD *)v78 )
    {
      v53 = (_QWORD *)v78;
      do
      {
        *v53 = -*v53;
        ++v53;
      }
      while ( v53 != v52 );
      v50 = v78;
      v51 = v79;
    }
    v54 = sub_27381E0((_QWORD *)(v50 + 8), v50 + 8 * v51);
    if ( (_QWORD *)v55 != v54 )
    {
      v58 = v57;
      if ( *v65 >= v57 )
        v58 = *v65;
      *v65 = v58;
      sub_27406A0((__int64)v65, v56, v57, v55, (__int64)v56, v57);
    }
    v88 = (__int64 *)v90;
    v89 = 0x200000000LL;
    sub_27393B0(a7, v67, &v66, v87, (__int64)&v88, v57);
    if ( v88 != (__int64 *)v90 )
      _libc_free((unsigned __int64)v88);
  }
  if ( v69 != v71 )
    _libc_free((unsigned __int64)v69);
  v59 = v84;
  v9 = &v84[10 * v85];
  if ( v84 != v9 )
  {
    do
    {
      v9 -= 10;
      if ( (unsigned __int64 *)*v9 != v9 + 2 )
        _libc_free(*v9);
    }
    while ( v59 != v9 );
LABEL_6:
    v9 = v84;
  }
LABEL_7:
  if ( v9 != (unsigned __int64 *)&v86 )
    _libc_free((unsigned __int64)v9);
  if ( v81 != &v83 )
    _libc_free((unsigned __int64)v81);
  if ( (char *)v78 != &v80 )
    _libc_free(v78);
  if ( v72 != (__int64 **)v74 )
    _libc_free((unsigned __int64)v72);
}
