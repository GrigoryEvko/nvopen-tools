// Function: sub_25DBB60
// Address: 0x25dbb60
//
__int64 __fastcall sub_25DBB60(__int64 a1, _QWORD *a2)
{
  int v3; // r11d
  _QWORD *v4; // rdx
  unsigned int v5; // r8d
  _QWORD *v6; // rax
  void *v7; // rdi
  __int64 *v8; // r13
  _QWORD *v9; // rdx
  unsigned int v10; // r12d
  unsigned int v11; // eax
  _QWORD **v12; // r14
  _QWORD **i; // r13
  __int64 v14; // rax
  _QWORD *v15; // rbx
  unsigned __int64 v16; // r15
  __int64 v17; // rdi
  unsigned int v18; // eax
  _QWORD *v19; // rbx
  _QWORD *v20; // r13
  __int64 v21; // rdi
  unsigned int v22; // eax
  _QWORD **v23; // r14
  _QWORD **j; // r13
  __int64 v25; // rax
  _QWORD *v26; // rbx
  unsigned __int64 v27; // r15
  __int64 v28; // rdi
  unsigned int v29; // eax
  _QWORD *v30; // rbx
  _QWORD *v31; // r13
  __int64 v32; // rdi
  __int64 **v34; // rax
  __int64 **v35; // rdx
  int v36; // edi
  _QWORD *v37; // rax
  __int64 v38; // rdi
  unsigned int v39; // eax
  void *v40; // r15
  int v41; // r9d
  _QWORD *v42; // r8
  _QWORD *v43; // rsi
  __int64 v44; // r15
  int v45; // r8d
  void *v46; // r9
  _BYTE v48[8]; // [rsp+10h] [rbp-150h] BYREF
  _QWORD *v49; // [rsp+18h] [rbp-148h]
  unsigned int v50; // [rsp+28h] [rbp-138h]
  __int64 v51; // [rsp+38h] [rbp-128h]
  unsigned int v52; // [rsp+48h] [rbp-118h]
  __int64 v53; // [rsp+58h] [rbp-108h]
  unsigned int v54; // [rsp+68h] [rbp-F8h]
  __int64 v55; // [rsp+70h] [rbp-F0h] BYREF
  _QWORD *v56; // [rsp+78h] [rbp-E8h]
  int v57; // [rsp+80h] [rbp-E0h]
  int v58; // [rsp+84h] [rbp-DCh]
  unsigned int v59; // [rsp+88h] [rbp-D8h]
  __int64 v60; // [rsp+98h] [rbp-C8h]
  unsigned int v61; // [rsp+A8h] [rbp-B8h]
  __int64 v62; // [rsp+B8h] [rbp-A8h]
  unsigned int v63; // [rsp+C8h] [rbp-98h]
  __int64 v64; // [rsp+D0h] [rbp-90h] BYREF
  __int64 **v65; // [rsp+D8h] [rbp-88h]
  int v66; // [rsp+E4h] [rbp-7Ch]
  unsigned __int8 v67; // [rsp+ECh] [rbp-74h]
  unsigned __int64 v68; // [rsp+108h] [rbp-58h]
  int v69; // [rsp+114h] [rbp-4Ch]
  int v70; // [rsp+118h] [rbp-48h]
  char v71; // [rsp+11Ch] [rbp-44h]

  sub_BBB200((__int64)v48);
  sub_BBB1A0((__int64)&v55);
  if ( !v59 )
  {
    ++v55;
    goto LABEL_74;
  }
  v3 = 1;
  v4 = 0;
  v5 = (v59 - 1) & (((unsigned int)&unk_4F82418 >> 9) ^ ((unsigned int)&unk_4F82418 >> 4));
  v6 = &v56[2 * v5];
  v7 = (void *)*v6;
  if ( (_UNKNOWN *)*v6 == &unk_4F82418 )
  {
LABEL_3:
    v8 = v6 + 1;
    if ( v6[1] )
      goto LABEL_4;
    goto LABEL_68;
  }
  while ( v7 != (void *)-4096LL )
  {
    if ( !v4 && v7 == (void *)-8192LL )
      v4 = v6;
    v5 = (v59 - 1) & (v3 + v5);
    v6 = &v56[2 * v5];
    v7 = (void *)*v6;
    if ( (_UNKNOWN *)*v6 == &unk_4F82418 )
      goto LABEL_3;
    ++v3;
  }
  if ( !v4 )
    v4 = v6;
  ++v55;
  v36 = v57 + 1;
  if ( 4 * (v57 + 1) >= 3 * v59 )
  {
LABEL_74:
    sub_23622E0((__int64)&v55, 2 * v59);
    if ( v59 )
    {
      v36 = v57 + 1;
      v39 = (v59 - 1) & (((unsigned int)&unk_4F82418 >> 9) ^ ((unsigned int)&unk_4F82418 >> 4));
      v4 = &v56[2 * v39];
      v40 = (void *)*v4;
      if ( (_UNKNOWN *)*v4 != &unk_4F82418 )
      {
        v41 = 1;
        v42 = 0;
        while ( v40 != (void *)-4096LL )
        {
          if ( !v42 && v40 == (void *)-8192LL )
            v42 = v4;
          v39 = (v59 - 1) & (v41 + v39);
          v4 = &v56[2 * v39];
          v40 = (void *)*v4;
          if ( (_UNKNOWN *)*v4 == &unk_4F82418 )
            goto LABEL_65;
          ++v41;
        }
        if ( v42 )
          v4 = v42;
      }
      goto LABEL_65;
    }
    goto LABEL_97;
  }
  if ( v59 - v58 - v36 <= v59 >> 3 )
  {
    sub_23622E0((__int64)&v55, v59);
    if ( v59 )
    {
      v43 = 0;
      LODWORD(v44) = (v59 - 1) & (((unsigned int)&unk_4F82418 >> 9) ^ ((unsigned int)&unk_4F82418 >> 4));
      v45 = 1;
      v36 = v57 + 1;
      v4 = &v56[2 * (unsigned int)v44];
      v46 = (void *)*v4;
      if ( (_UNKNOWN *)*v4 != &unk_4F82418 )
      {
        while ( v46 != (void *)-4096LL )
        {
          if ( v46 == (void *)-8192LL && !v43 )
            v43 = v4;
          v44 = (v59 - 1) & ((_DWORD)v44 + v45);
          v4 = &v56[2 * v44];
          v46 = (void *)*v4;
          if ( (_UNKNOWN *)*v4 == &unk_4F82418 )
            goto LABEL_65;
          ++v45;
        }
        if ( v43 )
          v4 = v43;
      }
      goto LABEL_65;
    }
LABEL_97:
    ++v57;
    BUG();
  }
LABEL_65:
  v57 = v36;
  if ( *v4 != -4096 )
    --v58;
  v4[1] = 0;
  *v4 = &unk_4F82418;
  v8 = v4 + 1;
LABEL_68:
  v37 = (_QWORD *)sub_22077B0(0x10u);
  if ( v37 )
  {
    v37[1] = v48;
    *v37 = &unk_4A156F8;
  }
  v38 = *v8;
  *v8 = (__int64)v37;
  if ( v38 )
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v38 + 8LL))(v38);
LABEL_4:
  v9 = a2;
  v10 = 1;
  sub_25DA480(&v64, a1 + 176, v9);
  if ( v69 != v70 )
    goto LABEL_5;
  v10 = v67;
  if ( !v67 )
  {
    LOBYTE(v10) = sub_C8CA60((__int64)&v64, (__int64)&qword_4F82400) == 0;
LABEL_5:
    if ( v71 )
      goto LABEL_6;
    goto LABEL_54;
  }
  v34 = v65;
  v35 = &v65[v66];
  if ( v65 != v35 )
  {
    while ( *v34 != &qword_4F82400 )
    {
      if ( v35 == ++v34 )
        goto LABEL_53;
    }
    v10 = 0;
  }
LABEL_53:
  if ( !v71 )
  {
LABEL_54:
    _libc_free(v68);
LABEL_6:
    if ( !v67 )
      _libc_free((unsigned __int64)v65);
  }
  sub_C7D6A0(v62, 24LL * v63, 8);
  v11 = v61;
  if ( v61 )
  {
    v12 = (_QWORD **)(v60 + 32LL * v61);
    for ( i = (_QWORD **)(v60 + 8); ; i += 4 )
    {
      v14 = (__int64)*(i - 1);
      if ( v14 != -8192 && v14 != -4096 )
      {
        v15 = *i;
        while ( v15 != i )
        {
          v16 = (unsigned __int64)v15;
          v15 = (_QWORD *)*v15;
          v17 = *(_QWORD *)(v16 + 24);
          if ( v17 )
            (*(void (__fastcall **)(__int64))(*(_QWORD *)v17 + 8LL))(v17);
          j_j___libc_free_0(v16);
        }
      }
      if ( v12 == i + 3 )
        break;
    }
    v11 = v61;
  }
  sub_C7D6A0(v60, 32LL * v11, 8);
  v18 = v59;
  if ( v59 )
  {
    v19 = v56;
    v20 = &v56[2 * v59];
    do
    {
      if ( *v19 != -4096 && *v19 != -8192 )
      {
        v21 = v19[1];
        if ( v21 )
          (*(void (__fastcall **)(__int64))(*(_QWORD *)v21 + 8LL))(v21);
      }
      v19 += 2;
    }
    while ( v20 != v19 );
    v18 = v59;
  }
  sub_C7D6A0((__int64)v56, 16LL * v18, 8);
  sub_C7D6A0(v53, 24LL * v54, 8);
  v22 = v52;
  if ( v52 )
  {
    v23 = (_QWORD **)(v51 + 32LL * v52);
    for ( j = (_QWORD **)(v51 + 8); ; j += 4 )
    {
      v25 = (__int64)*(j - 1);
      if ( v25 != -4096 && v25 != -8192 )
      {
        v26 = *j;
        while ( v26 != j )
        {
          v27 = (unsigned __int64)v26;
          v26 = (_QWORD *)*v26;
          v28 = *(_QWORD *)(v27 + 24);
          if ( v28 )
            (*(void (__fastcall **)(__int64))(*(_QWORD *)v28 + 8LL))(v28);
          j_j___libc_free_0(v27);
        }
      }
      if ( v23 == j + 3 )
        break;
    }
    v22 = v52;
  }
  sub_C7D6A0(v51, 32LL * v22, 8);
  v29 = v50;
  if ( v50 )
  {
    v30 = v49;
    v31 = &v49[2 * v50];
    do
    {
      if ( *v30 != -4096 && *v30 != -8192 )
      {
        v32 = v30[1];
        if ( v32 )
          (*(void (__fastcall **)(__int64))(*(_QWORD *)v32 + 8LL))(v32);
      }
      v30 += 2;
    }
    while ( v31 != v30 );
    v29 = v50;
  }
  sub_C7D6A0((__int64)v49, 16LL * v29, 8);
  return v10;
}
