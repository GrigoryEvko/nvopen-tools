// Function: sub_F3FEF0
// Address: 0xf3fef0
//
__int64 __fastcall sub_F3FEF0(
        __int64 a1,
        __int64 **a2,
        __int64 a3,
        _BYTE *a4,
        _BYTE *a5,
        __int64 a6,
        __int64 a7,
        __int64 a8,
        __int64 a9,
        __int64 *a10,
        char a11)
{
  __int64 v14; // rbx
  __int64 *v15; // rax
  __int64 v16; // rdx
  __int64 v17; // r12
  __int64 v18; // r8
  __int64 v19; // r9
  __int64 v20; // r14
  __int64 v21; // rax
  __int64 *v22; // r12
  unsigned __int16 v23; // bx
  _QWORD *v24; // rax
  __int64 v25; // rax
  __int64 *v26; // rsi
  __int64 **v27; // r12
  __int64 **v28; // rbx
  __int64 **i; // r12
  unsigned __int64 v30; // rdi
  int v31; // eax
  __int64 v32; // rdi
  __int64 v33; // r8
  __int64 v34; // r9
  __int64 v35; // rdx
  __int64 v36; // rcx
  __int64 v37; // rbx
  __int64 v38; // rax
  __int64 v39; // r12
  __int64 v40; // r12
  __int64 v41; // r13
  unsigned __int8 *v42; // r15
  bool v43; // zf
  __int16 v44; // dx
  unsigned __int64 *v45; // r8
  unsigned __int8 v46; // al
  char v47; // dl
  __int64 v48; // rcx
  __int16 v49; // dx
  unsigned __int64 *v50; // r8
  unsigned __int8 v51; // al
  char v52; // dl
  __int64 v53; // rsi
  __int64 v54; // rcx
  __int64 v55; // rax
  __int64 v56; // rbx
  __int64 result; // rax
  __int64 v58; // rsi
  unsigned __int8 *v59; // rsi
  __int64 v60; // rdx
  __int64 v61; // rbx
  __int64 v62; // rax
  __int64 v63; // r8
  __int64 v64; // r9
  __int64 v65; // rax
  const char *v66; // r15
  unsigned __int16 v67; // bx
  _QWORD *v68; // rax
  __int64 v69; // rax
  const char *v70; // rsi
  const char **v71; // r15
  __int64 **v72; // rdx
  __int64 v73; // rcx
  __int64 *v74; // rbx
  __int64 *v75; // r15
  __int64 v76; // rdi
  unsigned __int64 v77; // rax
  __int64 v78; // rsi
  unsigned __int8 *v79; // rsi
  __int64 v83; // [rsp+20h] [rbp-E0h]
  __int64 v85; // [rsp+30h] [rbp-D0h]
  unsigned __int8 *v86; // [rsp+30h] [rbp-D0h]
  __int64 v87; // [rsp+30h] [rbp-D0h]
  __int64 v88; // [rsp+30h] [rbp-D0h]
  char v89; // [rsp+4Fh] [rbp-B1h] BYREF
  char *v90; // [rsp+50h] [rbp-B0h] BYREF
  __int64 v91; // [rsp+58h] [rbp-A8h]
  _BYTE *v92; // [rsp+60h] [rbp-A0h]
  __int16 v93; // [rsp+70h] [rbp-90h]
  __int64 *v94; // [rsp+80h] [rbp-80h] BYREF
  __int64 j; // [rsp+88h] [rbp-78h]
  _QWORD v96[2]; // [rsp+90h] [rbp-70h] BYREF
  __int16 v97; // [rsp+A0h] [rbp-60h]

  v14 = *(_QWORD *)(a1 + 72);
  v15 = (__int64 *)sub_BD5D20(a1);
  v96[0] = a4;
  v97 = 773;
  j = v16;
  v94 = v15;
  v17 = sub_AA48A0(a1);
  v20 = sub_22077B0(80);
  if ( v20 )
    sub_AA4D50(v20, v17, (__int64)&v94, v14, a1);
  v21 = *(unsigned int *)(a6 + 8);
  if ( v21 + 1 > (unsigned __int64)*(unsigned int *)(a6 + 12) )
  {
    sub_C8D5F0(a6, (const void *)(a6 + 16), v21 + 1, 8u, v18, v19);
    v21 = *(unsigned int *)(a6 + 8);
  }
  *(_QWORD *)(*(_QWORD *)a6 + 8 * v21) = v20;
  ++*(_DWORD *)(a6 + 8);
  sub_B43C20((__int64)&v94, v20);
  v22 = v94;
  v23 = j;
  v24 = sub_BD2C40(72, 1u);
  v85 = (__int64)v24;
  if ( v24 )
    sub_B4C8F0((__int64)v24, a1, 1u, (__int64)v22, v23);
  v25 = sub_AA4FF0(a1);
  if ( !v25 )
    BUG();
  v26 = *(__int64 **)(v25 + 24);
  v94 = v26;
  v27 = (__int64 **)(v85 + 48);
  if ( v26 )
  {
    sub_B96E90((__int64)&v94, (__int64)v26, 1);
    if ( v27 == &v94 )
    {
      if ( v94 )
        sub_B91220((__int64)&v94, (__int64)v94);
      goto LABEL_12;
    }
    v58 = *(_QWORD *)(v85 + 48);
    if ( !v58 )
    {
LABEL_54:
      v59 = (unsigned __int8 *)v94;
      *(_QWORD *)(v85 + 48) = v94;
      if ( v59 )
        sub_B976B0((__int64)&v94, v59, (__int64)v27);
      goto LABEL_12;
    }
LABEL_53:
    sub_B91220((__int64)v27, v58);
    goto LABEL_54;
  }
  if ( v27 != &v94 )
  {
    v58 = *(_QWORD *)(v85 + 48);
    if ( v58 )
      goto LABEL_53;
  }
LABEL_12:
  v28 = &a2[a3];
  for ( i = a2; v28 != i; ++i )
  {
    v30 = (*i)[6] & 0xFFFFFFFFFFFFFFF8LL;
    if ( (__int64 *)v30 == *i + 6 )
    {
      v32 = 0;
    }
    else
    {
      if ( !v30 )
        BUG();
      v31 = *(unsigned __int8 *)(v30 - 24);
      v32 = v30 - 24;
      if ( (unsigned int)(v31 - 30) >= 0xB )
        v32 = 0;
    }
    sub_BD2ED0(v32, a1, v20);
  }
  v89 = 0;
  sub_F3F350(a1, v20, a2, a3, a7, a8, a9, a10, a11, &v89);
  sub_F33910(a1, v20, (__int64 *)a2, a3, v85, v89);
  v35 = *(_QWORD *)(a1 + 16);
  v94 = v96;
  for ( j = 0x800000000LL; v35; v35 = *(_QWORD *)(v35 + 8) )
  {
    if ( (unsigned __int8)(**(_BYTE **)(v35 + 24) - 30) <= 0xAu )
      break;
  }
  v36 = 0;
LABEL_21:
  if ( v35 )
  {
    do
    {
      v37 = *(_QWORD *)(v35 + 8);
      if ( v37 )
      {
        do
        {
          if ( (unsigned __int8)(**(_BYTE **)(v37 + 24) - 30) <= 0xAu )
            break;
          v37 = *(_QWORD *)(v37 + 8);
        }
        while ( v37 );
        v38 = *(_QWORD *)(v35 + 24);
        v35 = v37;
        v39 = *(_QWORD *)(v38 + 40);
        if ( v39 == v20 )
          goto LABEL_21;
      }
      else
      {
        v39 = *(_QWORD *)(*(_QWORD *)(v35 + 24) + 40LL);
        if ( v20 == v39 )
          break;
      }
      if ( v36 + 1 > (unsigned __int64)HIDWORD(j) )
      {
        sub_C8D5F0((__int64)&v94, v96, v36 + 1, 8u, v33, v34);
        v36 = (unsigned int)j;
      }
      v35 = v37;
      v94[v36] = v39;
      v36 = (unsigned int)(j + 1);
      LODWORD(j) = j + 1;
    }
    while ( v37 );
  }
  v40 = 0;
  if ( (_DWORD)v36 )
  {
    v87 = *(_QWORD *)(a1 + 72);
    v90 = (char *)sub_BD5D20(a1);
    v93 = 773;
    v91 = v60;
    v92 = a5;
    v61 = sub_AA48A0(a1);
    v62 = sub_22077B0(80);
    v40 = v62;
    if ( v62 )
      sub_AA4D50(v62, v61, (__int64)&v90, v87, a1);
    v65 = *(unsigned int *)(a6 + 8);
    if ( v65 + 1 > (unsigned __int64)*(unsigned int *)(a6 + 12) )
    {
      sub_C8D5F0(a6, (const void *)(a6 + 16), v65 + 1, 8u, v63, v64);
      v65 = *(unsigned int *)(a6 + 8);
    }
    *(_QWORD *)(*(_QWORD *)a6 + 8 * v65) = v40;
    ++*(_DWORD *)(a6 + 8);
    sub_B43C20((__int64)&v90, v40);
    v66 = v90;
    v67 = v91;
    v68 = sub_BD2C40(72, 1u);
    v88 = (__int64)v68;
    if ( v68 )
      sub_B4C8F0((__int64)v68, a1, 1u, (__int64)v66, v67);
    v69 = sub_AA4FF0(a1);
    if ( !v69 )
      BUG();
    v70 = *(const char **)(v69 + 24);
    v90 = (char *)v70;
    v71 = (const char **)(v88 + 48);
    if ( v70 )
    {
      sub_B96E90((__int64)&v90, (__int64)v70, 1);
      if ( v71 == (const char **)&v90 )
      {
        if ( v90 )
          sub_B91220((__int64)&v90, (__int64)v90);
        goto LABEL_72;
      }
      v78 = *(_QWORD *)(v88 + 48);
      if ( !v78 )
      {
LABEL_83:
        v79 = (unsigned __int8 *)v90;
        *(_QWORD *)(v88 + 48) = v90;
        if ( v79 )
          sub_B976B0((__int64)&v90, v79, (__int64)v71);
        goto LABEL_72;
      }
    }
    else if ( v71 == (const char **)&v90 || (v78 = *(_QWORD *)(v88 + 48)) == 0 )
    {
LABEL_72:
      v72 = (__int64 **)v94;
      v73 = (unsigned int)j;
      v74 = &v94[(unsigned int)j];
      if ( v74 != v94 )
      {
        v75 = v94;
        do
        {
          v76 = *v75++;
          v77 = sub_986580(v76);
          sub_BD2ED0(v77, a1, v40);
        }
        while ( v74 != v75 );
        v72 = (__int64 **)v94;
        v73 = (unsigned int)j;
      }
      v89 = 0;
      sub_F3F350(a1, v40, v72, v73, a7, a8, a9, a10, a11, &v89);
      sub_F33910(a1, v40, v94, (unsigned int)j, v88, v89);
      goto LABEL_29;
    }
    sub_B91220((__int64)v71, v78);
    goto LABEL_83;
  }
LABEL_29:
  v41 = sub_AA5EB0(a1);
  v42 = (unsigned __int8 *)sub_B47F80((_BYTE *)v41);
  v43 = *a4 == 0;
  v90 = "lpad";
  if ( v43 )
  {
    v93 = 259;
  }
  else
  {
    v92 = a4;
    v93 = 771;
  }
  sub_BD6B50(v42, (const char **)&v90);
  v45 = (unsigned __int64 *)sub_AA5190(v20);
  if ( v45 )
  {
    v46 = v44;
    v47 = HIBYTE(v44);
  }
  else
  {
    v47 = 0;
    v46 = 0;
  }
  v48 = v46;
  BYTE1(v48) = v47;
  sub_B44240(v42, v20, v45, v48);
  if ( v40 )
  {
    v86 = (unsigned __int8 *)sub_B47F80((_BYTE *)v41);
    v43 = *a5 == 0;
    v90 = "lpad";
    if ( v43 )
    {
      v93 = 259;
    }
    else
    {
      v92 = a5;
      v93 = 771;
    }
    sub_BD6B50(v86, (const char **)&v90);
    v50 = (unsigned __int64 *)sub_AA5190(v40);
    if ( v50 )
    {
      v51 = v49;
      v52 = HIBYTE(v49);
    }
    else
    {
      v52 = 0;
      v51 = 0;
    }
    v53 = v40;
    v54 = v51;
    BYTE1(v54) = v52;
    sub_B44240(v86, v40, v50, v54);
    if ( *(_QWORD *)(v41 + 16) )
    {
      v90 = "lpad.phi";
      v93 = 259;
      v83 = *(_QWORD *)(v41 + 8);
      v55 = sub_BD2DA0(80);
      v56 = v55;
      if ( v55 )
      {
        sub_B44260(v55, v83, 55, 0x8000000u, v41 + 24, 0);
        *(_DWORD *)(v56 + 72) = 2;
        sub_BD6B50((unsigned __int8 *)v56, (const char **)&v90);
        sub_BD2A10(v56, *(_DWORD *)(v56 + 72), 1);
      }
      sub_F0A850(v56, (__int64)v42, v20);
      sub_F0A850(v56, (__int64)v86, v40);
      v53 = v56;
      sub_BD84D0(v41, v56);
    }
  }
  else
  {
    v53 = (__int64)v42;
    sub_BD84D0(v41, (__int64)v42);
  }
  result = sub_B43D60((_QWORD *)v41);
  if ( v94 != v96 )
    return _libc_free(v94, v53);
  return result;
}
