// Function: sub_2555B50
// Address: 0x2555b50
//
__int64 *__fastcall sub_2555B50(__int64 **a1, __int64 a2, __int64 a3, __int64 *a4)
{
  __int64 *v5; // rax
  __int64 v6; // rdi
  __int64 v7; // rbx
  __int16 v8; // dx
  __int64 v9; // r15
  __int64 v10; // r12
  __int64 v11; // rdx
  __int64 v12; // rdx
  __int64 v13; // r14
  __int64 v14; // rbx
  __int64 v15; // rax
  unsigned __int64 v16; // rsi
  __int64 v17; // rsi
  __int64 v18; // r8
  __int64 v19; // r9
  __int64 v20; // r12
  __int64 v21; // rax
  int v22; // ecx
  const char *v23; // rdx
  __int64 v24; // rdx
  __int64 v25; // rcx
  __int64 v26; // rdi
  char v27; // al
  __int64 v28; // rax
  int v29; // edx
  _BYTE *v30; // r12
  __int64 v31; // rdx
  __int64 v32; // r14
  _QWORD *v33; // rdi
  char v34; // al
  __int64 v35; // rsi
  __int64 v36; // rcx
  __int64 v37; // rdx
  __int64 v38; // rcx
  _QWORD *v39; // r15
  __int64 v40; // rdi
  unsigned __int16 v41; // ax
  __int64 v42; // rax
  __int64 *result; // rax
  __int64 *i; // rcx
  __int64 v45; // rdx
  __int64 v46; // rax
  __int64 v47; // rdx
  __int64 v48; // rcx
  __int64 v49; // rax
  __int64 v50; // r12
  __int64 v51; // rdx
  __int64 v52; // r14
  _QWORD *v53; // rdi
  __int64 v54; // rdx
  __int64 v55; // rcx
  _QWORD *v56; // r15
  __int64 v57; // rdx
  __int64 v58; // r12
  __int64 v59; // rbx
  _QWORD *v60; // rdi
  __int64 *v61; // [rsp+20h] [rbp-120h]
  __int64 v62; // [rsp+28h] [rbp-118h]
  __int64 v63; // [rsp+28h] [rbp-118h]
  unsigned int v64; // [rsp+30h] [rbp-110h]
  __int64 v65; // [rsp+30h] [rbp-110h]
  __int64 *v66; // [rsp+38h] [rbp-108h]
  __int64 v67; // [rsp+40h] [rbp-100h]
  __int64 v68; // [rsp+40h] [rbp-100h]
  __int64 v69; // [rsp+40h] [rbp-100h]
  __int64 v70; // [rsp+40h] [rbp-100h]
  _QWORD *v71; // [rsp+48h] [rbp-F8h]
  char v72; // [rsp+50h] [rbp-F0h]
  int v73; // [rsp+58h] [rbp-E8h]
  int v74; // [rsp+58h] [rbp-E8h]
  unsigned __int8 v75; // [rsp+5Fh] [rbp-E1h]
  unsigned __int64 v76; // [rsp+70h] [rbp-D0h] BYREF
  __int64 v77; // [rsp+78h] [rbp-C8h]
  const char *v78; // [rsp+80h] [rbp-C0h] BYREF
  __int64 v79; // [rsp+88h] [rbp-B8h]
  _QWORD v80[2]; // [rsp+90h] [rbp-B0h] BYREF
  __int16 v81; // [rsp+A0h] [rbp-A0h]
  __int64 v82; // [rsp+B0h] [rbp-90h]
  __int64 v83; // [rsp+B8h] [rbp-88h]
  __int16 v84; // [rsp+C0h] [rbp-80h]
  __int64 v85; // [rsp+C8h] [rbp-78h]
  void **v86; // [rsp+D0h] [rbp-70h]
  void **v87; // [rsp+D8h] [rbp-68h]
  __int64 v88; // [rsp+E0h] [rbp-60h]
  int v89; // [rsp+E8h] [rbp-58h]
  __int16 v90; // [rsp+ECh] [rbp-54h]
  char v91; // [rsp+EEh] [rbp-52h]
  __int64 v92; // [rsp+F0h] [rbp-50h]
  __int64 v93; // [rsp+F8h] [rbp-48h]
  void *v94; // [rsp+100h] [rbp-40h] BYREF
  void *v95; // [rsp+108h] [rbp-38h] BYREF

  v5 = *a1;
  v6 = *(_QWORD *)(a3 + 80);
  v7 = *a4;
  v61 = v5;
  if ( v6 )
    v6 -= 24;
  v9 = sub_AA5190(v6);
  if ( v9 )
  {
    v75 = v8;
    v10 = v9 - 24;
    v72 = HIBYTE(v8);
  }
  else
  {
    v72 = 0;
    v10 = 0;
    v75 = 0;
  }
  v64 = *(_DWORD *)(sub_B43CC0(v10) + 4);
  v66 = *(__int64 **)(*v61 + 104);
  v78 = sub_BD5D20(v61[1]);
  v79 = v11;
  v12 = v75;
  v80[0] = ".priv";
  v81 = 773;
  BYTE1(v12) = v72;
  v67 = v12;
  v71 = sub_BD2C40(80, unk_3F10A14);
  if ( v71 )
    sub_B4CE50((__int64)v71, v66, v64, (__int64)&v78, v9, v67);
  v13 = *(_QWORD *)(v10 + 40);
  v14 = *(unsigned int *)(v7 + 32);
  v68 = *(_QWORD *)(*v61 + 104);
  v15 = sub_AA48A0(v13);
  v16 = (unsigned __int64)v80;
  v82 = v13;
  v85 = v15;
  v86 = &v94;
  v87 = &v95;
  v79 = 0x200000000LL;
  v94 = &unk_49DA1B0;
  v78 = (const char *)v80;
  v88 = 0;
  v95 = &unk_49DA0B0;
  LOBYTE(v15) = v75;
  v89 = 0;
  BYTE1(v15) = v72;
  v90 = 512;
  v91 = 7;
  v92 = 0;
  v93 = 0;
  v83 = v9;
  v84 = v15;
  if ( v9 != v13 + 48 )
  {
    v17 = *(_QWORD *)sub_B46C60(v10);
    v76 = v17;
    if ( v17 && (sub_B96E90((__int64)&v76, v17, 1), (v20 = v76) != 0) )
    {
      v16 = (unsigned int)v79;
      v21 = (__int64)v78;
      v22 = v79;
      v23 = &v78[16 * (unsigned int)v79];
      if ( v78 != v23 )
      {
        while ( *(_DWORD *)v21 )
        {
          v21 += 16;
          if ( v23 == (const char *)v21 )
            goto LABEL_43;
        }
        *(_QWORD *)(v21 + 8) = v76;
        goto LABEL_15;
      }
LABEL_43:
      if ( (unsigned int)v79 >= (unsigned __int64)HIDWORD(v79) )
      {
        v16 = (unsigned int)v79 + 1LL;
        if ( HIDWORD(v79) < v16 )
        {
          v16 = (unsigned __int64)v80;
          sub_C8D5F0((__int64)&v78, v80, (unsigned int)v79 + 1LL, 0x10u, v18, v19);
          v23 = &v78[16 * (unsigned int)v79];
        }
        *(_QWORD *)v23 = 0;
        *((_QWORD *)v23 + 1) = v20;
        v20 = v76;
        LODWORD(v79) = v79 + 1;
      }
      else
      {
        if ( v23 )
        {
          *(_DWORD *)v23 = 0;
          *((_QWORD *)v23 + 1) = v20;
          v22 = v79;
          v20 = v76;
        }
        LODWORD(v79) = v22 + 1;
      }
    }
    else
    {
      v16 = 0;
      sub_93FB40((__int64)&v78, 0);
      v20 = v76;
    }
    if ( v20 )
    {
LABEL_15:
      v16 = v20;
      sub_B91220((__int64)&v76, v20);
    }
  }
  v26 = sub_B2BEC0(a3);
  v27 = *(_BYTE *)(v68 + 8);
  if ( v27 == 15 )
  {
    v28 = sub_AE4AC0(v26, v68);
    v29 = *(_DWORD *)(v68 + 12);
    if ( v29 )
    {
      v30 = (_BYTE *)(v28 + 32);
      v62 = v9;
      v73 = v14 + v29;
      do
      {
        v34 = *v30;
        v76 = *((_QWORD *)v30 - 1);
        LOBYTE(v77) = v34;
        v35 = sub_CA1930(&v76);
        v39 = sub_2538D20((__int64)v71, v35, (__int64 *)&v78, v36);
        if ( (*(_BYTE *)(a3 + 2) & 1) != 0 )
          sub_B2C6D0(a3, v35, v37, v38);
        v31 = v75;
        v32 = *(_QWORD *)(a3 + 96) + 40LL * (unsigned int)v14;
        BYTE1(v31) = v72;
        v69 = v31;
        v33 = sub_BD2C40(80, unk_3F10A10);
        if ( v33 )
          sub_B4D460((__int64)v33, v32, (__int64)v39, v62, v69);
        LODWORD(v14) = v14 + 1;
        v30 += 16;
      }
      while ( v73 != (_DWORD)v14 );
      v9 = v62;
    }
  }
  else if ( v27 == 16 )
  {
    v46 = sub_9208B0(v26, *(_QWORD *)(v68 + 24));
    v77 = v47;
    v76 = (unsigned __int64)(v46 + 7) >> 3;
    v63 = sub_CA1930(&v76);
    v49 = *(_QWORD *)(v68 + 32);
    if ( (_DWORD)v49 )
    {
      v65 = v9;
      v50 = 0;
      v74 = v14 + v49;
      do
      {
        v56 = sub_2538D20((__int64)v71, v50, (__int64 *)&v78, v48);
        if ( (*(_BYTE *)(a3 + 2) & 1) != 0 )
          sub_B2C6D0(a3, v50, v54, v55);
        v51 = v75;
        v52 = *(_QWORD *)(a3 + 96) + 40LL * (unsigned int)v14;
        BYTE1(v51) = v72;
        v70 = v51;
        v53 = sub_BD2C40(80, unk_3F10A10);
        if ( v53 )
          sub_B4D460((__int64)v53, v52, (__int64)v56, v65, v70);
        v50 += v63;
        LODWORD(v14) = v14 + 1;
      }
      while ( v74 != (_DWORD)v14 );
      v9 = v65;
    }
  }
  else
  {
    if ( (*(_BYTE *)(a3 + 2) & 1) != 0 )
      sub_B2C6D0(a3, v16, v24, v25);
    v57 = 5 * v14;
    v59 = v75;
    v58 = *(_QWORD *)(a3 + 96) + 8 * v57;
    BYTE1(v59) = v72;
    v60 = sub_BD2C40(80, unk_3F10A10);
    if ( v60 )
      sub_B4D460((__int64)v60, v58, (__int64)v71, v9, v59);
  }
  nullsub_61();
  v94 = &unk_49DA1B0;
  nullsub_63();
  if ( v78 != (const char *)v80 )
    _libc_free((unsigned __int64)v78);
  v40 = v61[1];
  if ( v71[1] != *(_QWORD *)(v40 + 8) )
  {
    LOBYTE(v41) = v75;
    v81 = 257;
    HIBYTE(v41) = v72;
    v42 = sub_B52190((__int64)v71, *(_QWORD *)(v40 + 8), (__int64)&v78, v9, v41);
    v40 = v61[1];
    v71 = (_QWORD *)v42;
  }
  sub_BD84D0(v40, (__int64)v71);
  result = (__int64 *)v61[2];
  for ( i = &result[*((unsigned int *)v61 + 6)]; result != i; *(_WORD *)(v45 + 2) &= 0xFFFCu )
    v45 = *result++;
  return result;
}
