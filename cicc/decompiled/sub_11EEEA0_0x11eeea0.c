// Function: sub_11EEEA0
// Address: 0x11eeea0
//
__int64 __fastcall sub_11EEEA0(__int64 a1, __int64 a2, char a3, __int64 a4)
{
  _QWORD *v5; // rdi
  __int64 v6; // rsi
  int v7; // eax
  __int64 v8; // rcx
  __int64 v9; // r14
  __int64 v10; // rax
  __int64 v11; // r14
  __int64 v12; // rbx
  __int64 **v13; // r8
  __int64 v14; // r15
  __int64 v15; // rbx
  __int64 v16; // r12
  __int64 *v17; // r14
  _BYTE *v18; // rsi
  __int64 *v19; // r15
  __int64 v20; // rdx
  int v21; // eax
  _QWORD *v22; // rdi
  __int64 v23; // rax
  size_t v24; // r8
  char *v25; // r9
  unsigned __int64 v26; // rax
  unsigned __int64 v27; // r12
  int v28; // edx
  int v29; // r15d
  __int64 v30; // rax
  _BYTE *v31; // rbx
  _QWORD *v32; // rdi
  __int64 v33; // rax
  __int64 v34; // rax
  __int64 v35; // rdi
  unsigned __int8 *v36; // r15
  __int64 (__fastcall *v37)(__int64, _BYTE *, unsigned __int8 *); // rax
  __int64 v38; // r12
  __int64 v39; // rax
  unsigned __int8 *v40; // r15
  __int64 v41; // rdi
  __int64 (__fastcall *v42)(__int64, _BYTE *, unsigned __int8 *); // rax
  __int64 *v43; // r15
  __int64 *v44; // r14
  __int64 *v45; // rbx
  __int64 *v46; // r15
  __int64 *v47; // r14
  __int64 *v48; // r15
  _QWORD *v50; // rdi
  __int64 *v51; // rax
  __int64 v52; // rax
  __int64 v53; // r14
  unsigned int *v54; // rbx
  __int64 v55; // rdx
  _QWORD *v56; // rax
  unsigned int *v57; // rax
  __int64 v58; // rdx
  unsigned int *v59; // rbx
  unsigned int *v60; // r15
  __int64 v61; // rdx
  unsigned int v62; // esi
  __int64 v63; // [rsp+8h] [rbp-188h]
  char v64; // [rsp+13h] [rbp-17Dh]
  __int64 *v68; // [rsp+38h] [rbp-158h]
  char *v69; // [rsp+40h] [rbp-150h]
  unsigned int v70; // [rsp+40h] [rbp-150h]
  __int64 **v71; // [rsp+48h] [rbp-148h]
  size_t v72; // [rsp+48h] [rbp-148h]
  __int64 v73; // [rsp+48h] [rbp-148h]
  __int64 v74; // [rsp+48h] [rbp-148h]
  _BYTE *v75; // [rsp+48h] [rbp-148h]
  unsigned int v76; // [rsp+54h] [rbp-13Ch] BYREF
  __int64 v77; // [rsp+58h] [rbp-138h] BYREF
  __int64 *v78; // [rsp+60h] [rbp-130h] BYREF
  __int64 v79; // [rsp+68h] [rbp-128h]
  _BYTE v80[16]; // [rsp+70h] [rbp-120h] BYREF
  __int64 *v81; // [rsp+80h] [rbp-110h] BYREF
  __int64 v82; // [rsp+88h] [rbp-108h]
  _BYTE v83[16]; // [rsp+90h] [rbp-100h] BYREF
  __int64 *v84; // [rsp+A0h] [rbp-F0h] BYREF
  __int64 v85; // [rsp+A8h] [rbp-E8h]
  _BYTE v86[16]; // [rsp+B0h] [rbp-E0h] BYREF
  _QWORD v87[4]; // [rsp+C0h] [rbp-D0h] BYREF
  char v88; // [rsp+E0h] [rbp-B0h]
  char v89; // [rsp+E1h] [rbp-AFh]
  char *v90; // [rsp+F0h] [rbp-A0h] BYREF
  __int64 v91; // [rsp+F8h] [rbp-98h]
  _QWORD v92[2]; // [rsp+100h] [rbp-90h] BYREF
  __int16 v93; // [rsp+110h] [rbp-80h]
  _QWORD *v94; // [rsp+120h] [rbp-70h] BYREF
  _QWORD v95[12]; // [rsp+130h] [rbp-60h] BYREF

  v5 = (_QWORD *)(a2 + 72);
  v6 = 41;
  if ( !(unsigned __int8)sub_A73ED0(v5, 41) )
  {
    v6 = 41;
    if ( !(unsigned __int8)sub_B49560(a2, 41) )
      return 0;
  }
  if ( !sub_B49E00(a2) )
    return 0;
  v7 = *(_DWORD *)(a2 + 4);
  v84 = (__int64 *)v86;
  v81 = (__int64 *)v83;
  v8 = *(_QWORD *)(a2 - 32LL * (v7 & 0x7FFFFFF));
  v78 = (__int64 *)v80;
  v9 = v8;
  v79 = 0x100000000LL;
  v82 = 0x100000000LL;
  v85 = 0x100000000LL;
  v63 = v8;
  v64 = *(_BYTE *)(*(_QWORD *)(v8 + 8) + 8LL);
  v10 = sub_B43CB0(a2);
  v11 = *(_QWORD *)(v9 + 16);
  v12 = v10;
  if ( v11 )
  {
    v13 = &v78;
    v14 = v11;
    do
    {
      v6 = *(_QWORD *)(v14 + 24);
      v71 = v13;
      sub_11EED10(a1, v6, v12, v64 == 2, (__int64)v13, (__int64)&v81, (__int64)&v84);
      v14 = *(_QWORD *)(v14 + 8);
      v13 = v71;
    }
    while ( v14 );
  }
  if ( !(_DWORD)v79 || !(_DWORD)v82 )
    goto LABEL_54;
  v15 = *(_QWORD *)(a2 - 32);
  if ( !v15 || *(_BYTE *)v15 || *(_QWORD *)(v15 + 24) != *(_QWORD *)(a2 + 80) )
  {
    v77 = v63;
    BUG();
  }
  v16 = *(_QWORD *)(v15 + 40);
  v17 = *(__int64 **)(a1 + 24);
  v18 = *(_BYTE **)(v16 + 232);
  v19 = *(__int64 **)(v63 + 8);
  v77 = v63;
  v20 = *(_QWORD *)(v16 + 240);
  v94 = v95;
  sub_11DA140((__int64 *)&v94, v18, (__int64)&v18[v20]);
  v21 = *(_DWORD *)(v16 + 264);
  v95[2] = *(_QWORD *)(v16 + 264);
  v95[3] = *(_QWORD *)(v16 + 272);
  v95[4] = *(_QWORD *)(v16 + 280);
  if ( v64 == 2 )
  {
    if ( v21 == 39 )
    {
      v23 = sub_BCDA70(v19, 2);
    }
    else
    {
      v22 = (_QWORD *)*v19;
      v90 = (char *)v19;
      v91 = (__int64)v19;
      v23 = (__int64)sub_BD0B90(v22, &v90, 2, 0);
    }
    v24 = 17;
    v25 = "__sincospif_stret";
    v68 = (__int64 *)v23;
  }
  else
  {
    v50 = (_QWORD *)*v19;
    v90 = (char *)v19;
    v91 = (__int64)v19;
    v51 = sub_BD0B90(v50, &v90, 2, 0);
    v24 = 16;
    v25 = "__sincospi_stret";
    v68 = v51;
  }
  v6 = (__int64)v17;
  v69 = v25;
  v72 = v24;
  if ( !sub_11C9D10((__int64 *)v16, v17, v25, v24) )
  {
    if ( v94 != v95 )
    {
      v6 = v95[0] + 1LL;
      j_j___libc_free_0(v94, v95[0] + 1LL);
    }
LABEL_54:
    v38 = 0;
    if ( v84 != (__int64 *)v86 )
      _libc_free(v84, v6);
    goto LABEL_45;
  }
  sub_980AF0(*v17, v69, v72, &v76);
  v73 = *(_QWORD *)(v15 + 120);
  v70 = v76;
  v92[0] = v19;
  v90 = (char *)v92;
  v91 = 0x100000001LL;
  v26 = sub_BCF480(v68, v92, 1, 0);
  v27 = sub_11C96C0(v16, v17, v70, v26, v73);
  v29 = v28;
  if ( v90 != (char *)v92 )
    _libc_free(v90, v17);
  if ( *(_BYTE *)v77 <= 0x1Cu )
  {
    v52 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a4 + 48) + 72LL) + 80LL);
    if ( !v52 )
      BUG();
    sub_A88F30(a4, v52 - 24, *(_QWORD *)(v52 + 32), 1);
  }
  else
  {
    sub_A88F30(a4, *(_QWORD *)(v77 + 40), *(_QWORD *)(v77 + 32), 0);
  }
  v90 = "sincospi";
  v93 = 259;
  v30 = sub_921880((unsigned int **)a4, v27, v29, (int)&v77, 1, (__int64)&v90, 0);
  v31 = (_BYTE *)v30;
  if ( *(_BYTE *)(*(_QWORD *)(v30 + 8) + 8LL) == 15 )
  {
    v93 = 259;
    v90 = "sinpi";
    LODWORD(v87[0]) = 0;
    v38 = sub_94D3D0((unsigned int **)a4, v30, (__int64)v87, 1, (__int64)&v90);
    v6 = (__int64)v31;
    v93 = 259;
    v90 = "cospi";
    LODWORD(v87[0]) = 1;
    v74 = sub_94D3D0((unsigned int **)a4, (__int64)v31, (__int64)v87, 1, (__int64)&v90);
    goto LABEL_31;
  }
  v89 = 1;
  v87[0] = "sinpi";
  v32 = *(_QWORD **)(a4 + 72);
  v88 = 3;
  v33 = sub_BCB2D0(v32);
  v34 = sub_ACD640(v33, 0, 0);
  v35 = *(_QWORD *)(a4 + 80);
  v36 = (unsigned __int8 *)v34;
  v37 = *(__int64 (__fastcall **)(__int64, _BYTE *, unsigned __int8 *))(*(_QWORD *)v35 + 96LL);
  if ( v37 != sub_948070 )
  {
    v38 = v37(v35, v31, v36);
LABEL_25:
    if ( v38 )
      goto LABEL_26;
    goto LABEL_67;
  }
  if ( *v31 <= 0x15u && *v36 <= 0x15u )
  {
    v38 = sub_AD5840((__int64)v31, v36, 0);
    goto LABEL_25;
  }
LABEL_67:
  v93 = 257;
  v56 = sub_BD2C40(72, 2u);
  v38 = (__int64)v56;
  if ( v56 )
    sub_B4DE80((__int64)v56, (__int64)v31, (__int64)v36, (__int64)&v90, 0, 0);
  (*(void (__fastcall **)(_QWORD, __int64, _QWORD *, _QWORD, _QWORD))(**(_QWORD **)(a4 + 88) + 16LL))(
    *(_QWORD *)(a4 + 88),
    v38,
    v87,
    *(_QWORD *)(a4 + 56),
    *(_QWORD *)(a4 + 64));
  v57 = *(unsigned int **)a4;
  v58 = 4LL * *(unsigned int *)(a4 + 8);
  if ( v57 != &v57[v58] )
  {
    v75 = v31;
    v59 = *(unsigned int **)a4;
    v60 = &v57[v58];
    do
    {
      v61 = *((_QWORD *)v59 + 1);
      v62 = *v59;
      v59 += 4;
      sub_B99FD0(v38, v62, v61);
    }
    while ( v60 != v59 );
    v31 = v75;
  }
LABEL_26:
  v89 = 1;
  v87[0] = "cospi";
  v88 = 3;
  v39 = sub_BCB2D0(*(_QWORD **)(a4 + 72));
  v40 = (unsigned __int8 *)sub_ACD640(v39, 1, 0);
  v41 = *(_QWORD *)(a4 + 80);
  v42 = *(__int64 (__fastcall **)(__int64, _BYTE *, unsigned __int8 *))(*(_QWORD *)v41 + 96LL);
  if ( v42 == sub_948070 )
  {
    if ( *v31 > 0x15u || *v40 > 0x15u )
      goto LABEL_61;
    v6 = (__int64)v40;
    v74 = sub_AD5840((__int64)v31, v40, 0);
  }
  else
  {
    v6 = (__int64)v31;
    v74 = v42(v41, v31, v40);
  }
  if ( !v74 )
  {
LABEL_61:
    v93 = 257;
    v74 = (__int64)sub_BD2C40(72, 2u);
    if ( v74 )
      sub_B4DE80(v74, (__int64)v31, (__int64)v40, (__int64)&v90, 0, 0);
    v6 = v74;
    (*(void (__fastcall **)(_QWORD, __int64, _QWORD *, _QWORD, _QWORD))(**(_QWORD **)(a4 + 88) + 16LL))(
      *(_QWORD *)(a4 + 88),
      v74,
      v87,
      *(_QWORD *)(a4 + 56),
      *(_QWORD *)(a4 + 64));
    v53 = *(_QWORD *)a4 + 16LL * *(unsigned int *)(a4 + 8);
    if ( *(_QWORD *)a4 != v53 )
    {
      v54 = *(unsigned int **)a4;
      do
      {
        v55 = *((_QWORD *)v54 + 1);
        v6 = *v54;
        v54 += 4;
        sub_B99FD0(v74, v6, v55);
      }
      while ( (unsigned int *)v53 != v54 );
    }
  }
LABEL_31:
  if ( v94 != v95 )
  {
    v6 = v95[0] + 1LL;
    j_j___libc_free_0(v94, v95[0] + 1LL);
  }
  v43 = v78;
  v44 = &v78[(unsigned int)v79];
  if ( v78 != v44 )
  {
    do
    {
      v6 = *v43++;
      sub_11EA700(a1);
    }
    while ( v44 != v43 );
  }
  if ( v81 != &v81[(unsigned int)v82] )
  {
    v45 = v81;
    v46 = &v81[(unsigned int)v82];
    do
    {
      v6 = *v45++;
      sub_11EA700(a1);
    }
    while ( v46 != v45 );
  }
  v47 = v84;
  v48 = &v84[(unsigned int)v85];
  if ( v84 != v48 )
  {
    do
    {
      v6 = *v47++;
      sub_11EA700(a1);
    }
    while ( v48 != v47 );
    v48 = v84;
  }
  if ( !a3 )
    v38 = v74;
  if ( v48 != (__int64 *)v86 )
    _libc_free(v48, v6);
LABEL_45:
  if ( v81 != (__int64 *)v83 )
    _libc_free(v81, v6);
  if ( v78 != (__int64 *)v80 )
    _libc_free(v78, v6);
  return v38;
}
