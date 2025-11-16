// Function: sub_2996750
// Address: 0x2996750
//
__int64 __fastcall sub_2996750(__int64 a1, _BYTE *a2, __int64 a3, __int64 a4)
{
  __int64 v7; // rbx
  __int64 v8; // rdi
  __int64 v9; // rax
  __int64 v10; // r15
  int v11; // r12d
  _QWORD *v12; // r13
  unsigned __int8 v13; // al
  __int64 v14; // rdx
  __int64 v15; // rcx
  void **v16; // rdx
  __int64 v17; // rcx
  __int64 v18; // r8
  __int64 v19; // r9
  void **v20; // rax
  unsigned __int8 v22; // [rsp+Fh] [rbp-441h]
  __int64 v23; // [rsp+10h] [rbp-440h]
  __int64 v24; // [rsp+10h] [rbp-440h]
  __int64 *v25; // [rsp+18h] [rbp-438h]
  unsigned __int64 v26; // [rsp+60h] [rbp-3F0h] BYREF
  __int64 v27; // [rsp+68h] [rbp-3E8h]
  __int64 v28; // [rsp+70h] [rbp-3E0h]
  __int64 v29; // [rsp+80h] [rbp-3D0h] BYREF
  void **v30; // [rsp+88h] [rbp-3C8h]
  __int64 v31; // [rsp+90h] [rbp-3C0h]
  int v32; // [rsp+98h] [rbp-3B8h]
  char v33; // [rsp+9Ch] [rbp-3B4h]
  _BYTE v34[16]; // [rsp+A0h] [rbp-3B0h] BYREF
  __int64 v35; // [rsp+B0h] [rbp-3A0h] BYREF
  _BYTE *v36; // [rsp+B8h] [rbp-398h]
  __int64 v37; // [rsp+C0h] [rbp-390h]
  __int64 v38; // [rsp+C8h] [rbp-388h]
  _BYTE v39[64]; // [rsp+D0h] [rbp-380h] BYREF
  __int64 v40; // [rsp+110h] [rbp-340h]
  char *v41; // [rsp+118h] [rbp-338h]
  __int64 v42; // [rsp+120h] [rbp-330h]
  int v43; // [rsp+128h] [rbp-328h]
  char v44; // [rsp+12Ch] [rbp-324h]
  char v45; // [rsp+130h] [rbp-320h] BYREF
  __int64 v46; // [rsp+170h] [rbp-2E0h]
  char *v47; // [rsp+178h] [rbp-2D8h]
  __int64 v48; // [rsp+180h] [rbp-2D0h]
  int v49; // [rsp+188h] [rbp-2C8h]
  char v50; // [rsp+18Ch] [rbp-2C4h]
  char v51; // [rsp+190h] [rbp-2C0h] BYREF
  char *v52; // [rsp+1D0h] [rbp-280h]
  __int64 v53; // [rsp+1D8h] [rbp-278h]
  char v54; // [rsp+1E0h] [rbp-270h] BYREF
  __int64 v55; // [rsp+2A0h] [rbp-1B0h]
  __int64 v56; // [rsp+2A8h] [rbp-1A8h]
  __int64 v57; // [rsp+2B0h] [rbp-1A0h]
  int v58; // [rsp+2B8h] [rbp-198h]
  __int64 v59; // [rsp+2C0h] [rbp-190h]
  __int64 v60; // [rsp+2C8h] [rbp-188h]
  __int64 v61; // [rsp+2D0h] [rbp-180h]
  int v62; // [rsp+2D8h] [rbp-178h]
  _QWORD *v63; // [rsp+2E0h] [rbp-170h]
  __int64 v64; // [rsp+2E8h] [rbp-168h]
  _QWORD v65[3]; // [rsp+2F0h] [rbp-160h] BYREF
  int v66; // [rsp+308h] [rbp-148h]
  char *v67; // [rsp+310h] [rbp-140h]
  __int64 v68; // [rsp+318h] [rbp-138h]
  char v69; // [rsp+320h] [rbp-130h] BYREF
  __int64 v70; // [rsp+360h] [rbp-F0h]
  __int64 v71; // [rsp+368h] [rbp-E8h]
  __int64 v72; // [rsp+370h] [rbp-E0h]
  int v73; // [rsp+378h] [rbp-D8h]
  __int64 v74; // [rsp+380h] [rbp-D0h]
  __int64 v75; // [rsp+388h] [rbp-C8h]
  __int64 v76; // [rsp+390h] [rbp-C0h]
  int v77; // [rsp+398h] [rbp-B8h]
  char *v78; // [rsp+3A0h] [rbp-B0h]
  __int64 v79; // [rsp+3A8h] [rbp-A8h]
  char v80; // [rsp+3B0h] [rbp-A0h] BYREF
  __int64 v81; // [rsp+3F0h] [rbp-60h]
  __int64 v82; // [rsp+3F8h] [rbp-58h]
  __int64 v83; // [rsp+400h] [rbp-50h]
  int v84; // [rsp+408h] [rbp-48h]

  v23 = sub_BC1CD0(a4, &unk_4F81450, a3) + 8;
  v25 = 0;
  v7 = sub_BC1CD0(a4, &unk_4FDBD00, a3);
  if ( *a2 )
    v25 = (__int64 *)(sub_BC1CD0(a4, &unk_4F8FC88, a3) + 8);
  v8 = *(_QWORD *)(v7 + 40);
  v26 = 0;
  v27 = 0;
  v28 = 0;
  sub_298BDD0(v8, (__int64)&v26);
  v9 = v27;
  if ( v27 == v26 )
    goto LABEL_25;
  v10 = v23;
  v24 = a1;
  v11 = 0;
  do
  {
    v12 = *(_QWORD **)(v9 - 8);
    v35 = 0;
    v27 = v9 - 8;
    v38 = 0x800000000LL;
    v37 = (__int64)v39;
    v40 = 0;
    v41 = &v45;
    v42 = 8;
    v47 = &v51;
    v43 = 0;
    v52 = &v54;
    v44 = 1;
    v63 = v65;
    v46 = 0;
    v48 = 8;
    v49 = 0;
    v50 = 1;
    v53 = 0x800000000LL;
    v55 = 0;
    v56 = 0;
    v57 = 0;
    v58 = 0;
    v59 = 0;
    v60 = 0;
    v61 = 0;
    v62 = 0;
    v64 = 0;
    memset(v65, 0, sizeof(v65));
    v67 = &v69;
    v78 = &v80;
    v66 = 0;
    v68 = 0x800000000LL;
    v70 = 0;
    v71 = 0;
    v72 = 0;
    v73 = 0;
    v74 = 0;
    v75 = 0;
    v76 = 0;
    v77 = 0;
    v79 = 0x800000000LL;
    v81 = 0;
    v82 = 0;
    v83 = 0;
    v84 = 0;
    sub_2988360(&v29, v12);
    if ( !*a2 )
      goto LABEL_34;
    if ( !v12[4] )
      goto LABEL_10;
    v13 = sub_298D780((__int64)&v29, v12, v25);
    if ( !v13 )
    {
LABEL_34:
      if ( v12[4] )
        v11 |= sub_2994AD0((__int64)&v29, (__int64)v12, v10);
LABEL_10:
      sub_2989910((__int64)&v29);
      goto LABEL_11;
    }
    v22 = v13;
    sub_2989910((__int64)&v29);
    v11 = v22;
LABEL_11:
    v9 = v27;
  }
  while ( v27 != v26 );
  a1 = v24;
  if ( !(_BYTE)v11 )
  {
LABEL_25:
    *(_QWORD *)(a1 + 48) = 0;
    *(_QWORD *)(a1 + 8) = a1 + 32;
    *(_QWORD *)(a1 + 56) = a1 + 80;
    *(_QWORD *)(a1 + 16) = 0x100000002LL;
    *(_QWORD *)(a1 + 64) = 2;
    *(_DWORD *)(a1 + 72) = 0;
    *(_BYTE *)(a1 + 76) = 1;
    *(_DWORD *)(a1 + 24) = 0;
    *(_BYTE *)(a1 + 28) = 1;
    *(_QWORD *)(a1 + 32) = &qword_4F82400;
    *(_QWORD *)a1 = 1;
    goto LABEL_21;
  }
  v29 = 0;
  v30 = (void **)v34;
  v31 = 2;
  v32 = 0;
  v33 = 1;
  v35 = 0;
  v36 = v39;
  v37 = 2;
  LODWORD(v38) = 0;
  BYTE4(v38) = 1;
  if ( (unsigned __int8)sub_B19060((__int64)&v29, (__int64)&qword_4F82400, v14, v15) )
    goto LABEL_19;
  if ( !v33 )
    goto LABEL_26;
  v20 = v30;
  v17 = HIDWORD(v31);
  v16 = &v30[HIDWORD(v31)];
  if ( v30 == v16 )
  {
LABEL_30:
    if ( HIDWORD(v31) < (unsigned int)v31 )
    {
      ++HIDWORD(v31);
      *v16 = &unk_4F81450;
      ++v29;
      goto LABEL_19;
    }
LABEL_26:
    sub_C8CC70((__int64)&v29, (__int64)&unk_4F81450, (__int64)v16, v17, v18, v19);
    goto LABEL_19;
  }
  while ( *v20 != &unk_4F81450 )
  {
    if ( v16 == ++v20 )
      goto LABEL_30;
  }
LABEL_19:
  sub_C8CF70(v24, (void *)(v24 + 32), 2, (__int64)v34, (__int64)&v29);
  sub_C8CF70(v24 + 48, (void *)(v24 + 80), 2, (__int64)v39, (__int64)&v35);
  if ( BYTE4(v38) )
  {
    if ( v33 )
      goto LABEL_21;
LABEL_27:
    _libc_free((unsigned __int64)v30);
  }
  else
  {
    _libc_free((unsigned __int64)v36);
    if ( !v33 )
      goto LABEL_27;
  }
LABEL_21:
  if ( v26 )
    j_j___libc_free_0(v26);
  return a1;
}
