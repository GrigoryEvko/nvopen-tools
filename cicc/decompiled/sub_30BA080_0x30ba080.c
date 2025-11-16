// Function: sub_30BA080
// Address: 0x30ba080
//
void __fastcall sub_30BA080(_QWORD *a1)
{
  __int64 v1; // rdx
  __int64 v2; // rax
  __int64 v3; // rcx
  __int64 v4; // rdx
  __int64 v5; // rcx
  __int64 v6; // r8
  __int64 v7; // r9
  __int64 v8; // rdx
  __int64 v9; // rcx
  __int64 v10; // r8
  __int64 v11; // r9
  __int64 v12; // rdx
  __int64 v13; // r8
  __int64 v14; // r9
  __int64 v15; // rcx
  __int64 v16; // rax
  __int64 v17; // rdx
  _QWORD *v18; // rax
  __int64 v19; // r14
  __int64 v20; // rax
  unsigned __int64 v21; // rdx
  bool v22; // zf
  __int64 (__fastcall *v23)(__int64, __int64); // rax
  __int64 v24; // rsi
  _BYTE *v25; // r14
  __int64 v26; // r13
  __int64 v27; // rdx
  __int64 v28; // r15
  __int64 v29; // rbx
  __int64 v30; // r12
  unsigned __int64 v31; // r8
  __int64 v32; // rax
  _BYTE *v33; // rdi
  unsigned __int64 v34[54]; // [rsp+30h] [rbp-C60h] BYREF
  __int64 v35; // [rsp+1E0h] [rbp-AB0h] BYREF
  __int64 *v36; // [rsp+1E8h] [rbp-AA8h]
  int v37; // [rsp+1F0h] [rbp-AA0h]
  int v38; // [rsp+1F4h] [rbp-A9Ch]
  int v39; // [rsp+1F8h] [rbp-A98h]
  char v40; // [rsp+1FCh] [rbp-A94h]
  __int64 v41; // [rsp+200h] [rbp-A90h] BYREF
  _QWORD *v42; // [rsp+240h] [rbp-A50h]
  __int64 v43; // [rsp+248h] [rbp-A48h]
  _QWORD v44[40]; // [rsp+250h] [rbp-A40h] BYREF
  _BYTE v45[8]; // [rsp+390h] [rbp-900h] BYREF
  unsigned __int64 v46; // [rsp+398h] [rbp-8F8h]
  char v47; // [rsp+3ACh] [rbp-8E4h]
  _BYTE v48[64]; // [rsp+3B0h] [rbp-8E0h] BYREF
  _BYTE *v49; // [rsp+3F0h] [rbp-8A0h] BYREF
  __int64 v50; // [rsp+3F8h] [rbp-898h]
  _BYTE v51[320]; // [rsp+400h] [rbp-890h] BYREF
  _BYTE v52[8]; // [rsp+540h] [rbp-750h] BYREF
  unsigned __int64 v53; // [rsp+548h] [rbp-748h]
  char v54; // [rsp+55Ch] [rbp-734h]
  _BYTE v55[64]; // [rsp+560h] [rbp-730h] BYREF
  _BYTE *v56; // [rsp+5A0h] [rbp-6F0h] BYREF
  __int64 v57; // [rsp+5A8h] [rbp-6E8h]
  _BYTE v58[320]; // [rsp+5B0h] [rbp-6E0h] BYREF
  _BYTE *v59; // [rsp+6F0h] [rbp-5A0h] BYREF
  __int64 v60; // [rsp+6F8h] [rbp-598h]
  _BYTE v61[512]; // [rsp+700h] [rbp-590h] BYREF
  _BYTE v62[8]; // [rsp+900h] [rbp-390h] BYREF
  unsigned __int64 v63; // [rsp+908h] [rbp-388h]
  char v64; // [rsp+91Ch] [rbp-374h]
  char *v65; // [rsp+960h] [rbp-330h] BYREF
  int v66; // [rsp+968h] [rbp-328h]
  char v67; // [rsp+970h] [rbp-320h] BYREF
  _BYTE v68[8]; // [rsp+AB0h] [rbp-1E0h] BYREF
  unsigned __int64 v69; // [rsp+AB8h] [rbp-1D8h]
  char v70; // [rsp+ACCh] [rbp-1C4h]
  char *v71; // [rsp+B10h] [rbp-180h] BYREF
  unsigned int v72; // [rsp+B18h] [rbp-178h]
  char v73; // [rsp+B20h] [rbp-170h] BYREF

  if ( !(*(unsigned __int8 (__fastcall **)(_QWORD *))(*a1 + 88LL))(a1) )
    return;
  v1 = a1[1];
  v59 = v61;
  v60 = 0x4000000000LL;
  memset(v34, 0, sizeof(v34));
  HIDWORD(v34[13]) = 8;
  v34[1] = (unsigned __int64)&v34[4];
  v34[12] = (unsigned __int64)&v34[14];
  v2 = *(_QWORD *)(v1 + 88);
  v42 = v44;
  v43 = 0x800000000LL;
  v3 = *(_QWORD *)(v2 + 40);
  v36 = &v41;
  v4 = *(unsigned int *)(v2 + 48);
  v41 = v2;
  v44[4] = v2;
  v44[0] = v3 + 8 * v4;
  v44[2] = v3;
  v44[1] = sub_30B9540;
  v44[3] = sub_30B9540;
  LODWORD(v34[2]) = 8;
  BYTE4(v34[3]) = 1;
  v37 = 8;
  v39 = 0;
  v40 = 1;
  v38 = 1;
  v35 = 1;
  LODWORD(v43) = 1;
  sub_30B99C0((__int64)&v35);
  sub_30B9DF0((__int64)v52, (__int64)v34);
  sub_30B9DF0((__int64)v45, (__int64)&v35);
  sub_30B9DF0((__int64)v62, (__int64)v45);
  sub_30B9DF0((__int64)v68, (__int64)v52);
  if ( v49 != v51 )
    _libc_free((unsigned __int64)v49);
  if ( !v47 )
    _libc_free(v46);
  if ( v56 != v58 )
    _libc_free((unsigned __int64)v56);
  if ( !v54 )
    _libc_free(v53);
  if ( v42 != v44 )
    _libc_free((unsigned __int64)v42);
  if ( !v40 )
    _libc_free((unsigned __int64)v36);
  if ( (unsigned __int64 *)v34[12] != &v34[14] )
    _libc_free(v34[12]);
  if ( !BYTE4(v34[3]) )
    _libc_free(v34[1]);
  sub_C8CD80((__int64)v45, (__int64)v48, (__int64)v62, v5, v6, v7);
  v49 = v51;
  v50 = 0x800000000LL;
  if ( v66 )
    sub_30B9F10((__int64)&v49, (__int64 *)&v65, v8, v9, v10, v11);
  sub_C8CD80((__int64)v52, (__int64)v55, (__int64)v68, v9, v10, v11);
  v15 = v72;
  v56 = v58;
  v57 = 0x800000000LL;
  if ( v72 )
  {
    sub_30B9F10((__int64)&v56, (__int64 *)&v71, v12, v72, v13, v14);
    v15 = (unsigned int)v57;
  }
LABEL_23:
  v16 = (unsigned int)v50;
  while ( 1 )
  {
    v17 = 40 * v16;
    if ( v16 != v15 )
      goto LABEL_28;
    v13 = (__int64)v56;
    if ( &v49[v17] == v49 )
      break;
    v15 = (__int64)v56;
    v18 = v49;
    while ( 1 )
    {
      v14 = *(_QWORD *)(v15 + 32);
      if ( v18[4] != v14 || v18[2] != *(_QWORD *)(v15 + 16) || *v18 != *(_QWORD *)v15 )
        break;
      v18 += 5;
      v15 += 40;
      if ( &v49[v17] == (_BYTE *)v18 )
        goto LABEL_40;
    }
LABEL_28:
    v19 = *(_QWORD *)&v49[v17 - 8];
    if ( *(_DWORD *)(v19 + 56) == 3 )
    {
      v23 = *(__int64 (__fastcall **)(__int64, __int64))(*a1 + 64LL);
      if ( v23 == sub_30B01E0 )
        v24 = v19 + 64;
      else
        v24 = v23((__int64)a1, *(_QWORD *)&v49[v17 - 8]);
      sub_30B9940((__int64)&v59, v24, v17, v15, v13, v14);
    }
    v20 = (unsigned int)v60;
    v21 = (unsigned int)v60 + 1LL;
    if ( v21 > HIDWORD(v60) )
    {
      sub_C8D5F0((__int64)&v59, v61, v21, 8u, v13, v14);
      v20 = (unsigned int)v60;
    }
    *(_QWORD *)&v59[8 * v20] = v19;
    LODWORD(v60) = v60 + 1;
    v22 = (_DWORD)v50 == 1;
    v16 = (unsigned int)(v50 - 1);
    LODWORD(v50) = v50 - 1;
    if ( !v22 )
    {
      sub_30B99C0((__int64)v45);
      v15 = (unsigned int)v57;
      goto LABEL_23;
    }
    v15 = (unsigned int)v57;
  }
LABEL_40:
  if ( v56 != v58 )
    _libc_free((unsigned __int64)v56);
  if ( !v54 )
    _libc_free(v53);
  if ( v49 != v51 )
    _libc_free((unsigned __int64)v49);
  if ( !v47 )
    _libc_free(v46);
  if ( v71 != &v73 )
    _libc_free((unsigned __int64)v71);
  if ( !v70 )
    _libc_free(v69);
  if ( v65 != &v67 )
    _libc_free((unsigned __int64)v65);
  if ( !v64 )
    _libc_free(v63);
  v25 = v59;
  *(_DWORD *)(a1[1] + 104LL) = 0;
  v26 = a1[1];
  v27 = *(unsigned int *)(v26 + 104);
  v28 = 8LL * (unsigned int)v60;
  v29 = (unsigned int)v60;
  v30 = (unsigned int)v60;
  v31 = v27 + (unsigned int)v60;
  if ( v31 > *(unsigned int *)(v26 + 108) )
  {
    sub_C8D5F0(v26 + 96, (const void *)(v26 + 112), v27 + (unsigned int)v60, 8u, v31, v14);
    v27 = *(unsigned int *)(v26 + 104);
  }
  v32 = *(_QWORD *)(v26 + 96) + 8 * v27;
  if ( v28 )
  {
    do
    {
      v32 += 8;
      *(_QWORD *)(v32 - 8) = *(_QWORD *)&v25[8 * v30-- - 8 + v28 - 8 * v29];
    }
    while ( v30 );
    LODWORD(v27) = *(_DWORD *)(v26 + 104);
  }
  v33 = v59;
  *(_DWORD *)(v26 + 104) = v27 + v29;
  if ( v33 != v61 )
    _libc_free((unsigned __int64)v33);
}
