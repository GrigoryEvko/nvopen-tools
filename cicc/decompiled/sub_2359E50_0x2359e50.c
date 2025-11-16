// Function: sub_2359E50
// Address: 0x2359e50
//
__int64 __fastcall sub_2359E50(__int64 a1, __int64 *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  char v6; // r15
  __int64 v9; // rax
  int v10; // esi
  __int64 v11; // rax
  __int64 v12; // rcx
  _QWORD *v13; // rax
  __int64 v14; // rcx
  __int64 v15; // r8
  __int64 v16; // r9
  _QWORD *v17; // r12
  __int64 v18; // rax
  __int64 v19; // rax
  __int64 v20; // rdx
  __int64 v21; // rax
  unsigned __int64 v22; // r12
  __int64 *v23; // r14
  __int64 *v24; // r12
  unsigned __int64 v25; // rdi
  __int64 v27; // rax
  _BYTE *v28; // r15
  unsigned __int64 v29; // rdi
  __int64 v30; // [rsp+0h] [rbp-E0h]
  __int64 v31; // [rsp+8h] [rbp-D8h]
  __int64 v32; // [rsp+10h] [rbp-D0h]
  __int64 v33; // [rsp+18h] [rbp-C8h]
  __int64 v34; // [rsp+20h] [rbp-C0h]
  __int64 v35; // [rsp+28h] [rbp-B8h]
  __int64 v36; // [rsp+30h] [rbp-B0h]
  __int64 v37; // [rsp+38h] [rbp-A8h]
  __int64 v38; // [rsp+40h] [rbp-A0h]
  __int64 v39; // [rsp+48h] [rbp-98h]
  __int64 v40; // [rsp+50h] [rbp-90h]
  __int64 v41; // [rsp+58h] [rbp-88h]
  unsigned int v42; // [rsp+60h] [rbp-80h]
  __int64 *v43; // [rsp+68h] [rbp-78h] BYREF
  __int64 v44; // [rsp+70h] [rbp-70h]
  __int64 v45; // [rsp+78h] [rbp-68h] BYREF
  __int64 v46; // [rsp+80h] [rbp-60h]
  __int64 v47; // [rsp+88h] [rbp-58h]
  unsigned int v48; // [rsp+90h] [rbp-50h]
  _BYTE *v49; // [rsp+98h] [rbp-48h] BYREF
  __int64 v50; // [rsp+A0h] [rbp-40h]
  _BYTE v51[56]; // [rsp+A8h] [rbp-38h] BYREF

  v6 = a3;
  v9 = *a2;
  ++a2[9];
  v39 = 1;
  v30 = v9;
  v31 = a2[1];
  v32 = a2[2];
  v33 = a2[3];
  v34 = a2[4];
  v35 = a2[5];
  v36 = a2[6];
  v37 = a2[7];
  v38 = a2[8];
  v40 = a2[10];
  LODWORD(v41) = *((_DWORD *)a2 + 22);
  LODWORD(v9) = *((_DWORD *)a2 + 23);
  a2[10] = 0;
  HIDWORD(v41) = v9;
  LODWORD(v9) = *((_DWORD *)a2 + 24);
  a2[11] = 0;
  *((_DWORD *)a2 + 24) = 0;
  v10 = *((_DWORD *)a2 + 28);
  v42 = v9;
  v43 = &v45;
  v44 = 0;
  if ( v10 )
    sub_23591E0((__int64)&v43, (__int64)(a2 + 13), a3, a4, a5, a6);
  v11 = a2[16];
  v12 = *((unsigned int *)a2 + 40);
  v45 = 1;
  ++a2[15];
  v46 = v11;
  LODWORD(v11) = *((_DWORD *)a2 + 34);
  a2[16] = 0;
  LODWORD(v47) = v11;
  LODWORD(v11) = *((_DWORD *)a2 + 35);
  v49 = v51;
  HIDWORD(v47) = v11;
  LODWORD(v11) = *((_DWORD *)a2 + 36);
  a2[17] = 0;
  v48 = v11;
  *((_DWORD *)a2 + 36) = 0;
  v50 = 0;
  if ( (_DWORD)v12 )
    sub_23596A0((__int64)&v49, (__int64)(a2 + 19), a3, v12, a5, a6);
  v13 = (_QWORD *)sub_22077B0(0xB0u);
  v17 = v13;
  if ( v13 )
  {
    ++v39;
    v13[10] = 1;
    *v13 = &unk_4A10E78;
    v13[1] = v30;
    v13[2] = v31;
    v13[3] = v32;
    v13[4] = v33;
    v13[5] = v34;
    v13[6] = v35;
    v13[7] = v36;
    v13[8] = v37;
    v13[9] = v38;
    v18 = v40;
    v40 = 0;
    v17[11] = v18;
    v19 = v41;
    v41 = 0;
    v17[12] = v19;
    LODWORD(v19) = v42;
    v42 = 0;
    *((_DWORD *)v17 + 26) = v19;
    v17[14] = v17 + 16;
    v20 = (unsigned int)v44;
    v17[15] = 0;
    if ( (_DWORD)v20 )
      sub_23591E0((__int64)(v17 + 14), (__int64)&v43, v20, v14, v15, v16);
    v21 = v46;
    ++v45;
    v17[16] = 1;
    v17[17] = v21;
    v46 = 0;
    v17[18] = v47;
    v47 = 0;
    *((_DWORD *)v17 + 38) = v48;
    v17[20] = v17 + 22;
    LODWORD(v21) = v50;
    v48 = 0;
    v17[21] = 0;
    if ( !(_DWORD)v21 )
    {
      *(_BYTE *)(a1 + 8) = v6;
      *(_QWORD *)a1 = v17;
      v22 = (unsigned __int64)v49;
      goto LABEL_10;
    }
    sub_23596A0((__int64)(v17 + 20), (__int64)&v49, v20, v14, v15, v16);
  }
  v27 = (unsigned int)v50;
  *(_BYTE *)(a1 + 8) = v6;
  v28 = v49;
  *(_QWORD *)a1 = v17;
  v22 = (unsigned __int64)&v28[88 * v27];
  if ( v28 != (_BYTE *)v22 )
  {
    do
    {
      v22 -= 88LL;
      v29 = *(_QWORD *)(v22 + 8);
      if ( v29 != v22 + 24 )
        _libc_free(v29);
    }
    while ( v28 != (_BYTE *)v22 );
    v22 = (unsigned __int64)v49;
  }
LABEL_10:
  if ( (_BYTE *)v22 != v51 )
    _libc_free(v22);
  sub_C7D6A0(v46, 16LL * v48, 8);
  v23 = v43;
  v24 = &v43[11 * (unsigned int)v44];
  if ( v43 != v24 )
  {
    do
    {
      v24 -= 11;
      v25 = v24[1];
      if ( (__int64 *)v25 != v24 + 3 )
        _libc_free(v25);
    }
    while ( v23 != v24 );
    v24 = v43;
  }
  if ( v24 != &v45 )
    _libc_free((unsigned __int64)v24);
  sub_C7D6A0(v40, 16LL * v42, 8);
  return a1;
}
