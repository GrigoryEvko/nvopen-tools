// Function: sub_2359A40
// Address: 0x2359a40
//
__int64 __fastcall sub_2359A40(unsigned __int64 *a1, __int64 *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // rax
  int v8; // esi
  __int64 v9; // rax
  __int64 v10; // rcx
  _QWORD *v11; // rax
  __int64 v12; // rcx
  __int64 v13; // r8
  __int64 v14; // r9
  __int64 v15; // r12
  __int64 v16; // rax
  __int64 v17; // rax
  __int64 v18; // rdx
  __int64 v19; // rax
  _BYTE *v20; // r14
  unsigned __int64 v21; // r12
  unsigned __int64 v22; // rdi
  __int64 *v23; // r13
  __int64 *v24; // r12
  unsigned __int64 v25; // rdi
  __int64 v27; // [rsp+8h] [rbp-D8h] BYREF
  __int64 v28; // [rsp+10h] [rbp-D0h]
  __int64 v29; // [rsp+18h] [rbp-C8h]
  __int64 v30; // [rsp+20h] [rbp-C0h]
  __int64 v31; // [rsp+28h] [rbp-B8h]
  __int64 v32; // [rsp+30h] [rbp-B0h]
  __int64 v33; // [rsp+38h] [rbp-A8h]
  __int64 v34; // [rsp+40h] [rbp-A0h]
  __int64 v35; // [rsp+48h] [rbp-98h]
  __int64 v36; // [rsp+50h] [rbp-90h]
  __int64 v37; // [rsp+58h] [rbp-88h]
  __int64 v38; // [rsp+60h] [rbp-80h]
  __int64 v39; // [rsp+68h] [rbp-78h]
  unsigned int v40; // [rsp+70h] [rbp-70h]
  __int64 *v41; // [rsp+78h] [rbp-68h] BYREF
  __int64 v42; // [rsp+80h] [rbp-60h]
  __int64 v43; // [rsp+88h] [rbp-58h] BYREF
  __int64 v44; // [rsp+90h] [rbp-50h]
  __int64 v45; // [rsp+98h] [rbp-48h]
  unsigned int v46; // [rsp+A0h] [rbp-40h]
  _BYTE *v47; // [rsp+A8h] [rbp-38h] BYREF
  __int64 v48; // [rsp+B0h] [rbp-30h]
  _BYTE v49[40]; // [rsp+B8h] [rbp-28h] BYREF

  v7 = *a2;
  ++a2[9];
  v37 = 1;
  v28 = v7;
  v29 = a2[1];
  v30 = a2[2];
  v31 = a2[3];
  v32 = a2[4];
  v33 = a2[5];
  v34 = a2[6];
  v35 = a2[7];
  v36 = a2[8];
  v38 = a2[10];
  v39 = a2[11];
  LODWORD(v7) = *((_DWORD *)a2 + 24);
  a2[10] = 0;
  a2[11] = 0;
  *((_DWORD *)a2 + 24) = 0;
  v8 = *((_DWORD *)a2 + 28);
  v40 = v7;
  v41 = &v43;
  v42 = 0;
  if ( v8 )
    sub_23591E0((__int64)&v41, (__int64)(a2 + 13), a3, a4, a5, a6);
  v9 = a2[16];
  v10 = *((unsigned int *)a2 + 40);
  v43 = 1;
  ++a2[15];
  v44 = v9;
  LODWORD(v9) = *((_DWORD *)a2 + 34);
  a2[16] = 0;
  LODWORD(v45) = v9;
  LODWORD(v9) = *((_DWORD *)a2 + 35);
  v47 = v49;
  HIDWORD(v45) = v9;
  LODWORD(v9) = *((_DWORD *)a2 + 36);
  a2[17] = 0;
  v46 = v9;
  *((_DWORD *)a2 + 36) = 0;
  v48 = 0;
  if ( (_DWORD)v10 )
    sub_23596A0((__int64)&v47, (__int64)(a2 + 19), a3, v10, a5, a6);
  v11 = (_QWORD *)sub_22077B0(0xB0u);
  v15 = (__int64)v11;
  if ( v11 )
  {
    ++v37;
    v11[10] = 1;
    *v11 = &unk_4A10E78;
    v11[1] = v28;
    v11[2] = v29;
    v11[3] = v30;
    v11[4] = v31;
    v11[5] = v32;
    v11[6] = v33;
    v11[7] = v34;
    v11[8] = v35;
    v11[9] = v36;
    v16 = v38;
    v38 = 0;
    *(_QWORD *)(v15 + 88) = v16;
    v17 = v39;
    v39 = 0;
    *(_QWORD *)(v15 + 96) = v17;
    LODWORD(v17) = v40;
    v40 = 0;
    *(_DWORD *)(v15 + 104) = v17;
    *(_QWORD *)(v15 + 112) = v15 + 128;
    v18 = (unsigned int)v42;
    *(_QWORD *)(v15 + 120) = 0;
    if ( (_DWORD)v18 )
      sub_23591E0(v15 + 112, (__int64)&v41, v18, v12, v13, v14);
    v19 = v44;
    ++v43;
    *(_QWORD *)(v15 + 128) = 1;
    *(_QWORD *)(v15 + 136) = v19;
    v44 = 0;
    *(_QWORD *)(v15 + 144) = v45;
    v45 = 0;
    *(_DWORD *)(v15 + 152) = v46;
    *(_QWORD *)(v15 + 160) = v15 + 176;
    LODWORD(v19) = v48;
    v46 = 0;
    *(_QWORD *)(v15 + 168) = 0;
    if ( (_DWORD)v19 )
      sub_23596A0(v15 + 160, (__int64)&v47, v18, v12, v13, v14);
  }
  v27 = v15;
  sub_2353900(a1, (unsigned __int64 *)&v27);
  sub_233EFE0(&v27);
  v20 = v47;
  v21 = (unsigned __int64)&v47[88 * (unsigned int)v48];
  if ( v47 != (_BYTE *)v21 )
  {
    do
    {
      v21 -= 88LL;
      v22 = *(_QWORD *)(v21 + 8);
      if ( v22 != v21 + 24 )
        _libc_free(v22);
    }
    while ( v20 != (_BYTE *)v21 );
    v21 = (unsigned __int64)v47;
  }
  if ( (_BYTE *)v21 != v49 )
    _libc_free(v21);
  sub_C7D6A0(v44, 16LL * v46, 8);
  v23 = v41;
  v24 = &v41[11 * (unsigned int)v42];
  if ( v41 != v24 )
  {
    do
    {
      v24 -= 11;
      v25 = v24[1];
      if ( (__int64 *)v25 != v24 + 3 )
        _libc_free(v25);
    }
    while ( v23 != v24 );
    v24 = v41;
  }
  if ( v24 != &v43 )
    _libc_free((unsigned __int64)v24);
  return sub_C7D6A0(v38, 16LL * v40, 8);
}
