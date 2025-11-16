// Function: sub_235A280
// Address: 0x235a280
//
__int64 __fastcall sub_235A280(__int64 a1, __int64 *a2, char a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 *v6; // rax
  char v9; // r12
  __int64 v10; // rdx
  int v11; // edi
  __int64 v12; // rdx
  __int64 v13; // rdx
  int v14; // esi
  __int64 v15; // rdx
  __int64 v16; // rax
  __int64 v17; // r8
  __int64 v18; // r9
  __int64 v19; // rdx
  __int64 v20; // rdx
  __int64 v21; // rcx
  __int64 v22; // rdx
  __int64 v23; // rdx
  unsigned __int64 v24; // r12
  __int64 *v25; // r14
  __int64 *v26; // r12
  unsigned __int64 v27; // rdi
  int v29; // edx
  _BYTE *v30; // r15
  unsigned __int64 v31; // rdi
  __int64 v32; // [rsp+8h] [rbp-E8h]
  __int64 v33; // [rsp+8h] [rbp-E8h]
  __int64 v34; // [rsp+10h] [rbp-E0h]
  __int64 v35; // [rsp+18h] [rbp-D8h]
  __int64 v36; // [rsp+20h] [rbp-D0h]
  __int64 v37; // [rsp+28h] [rbp-C8h]
  __int64 v38; // [rsp+30h] [rbp-C0h]
  __int64 v39; // [rsp+38h] [rbp-B8h]
  __int64 v40; // [rsp+40h] [rbp-B0h]
  __int64 v41; // [rsp+48h] [rbp-A8h]
  __int64 v42; // [rsp+50h] [rbp-A0h]
  __int64 v43; // [rsp+58h] [rbp-98h]
  __int64 v44; // [rsp+60h] [rbp-90h]
  __int64 v45; // [rsp+68h] [rbp-88h]
  unsigned int v46; // [rsp+70h] [rbp-80h]
  __int64 *v47; // [rsp+78h] [rbp-78h] BYREF
  __int64 v48; // [rsp+80h] [rbp-70h]
  __int64 v49; // [rsp+88h] [rbp-68h] BYREF
  __int64 v50; // [rsp+90h] [rbp-60h]
  __int64 v51; // [rsp+98h] [rbp-58h]
  unsigned int v52; // [rsp+A0h] [rbp-50h]
  _BYTE *v53; // [rsp+A8h] [rbp-48h] BYREF
  __int64 v54; // [rsp+B0h] [rbp-40h]
  _BYTE v55[56]; // [rsp+B8h] [rbp-38h] BYREF

  v6 = a2;
  v9 = a4;
  v10 = *a2;
  ++a2[9];
  v43 = 1;
  v34 = v10;
  v35 = a2[1];
  v36 = a2[2];
  v37 = a2[3];
  v38 = a2[4];
  v39 = a2[5];
  v40 = a2[6];
  v41 = a2[7];
  v42 = a2[8];
  v44 = a2[10];
  LODWORD(v45) = *((_DWORD *)a2 + 22);
  LODWORD(v10) = *((_DWORD *)a2 + 23);
  v11 = *((_DWORD *)a2 + 28);
  a2[10] = 0;
  HIDWORD(v45) = v10;
  v12 = *((unsigned int *)a2 + 24);
  a2[11] = 0;
  v46 = v12;
  *((_DWORD *)a2 + 24) = 0;
  v47 = &v49;
  v48 = 0;
  if ( v11 )
  {
    sub_23591E0((__int64)&v47, (__int64)(a2 + 13), v12, a4, a5, a6);
    v6 = a2;
  }
  v13 = v6[16];
  v14 = *((_DWORD *)v6 + 40);
  v49 = 1;
  ++v6[15];
  v50 = v13;
  LODWORD(v13) = *((_DWORD *)v6 + 34);
  v6[16] = 0;
  LODWORD(v51) = v13;
  LODWORD(v13) = *((_DWORD *)v6 + 35);
  v53 = v55;
  HIDWORD(v51) = v13;
  v15 = *((unsigned int *)v6 + 36);
  v6[17] = 0;
  v52 = v15;
  *((_DWORD *)v6 + 36) = 0;
  v54 = 0;
  if ( v14 )
    sub_23596A0((__int64)&v53, (__int64)(v6 + 19), v15, a4, a5, a6);
  v16 = sub_22077B0(0xB0u);
  if ( v16 )
  {
    ++v43;
    *(_QWORD *)(v16 + 80) = 1;
    *(_QWORD *)v16 = &unk_4A10E78;
    *(_QWORD *)(v16 + 8) = v34;
    *(_QWORD *)(v16 + 16) = v35;
    *(_QWORD *)(v16 + 24) = v36;
    *(_QWORD *)(v16 + 32) = v37;
    *(_QWORD *)(v16 + 40) = v38;
    *(_QWORD *)(v16 + 48) = v39;
    *(_QWORD *)(v16 + 56) = v40;
    *(_QWORD *)(v16 + 64) = v41;
    *(_QWORD *)(v16 + 72) = v42;
    v19 = v44;
    v44 = 0;
    *(_QWORD *)(v16 + 88) = v19;
    v20 = v45;
    v45 = 0;
    *(_QWORD *)(v16 + 96) = v20;
    LODWORD(v20) = v46;
    v46 = 0;
    *(_DWORD *)(v16 + 104) = v20;
    *(_QWORD *)(v16 + 112) = v16 + 128;
    v21 = (unsigned int)v48;
    *(_QWORD *)(v16 + 120) = 0;
    if ( (_DWORD)v21 )
    {
      v33 = v16;
      sub_23591E0(v16 + 112, (__int64)&v47, v16 + 128, v21, v17, v18);
      v16 = v33;
    }
    v22 = v50;
    ++v49;
    *(_QWORD *)(v16 + 128) = 1;
    *(_QWORD *)(v16 + 136) = v22;
    v50 = 0;
    *(_QWORD *)(v16 + 144) = v51;
    v51 = 0;
    *(_DWORD *)(v16 + 152) = v52;
    *(_QWORD *)(v16 + 160) = v16 + 176;
    v23 = (unsigned int)v54;
    v52 = 0;
    *(_QWORD *)(v16 + 168) = 0;
    if ( !(_DWORD)v23 )
    {
      *(_QWORD *)a1 = v16;
      *(_BYTE *)(a1 + 8) = a3;
      *(_BYTE *)(a1 + 9) = v9;
      v24 = (unsigned __int64)v53;
      goto LABEL_10;
    }
    v32 = v16;
    sub_23596A0(v16 + 160, (__int64)&v53, v23, v21, v17, v18);
    v29 = v54;
    v16 = v32;
  }
  else
  {
    v29 = v54;
  }
  *(_QWORD *)a1 = v16;
  *(_BYTE *)(a1 + 8) = a3;
  v30 = v53;
  *(_BYTE *)(a1 + 9) = v9;
  v24 = (unsigned __int64)&v30[88 * v29];
  if ( v30 != (_BYTE *)v24 )
  {
    do
    {
      v24 -= 88LL;
      v31 = *(_QWORD *)(v24 + 8);
      if ( v31 != v24 + 24 )
        _libc_free(v31);
    }
    while ( v30 != (_BYTE *)v24 );
    v24 = (unsigned __int64)v53;
  }
LABEL_10:
  if ( (_BYTE *)v24 != v55 )
    _libc_free(v24);
  sub_C7D6A0(v50, 16LL * v52, 8);
  v25 = v47;
  v26 = &v47[11 * (unsigned int)v48];
  if ( v47 != v26 )
  {
    do
    {
      v26 -= 11;
      v27 = v26[1];
      if ( (__int64 *)v27 != v26 + 3 )
        _libc_free(v27);
    }
    while ( v25 != v26 );
    v26 = v47;
  }
  if ( v26 != &v49 )
    _libc_free((unsigned __int64)v26);
  sub_C7D6A0(v44, 16LL * v46, 8);
  return a1;
}
