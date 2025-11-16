// Function: sub_29391F0
// Address: 0x29391f0
//
__int64 __fastcall sub_29391F0(__int64 a1, char *a2, __int64 a3, __int64 a4)
{
  __int64 v6; // r14
  __int64 v7; // r14
  char *v8; // rax
  unsigned __int16 v9; // r14
  __int64 v10; // rdx
  __int64 v11; // rcx
  __int64 v12; // r8
  void *v13; // rsi
  void *v14; // r9
  int v15; // eax
  void **v16; // rdx
  __int64 v17; // rdx
  __int64 v18; // rcx
  __int64 v19; // r8
  __int64 v20; // r9
  _QWORD *v21; // rbx
  _QWORD *v22; // r15
  void (__fastcall *v23)(_QWORD *, _QWORD *, __int64); // rax
  __int64 v24; // rax
  char v26; // [rsp+8h] [rbp-918h]
  __int64 v27; // [rsp+18h] [rbp-908h]
  unsigned __int64 v28[2]; // [rsp+20h] [rbp-900h] BYREF
  _BYTE v29[512]; // [rsp+30h] [rbp-8F0h] BYREF
  __int64 v30; // [rsp+230h] [rbp-6F0h]
  __int64 v31; // [rsp+238h] [rbp-6E8h]
  __int64 v32; // [rsp+240h] [rbp-6E0h]
  __int64 v33; // [rsp+248h] [rbp-6D8h]
  char v34; // [rsp+250h] [rbp-6D0h]
  __int64 v35; // [rsp+258h] [rbp-6C8h]
  char *v36; // [rsp+260h] [rbp-6C0h]
  __int64 v37; // [rsp+268h] [rbp-6B8h]
  int v38; // [rsp+270h] [rbp-6B0h]
  char v39; // [rsp+274h] [rbp-6ACh]
  char v40; // [rsp+278h] [rbp-6A8h] BYREF
  __int16 v41; // [rsp+2B8h] [rbp-668h]
  _QWORD *v42; // [rsp+2C0h] [rbp-660h]
  _QWORD *v43; // [rsp+2C8h] [rbp-658h]
  __int64 v44; // [rsp+2D0h] [rbp-650h]
  __int64 v45; // [rsp+2E0h] [rbp-640h] BYREF
  void **v46; // [rsp+2E8h] [rbp-638h]
  __int64 v47; // [rsp+2F0h] [rbp-630h]
  __int64 v48; // [rsp+2F8h] [rbp-628h]
  void *v49; // [rsp+300h] [rbp-620h] BYREF
  __int64 v50; // [rsp+308h] [rbp-618h] BYREF
  __int64 v51; // [rsp+310h] [rbp-610h] BYREF
  __int64 *v52; // [rsp+318h] [rbp-608h]
  __int64 v53; // [rsp+320h] [rbp-600h]
  char *v54; // [rsp+328h] [rbp-5F8h]
  __int64 v55; // [rsp+330h] [rbp-5F0h] BYREF
  char v56; // [rsp+338h] [rbp-5E8h] BYREF
  char *v57; // [rsp+3B8h] [rbp-568h]
  __int64 v58; // [rsp+3C0h] [rbp-560h]
  char v59; // [rsp+3C8h] [rbp-558h] BYREF
  __int64 v60; // [rsp+488h] [rbp-498h]
  __int64 v61; // [rsp+490h] [rbp-490h]
  __int64 v62; // [rsp+498h] [rbp-488h]
  __int64 v63; // [rsp+4A0h] [rbp-480h]
  char *v64; // [rsp+4A8h] [rbp-478h]
  __int64 v65; // [rsp+4B0h] [rbp-470h]
  char v66; // [rsp+4B8h] [rbp-468h] BYREF
  __int64 v67; // [rsp+538h] [rbp-3E8h]
  char *v68; // [rsp+540h] [rbp-3E0h]
  __int64 v69; // [rsp+548h] [rbp-3D8h]
  int v70; // [rsp+550h] [rbp-3D0h]
  char v71; // [rsp+554h] [rbp-3CCh]
  char v72; // [rsp+558h] [rbp-3C8h] BYREF
  char *v73; // [rsp+5D8h] [rbp-348h]
  __int64 v74; // [rsp+5E0h] [rbp-340h]
  char v75; // [rsp+5E8h] [rbp-338h] BYREF
  __int64 v76; // [rsp+618h] [rbp-308h]
  __int64 v77; // [rsp+620h] [rbp-300h]
  __int64 v78; // [rsp+628h] [rbp-2F8h]
  __int64 v79; // [rsp+630h] [rbp-2F0h]
  char *v80; // [rsp+638h] [rbp-2E8h]
  __int64 v81; // [rsp+640h] [rbp-2E0h]
  char v82; // [rsp+648h] [rbp-2D8h] BYREF
  __int64 v83; // [rsp+688h] [rbp-298h]
  __int64 v84; // [rsp+690h] [rbp-290h]
  char v85; // [rsp+698h] [rbp-288h] BYREF
  _QWORD v86[2]; // [rsp+718h] [rbp-208h] BYREF
  char v87; // [rsp+728h] [rbp-1F8h] BYREF

  v6 = sub_BC1CD0(a4, &unk_4F81450, a3);
  v27 = sub_BC1CD0(a4, &unk_4F86630, a3);
  v28[0] = (unsigned __int64)v29;
  v36 = &v40;
  v28[1] = 0x1000000000LL;
  v32 = v6 + 8;
  v30 = 0;
  v31 = 0;
  v33 = 0;
  v34 = 1;
  v35 = 0;
  v37 = 8;
  v38 = 0;
  v39 = 1;
  v41 = 0;
  v42 = 0;
  v43 = 0;
  v44 = 0;
  v7 = sub_BC1CD0(a4, &unk_4F86540, a3) + 8;
  v26 = *a2;
  v45 = sub_B2BE50(a3);
  LOBYTE(v49) = v26;
  v64 = &v66;
  v54 = &v56;
  v68 = &v72;
  v57 = &v59;
  v73 = &v75;
  v48 = v27 + 8;
  v58 = 0x800000000LL;
  v46 = (void **)v28;
  v47 = v7;
  v50 = 0;
  v51 = 0;
  v52 = 0;
  v53 = 0;
  v55 = 0x1000000000LL;
  v60 = 0;
  v61 = 0;
  v62 = 0;
  v63 = 0;
  v65 = 0x1000000000LL;
  v67 = 0;
  v69 = 16;
  v70 = 0;
  v71 = 1;
  v74 = 0x600000000LL;
  v76 = 0;
  v77 = 0;
  v78 = 0;
  v79 = 0;
  v80 = &v82;
  v83 = 0;
  v84 = 1;
  v81 = 0x800000000LL;
  v8 = &v85;
  do
  {
    *(_QWORD *)v8 = -4096;
    v8 += 16;
  }
  while ( v8 != (char *)v86 );
  v86[0] = &v87;
  v86[1] = 0x800000000LL;
  v9 = sub_29385B0((__int64)&v45, a3);
  sub_2913FD0((__int64)&v45);
  v13 = (void *)(a1 + 32);
  v14 = &unk_4F81450;
  if ( !(_BYTE)v9 )
  {
    *(_QWORD *)(a1 + 8) = v13;
    *(_QWORD *)(a1 + 16) = 0x100000002LL;
    *(_QWORD *)(a1 + 48) = 0;
    *(_QWORD *)(a1 + 56) = a1 + 80;
    *(_QWORD *)(a1 + 64) = 2;
    *(_DWORD *)(a1 + 72) = 0;
    *(_BYTE *)(a1 + 76) = 1;
    *(_DWORD *)(a1 + 24) = 0;
    *(_BYTE *)(a1 + 28) = 1;
    *(_QWORD *)(a1 + 32) = &qword_4F82400;
    *(_QWORD *)a1 = 1;
    goto LABEL_11;
  }
  v15 = 0;
  v45 = 0;
  v16 = &v49;
  v46 = &v49;
  LODWORD(v47) = 2;
  LODWORD(v48) = 0;
  BYTE4(v48) = 1;
  v51 = 0;
  v52 = &v55;
  v53 = 2;
  LODWORD(v54) = 0;
  BYTE4(v54) = 1;
  if ( !HIBYTE(v9) )
  {
    HIDWORD(v47) = 1;
    v45 = 1;
    v49 = &unk_4F82408;
    if ( &unk_4F82408 == (_UNKNOWN *)&qword_4F82400 || v49 == &unk_4F81450 )
      goto LABEL_9;
    v16 = (void **)&v50;
    v15 = 1;
  }
  HIDWORD(v47) = v15 + 1;
  *v16 = &unk_4F81450;
  ++v45;
LABEL_9:
  sub_C8CF70(a1, v13, 2, (__int64)&v49, (__int64)&v45);
  v13 = (void *)(a1 + 80);
  sub_C8CF70(a1 + 48, (void *)(a1 + 80), 2, (__int64)&v55, (__int64)&v51);
  if ( BYTE4(v54) )
  {
    if ( BYTE4(v48) )
      goto LABEL_11;
    goto LABEL_26;
  }
  _libc_free((unsigned __int64)v52);
  if ( !BYTE4(v48) )
LABEL_26:
    _libc_free((unsigned __int64)v46);
LABEL_11:
  sub_FFCE90((__int64)v28, (__int64)v13, v10, v11, v12, (__int64)v14);
  sub_FFD870((__int64)v28, (__int64)v13, v17, v18, v19, v20);
  sub_FFBC40((__int64)v28, (__int64)v13);
  v21 = v43;
  v22 = v42;
  if ( v43 != v42 )
  {
    do
    {
      v23 = (void (__fastcall *)(_QWORD *, _QWORD *, __int64))v22[7];
      *v22 = &unk_49E5048;
      if ( v23 )
        v23(v22 + 5, v22 + 5, 3);
      *v22 = &unk_49DB368;
      v24 = v22[3];
      if ( v24 != -4096 && v24 != 0 && v24 != -8192 )
        sub_BD60C0(v22 + 1);
      v22 += 9;
    }
    while ( v21 != v22 );
    v22 = v42;
  }
  if ( v22 )
    j_j___libc_free_0((unsigned __int64)v22);
  if ( !v39 )
    _libc_free((unsigned __int64)v36);
  if ( (_BYTE *)v28[0] != v29 )
    _libc_free(v28[0]);
  return a1;
}
