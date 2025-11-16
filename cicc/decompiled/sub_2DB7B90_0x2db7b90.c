// Function: sub_2DB7B90
// Address: 0x2db7b90
//
__int64 __fastcall sub_2DB7B90(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v7; // rdx
  __int64 v8; // rcx
  __int64 v9; // r8
  __int64 v10; // r9
  __int64 v11; // rdx
  __int64 v12; // rcx
  __int64 v13; // r8
  __int64 v14; // r9
  __int64 v15; // rdx
  __int64 v16; // rcx
  __int64 v17; // r8
  __int64 v18; // r9
  _BYTE v19[8]; // [rsp+30h] [rbp-400h] BYREF
  unsigned __int64 v20; // [rsp+38h] [rbp-3F8h]
  char v21; // [rsp+4Ch] [rbp-3E4h]
  _BYTE v22[16]; // [rsp+50h] [rbp-3E0h] BYREF
  _BYTE v23[8]; // [rsp+60h] [rbp-3D0h] BYREF
  unsigned __int64 v24; // [rsp+68h] [rbp-3C8h]
  char v25; // [rsp+7Ch] [rbp-3B4h]
  _BYTE v26[16]; // [rsp+80h] [rbp-3B0h] BYREF
  __m128i v27; // [rsp+90h] [rbp-3A0h] BYREF
  __int64 v28; // [rsp+F0h] [rbp-340h]
  __int64 v29; // [rsp+F8h] [rbp-338h]
  __int64 v30; // [rsp+100h] [rbp-330h]
  __int64 v31; // [rsp+108h] [rbp-328h]
  __int64 v32; // [rsp+110h] [rbp-320h]
  _BYTE *v33; // [rsp+150h] [rbp-2E0h]
  __int64 v34; // [rsp+158h] [rbp-2D8h]
  _BYTE v35[256]; // [rsp+160h] [rbp-2D0h] BYREF
  _BYTE *v36; // [rsp+260h] [rbp-1D0h]
  __int64 v37; // [rsp+268h] [rbp-1C8h]
  _BYTE v38[160]; // [rsp+270h] [rbp-1C0h] BYREF
  __int64 v39; // [rsp+310h] [rbp-120h]
  char *v40; // [rsp+318h] [rbp-118h]
  __int64 v41; // [rsp+320h] [rbp-110h]
  int v42; // [rsp+328h] [rbp-108h]
  char v43; // [rsp+32Ch] [rbp-104h]
  char v44; // [rsp+330h] [rbp-100h] BYREF
  _BYTE *v45; // [rsp+370h] [rbp-C0h]
  __int64 v46; // [rsp+378h] [rbp-B8h]
  _BYTE v47[48]; // [rsp+380h] [rbp-B0h] BYREF
  int v48; // [rsp+3B0h] [rbp-80h]
  _BYTE *v49; // [rsp+3B8h] [rbp-78h]
  __int64 v50; // [rsp+3C0h] [rbp-70h]
  _BYTE v51[32]; // [rsp+3C8h] [rbp-68h] BYREF
  unsigned __int64 v52; // [rsp+3E8h] [rbp-48h]
  int v53; // [rsp+3F0h] [rbp-40h]
  __int64 v54; // [rsp+3F8h] [rbp-38h]

  v29 = sub_2EB2140(a4, &unk_501FE48) + 8;
  v30 = sub_2EB2140(a4, &unk_50208B0) + 8;
  v31 = sub_2EB2140(a4, &unk_5022350) + 8;
  v33 = v35;
  v37 = 0x400000000LL;
  v27 = 0u;
  v28 = 0;
  v32 = 0;
  v34 = 0x800000000LL;
  v36 = v38;
  v39 = 0;
  v40 = &v44;
  v41 = 8;
  v42 = 0;
  v43 = 1;
  v45 = v47;
  v46 = 0x600000000LL;
  v48 = 0;
  v49 = v51;
  v50 = 0x800000000LL;
  v52 = 0;
  v53 = 0;
  v54 = 0;
  if ( (unsigned __int8)sub_2DB5D20(&v27, a3) )
  {
    sub_2EAFFB0(v19);
    sub_2DB43C0((__int64)v19, (__int64)&unk_501FE48, v7, v8, v9, v10);
    sub_2DB43C0((__int64)v19, (__int64)&unk_50208B0, v11, v12, v13, v14);
    sub_2DB43C0((__int64)v19, (__int64)&unk_5022350, v15, v16, v17, v18);
    sub_C8CF70(a1, (void *)(a1 + 32), 2, (__int64)v22, (__int64)v19);
    sub_C8CF70(a1 + 48, (void *)(a1 + 80), 2, (__int64)v26, (__int64)v23);
    if ( !v25 )
      _libc_free(v24);
    if ( !v21 )
      _libc_free(v20);
  }
  else
  {
    *(_QWORD *)(a1 + 8) = a1 + 32;
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
  }
  if ( v52 )
    _libc_free(v52);
  if ( v49 != v51 )
    _libc_free((unsigned __int64)v49);
  if ( v45 != v47 )
    _libc_free((unsigned __int64)v45);
  if ( !v43 )
    _libc_free((unsigned __int64)v40);
  if ( v36 != v38 )
    _libc_free((unsigned __int64)v36);
  if ( v33 != v35 )
    _libc_free((unsigned __int64)v33);
  return a1;
}
