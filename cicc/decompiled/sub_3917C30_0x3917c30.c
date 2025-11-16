// Function: sub_3917C30
// Address: 0x3917c30
//
__int64 __fastcall sub_3917C30(
        __int64 a1,
        _QWORD *a2,
        __int64 a3,
        unsigned __int64 a4,
        unsigned int a5,
        unsigned int a6,
        __int64 a7,
        unsigned int a8)
{
  __int64 v12; // r15
  char v13; // al
  __int64 v14; // rdi
  char *v15; // rsi
  size_t v16; // rdx
  size_t v17; // rdx
  char *v18; // rsi
  __int64 v19; // rdi
  int v20; // eax
  unsigned __int64 v21; // rdx
  unsigned __int64 v22; // rdx
  __int64 v23; // rdi
  __int64 v24; // rdi
  unsigned __int32 v25; // edx
  unsigned int v26; // edx
  unsigned int v27; // eax
  __int64 v28; // rdi
  unsigned __int32 v29; // ecx
  __int64 v30; // rdi
  unsigned int v31; // eax
  unsigned __int32 v32; // eax
  __int64 v33; // rdi
  unsigned __int32 v34; // edx
  __int64 v35; // rdi
  unsigned __int32 v36; // edx
  unsigned int v37; // eax
  unsigned int v38; // eax
  __int64 v39; // rsi
  unsigned int v40; // ecx
  __int64 *v41; // rdx
  __int64 v42; // rdi
  __int64 v43; // rdi
  unsigned __int32 v44; // ecx
  unsigned int v45; // eax
  __int64 v46; // rdi
  unsigned __int32 v47; // ecx
  __int64 result; // rax
  unsigned __int32 v49; // ecx
  __int64 v50; // rdi
  unsigned __int32 v51; // eax
  __int64 v52; // rdi
  int v53; // edx
  int v54; // r8d
  char v56[56]; // [rsp+18h] [rbp-38h] BYREF
  unsigned int v57; // [rsp+60h] [rbp+10h]

  v12 = sub_38D04A0(a2, a3);
  v13 = sub_38D9A30(a3);
  v14 = *(_QWORD *)(a1 + 240);
  if ( v13 )
    a5 = 0;
  (*(void (__fastcall **)(__int64))(*(_QWORD *)v14 + 64LL))(v14);
  v15 = (char *)(a3 + 168);
  v16 = 16;
  if ( !*(_BYTE *)(a3 + 183) )
  {
    v15 = (char *)(a3 + 168);
    v16 = strlen((const char *)(a3 + 168));
  }
  sub_3914570(a1, v15, v16, 16);
  v17 = 16;
  v18 = (char *)(a3 + 152);
  if ( !*(_BYTE *)(a3 + 167) )
  {
    v18 = (char *)(a3 + 152);
    v17 = strlen((const char *)(a3 + 152));
  }
  sub_3914570(a1, v18, v17, 16);
  v19 = *(_QWORD *)(a1 + 240);
  v20 = *(_DWORD *)(a1 + 248);
  if ( (*(_BYTE *)(*(_QWORD *)(a1 + 8) + 8LL) & 1) != 0 )
  {
    v21 = _byteswap_uint64(a4);
    if ( (unsigned int)(v20 - 1) > 1 )
      a4 = v21;
    *(_QWORD *)v56 = a4;
    sub_16E7EE0(v19, v56, 8u);
    v22 = _byteswap_uint64(v12);
    v23 = *(_QWORD *)(a1 + 240);
    if ( (unsigned int)(*(_DWORD *)(a1 + 248) - 1) > 1 )
      v12 = v22;
    *(_QWORD *)v56 = v12;
    sub_16E7EE0(v23, v56, 8u);
  }
  else
  {
    v49 = _byteswap_ulong(a4);
    if ( (unsigned int)(v20 - 1) > 1 )
      LODWORD(a4) = v49;
    *(_DWORD *)v56 = a4;
    sub_16E7EE0(v19, v56, 4u);
    v50 = *(_QWORD *)(a1 + 240);
    v51 = _byteswap_ulong(v12);
    if ( (unsigned int)(*(_DWORD *)(a1 + 248) - 1) > 1 )
      LODWORD(v12) = v51;
    *(_DWORD *)v56 = v12;
    sub_16E7EE0(v50, v56, 4u);
  }
  v24 = *(_QWORD *)(a1 + 240);
  v25 = _byteswap_ulong(a5);
  if ( (unsigned int)(*(_DWORD *)(a1 + 248) - 1) > 1 )
    a5 = v25;
  *(_DWORD *)v56 = a5;
  sub_16E7EE0(v24, v56, 4u);
  v26 = *(_DWORD *)(a3 + 24);
  v27 = -1;
  if ( v26 )
  {
    _BitScanReverse(&v26, v26);
    v27 = 31 - (v26 ^ 0x1F);
  }
  v28 = *(_QWORD *)(a1 + 240);
  v29 = _byteswap_ulong(v27);
  if ( (unsigned int)(*(_DWORD *)(a1 + 248) - 1) > 1 )
    v27 = v29;
  *(_DWORD *)v56 = v27;
  sub_16E7EE0(v28, v56, 4u);
  v30 = *(_QWORD *)(a1 + 240);
  v31 = 0;
  if ( a8 )
    v31 = a7;
  v57 = v31;
  v32 = _byteswap_ulong(v31);
  if ( (unsigned int)(*(_DWORD *)(a1 + 248) - 1) <= 1 )
    v32 = v57;
  *(_DWORD *)v56 = v32;
  sub_16E7EE0(v30, v56, 4u);
  v33 = *(_QWORD *)(a1 + 240);
  v34 = _byteswap_ulong(a8);
  if ( (unsigned int)(*(_DWORD *)(a1 + 248) - 1) <= 1 )
    v34 = a8;
  *(_DWORD *)v56 = v34;
  sub_16E7EE0(v33, v56, 4u);
  v35 = *(_QWORD *)(a1 + 240);
  v36 = _byteswap_ulong(a6);
  if ( (unsigned int)(*(_DWORD *)(a1 + 248) - 1) <= 1 )
    v36 = a6;
  *(_DWORD *)v56 = v36;
  sub_16E7EE0(v35, v56, 4u);
  v37 = *(_DWORD *)(a1 + 72);
  if ( v37 )
  {
    v38 = v37 - 1;
    v39 = *(_QWORD *)(a1 + 56);
    v40 = v38 & (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4));
    v41 = (__int64 *)(v39 + 16LL * v40);
    v42 = *v41;
    if ( a3 == *v41 )
    {
LABEL_29:
      v37 = *((_DWORD *)v41 + 2);
    }
    else
    {
      v53 = 1;
      while ( v42 != -8 )
      {
        v54 = v53 + 1;
        v40 = v38 & (v53 + v40);
        v41 = (__int64 *)(v39 + 16LL * v40);
        v42 = *v41;
        if ( a3 == *v41 )
          goto LABEL_29;
        v53 = v54;
      }
      v37 = 0;
    }
  }
  v43 = *(_QWORD *)(a1 + 240);
  v44 = _byteswap_ulong(v37);
  if ( (unsigned int)(*(_DWORD *)(a1 + 248) - 1) > 1 )
    v37 = v44;
  *(_DWORD *)v56 = v37;
  sub_16E7EE0(v43, v56, 4u);
  v45 = *(_DWORD *)(a3 + 188);
  v46 = *(_QWORD *)(a1 + 240);
  v47 = _byteswap_ulong(v45);
  if ( (unsigned int)(*(_DWORD *)(a1 + 248) - 1) > 1 )
    v45 = v47;
  *(_DWORD *)v56 = v45;
  sub_16E7EE0(v46, v56, 4u);
  result = *(_QWORD *)(a1 + 8);
  if ( (*(_BYTE *)(result + 8) & 1) != 0 )
  {
    v52 = *(_QWORD *)(a1 + 240);
    *(_DWORD *)v56 = 0;
    return sub_16E7EE0(v52, v56, 4u);
  }
  return result;
}
