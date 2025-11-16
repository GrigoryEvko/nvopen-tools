// Function: sub_392E0B0
// Address: 0x392e0b0
//
__int64 __fastcall sub_392E0B0(
        __int64 a1,
        unsigned int a2,
        unsigned int a3,
        unsigned __int64 a4,
        unsigned __int64 a5,
        unsigned __int64 a6,
        unsigned int a7,
        unsigned int a8,
        unsigned __int64 a9,
        unsigned __int64 a10)
{
  int v13; // eax
  __int64 v14; // rdi
  unsigned int v15; // r13d
  bool v16; // cc
  unsigned __int32 v17; // eax
  unsigned int v18; // r11d
  __int64 v19; // rdi
  unsigned __int32 v20; // edx
  unsigned __int64 v21; // rcx
  __int64 v22; // rdi
  int v23; // eax
  unsigned __int64 v24; // rdx
  __int64 v25; // rdi
  __int64 v26; // rdi
  unsigned int v27; // eax
  unsigned __int64 v28; // rdx
  __int64 v29; // rdi
  unsigned int v30; // eax
  unsigned __int64 v31; // rdx
  __int64 v32; // rdi
  unsigned __int32 v33; // edx
  __int64 v34; // rdi
  unsigned __int32 v35; // edx
  __int64 v36; // rdi
  int v37; // eax
  unsigned __int64 v38; // rdx
  __int64 v39; // rdi
  unsigned __int64 v40; // rax
  unsigned __int32 v42; // edx
  unsigned __int32 v43; // eax
  unsigned __int32 v44; // edx
  unsigned __int32 v45; // r9d
  unsigned __int32 v46; // r8d
  char v49[56]; // [rsp+18h] [rbp-38h] BYREF

  v13 = *(_DWORD *)(a1 + 16);
  v14 = *(_QWORD *)(a1 + 8);
  v15 = a8;
  v16 = (unsigned int)(v13 - 1) <= 1;
  v17 = _byteswap_ulong(a2);
  if ( !v16 )
    a2 = v17;
  *(_DWORD *)v49 = a2;
  sub_16E7EE0(v14, v49, 4u);
  v18 = a3;
  v19 = *(_QWORD *)(a1 + 8);
  v20 = _byteswap_ulong(a3);
  if ( (unsigned int)(*(_DWORD *)(a1 + 16) - 1) > 1 )
    v18 = v20;
  *(_DWORD *)v49 = v18;
  sub_16E7EE0(v19, v49, 4u);
  v21 = a4;
  v22 = *(_QWORD *)(a1 + 8);
  v23 = *(_DWORD *)(a1 + 16);
  if ( (*(_BYTE *)(*(_QWORD *)(*(_QWORD *)a1 + 8LL) + 12LL) & 2) != 0 )
  {
    v24 = _byteswap_uint64(a4);
    if ( (unsigned int)(v23 - 1) > 1 )
      v21 = v24;
    *(_QWORD *)v49 = v21;
    sub_16E7EE0(v22, v49, 8u);
  }
  else
  {
    v42 = _byteswap_ulong(a4);
    if ( (unsigned int)(v23 - 1) > 1 )
      LODWORD(v21) = v42;
    *(_DWORD *)v49 = v21;
    sub_16E7EE0(v22, v49, 4u);
  }
  v25 = *(_QWORD *)(a1 + 8);
  if ( (*(_BYTE *)(*(_QWORD *)(*(_QWORD *)a1 + 8LL) + 12LL) & 2) != 0 )
  {
    *(_QWORD *)v49 = 0;
    sub_16E7EE0(v25, v49, 8u);
  }
  else
  {
    *(_DWORD *)v49 = 0;
    sub_16E7EE0(v25, v49, 4u);
  }
  v26 = *(_QWORD *)(a1 + 8);
  v27 = *(_DWORD *)(a1 + 16) - 1;
  if ( (*(_BYTE *)(*(_QWORD *)(*(_QWORD *)a1 + 8LL) + 12LL) & 2) != 0 )
  {
    v28 = _byteswap_uint64(a5);
    if ( v27 > 1 )
      a5 = v28;
    *(_QWORD *)v49 = a5;
    sub_16E7EE0(v26, v49, 8u);
  }
  else
  {
    v46 = _byteswap_ulong(a5);
    if ( v27 > 1 )
      LODWORD(a5) = v46;
    *(_DWORD *)v49 = a5;
    sub_16E7EE0(v26, v49, 4u);
  }
  v29 = *(_QWORD *)(a1 + 8);
  v30 = *(_DWORD *)(a1 + 16) - 1;
  if ( (*(_BYTE *)(*(_QWORD *)(*(_QWORD *)a1 + 8LL) + 12LL) & 2) != 0 )
  {
    v31 = _byteswap_uint64(a6);
    if ( v30 > 1 )
      a6 = v31;
    *(_QWORD *)v49 = a6;
    sub_16E7EE0(v29, v49, 8u);
  }
  else
  {
    v45 = _byteswap_ulong(a6);
    if ( v30 > 1 )
      LODWORD(a6) = v45;
    *(_DWORD *)v49 = a6;
    sub_16E7EE0(v29, v49, 4u);
  }
  v32 = *(_QWORD *)(a1 + 8);
  v33 = _byteswap_ulong(a7);
  if ( (unsigned int)(*(_DWORD *)(a1 + 16) - 1) <= 1 )
    v33 = a7;
  *(_DWORD *)v49 = v33;
  sub_16E7EE0(v32, v49, 4u);
  v34 = *(_QWORD *)(a1 + 8);
  v35 = _byteswap_ulong(a8);
  if ( (unsigned int)(*(_DWORD *)(a1 + 16) - 1) > 1 )
    v15 = v35;
  *(_DWORD *)v49 = v15;
  sub_16E7EE0(v34, v49, 4u);
  v36 = *(_QWORD *)(a1 + 8);
  v37 = *(_DWORD *)(a1 + 16);
  if ( (*(_BYTE *)(*(_QWORD *)(*(_QWORD *)a1 + 8LL) + 12LL) & 2) != 0 )
  {
    v38 = _byteswap_uint64(a9);
    if ( (unsigned int)(v37 - 1) <= 1 )
      v38 = a9;
    *(_QWORD *)v49 = v38;
    sub_16E7EE0(v36, v49, 8u);
  }
  else
  {
    v44 = _byteswap_ulong(a9);
    if ( (unsigned int)(v37 - 1) <= 1 )
      v44 = a9;
    *(_DWORD *)v49 = v44;
    sub_16E7EE0(v36, v49, 4u);
  }
  v39 = *(_QWORD *)(a1 + 8);
  if ( (*(_BYTE *)(*(_QWORD *)(*(_QWORD *)a1 + 8LL) + 12LL) & 2) != 0 )
  {
    v40 = _byteswap_uint64(a10);
    if ( (unsigned int)(*(_DWORD *)(a1 + 16) - 1) <= 1 )
      v40 = a10;
    *(_QWORD *)v49 = v40;
    return sub_16E7EE0(v39, v49, 8u);
  }
  else
  {
    v43 = _byteswap_ulong(a10);
    if ( (unsigned int)(*(_DWORD *)(a1 + 16) - 1) <= 1 )
      v43 = a10;
    *(_DWORD *)v49 = v43;
    return sub_16E7EE0(v39, v49, 4u);
  }
}
