// Function: sub_39145F0
// Address: 0x39145f0
//
__int64 __fastcall sub_39145F0(
        __int64 a1,
        char *a2,
        size_t a3,
        unsigned int a4,
        unsigned __int64 a5,
        unsigned __int64 a6,
        unsigned __int64 a7,
        unsigned __int64 a8,
        unsigned int a9,
        unsigned int a10)
{
  unsigned int v13; // r15d
  __int64 v14; // rdi
  unsigned int v15; // eax
  unsigned __int32 v16; // esi
  __int64 v17; // rdi
  unsigned int v18; // eax
  unsigned __int32 v19; // ecx
  unsigned __int64 v20; // r8
  __int64 v21; // rdi
  int v22; // eax
  unsigned __int64 v23; // rdx
  __int64 v24; // rdi
  unsigned __int64 v25; // rdx
  __int64 v26; // rdi
  unsigned __int64 v27; // rdx
  __int64 v28; // rdi
  unsigned __int64 v29; // rdx
  __int64 v30; // rdi
  unsigned __int32 v31; // edx
  __int64 v32; // rdi
  unsigned __int32 v33; // edx
  __int64 v34; // rdi
  unsigned __int32 v35; // edx
  __int64 v36; // rdi
  unsigned __int32 v38; // edx
  unsigned __int32 v39; // r9d
  __int64 v40; // rdi
  __int64 v41; // rdi
  unsigned __int32 v42; // eax
  __int64 v43; // rdi
  unsigned __int32 v44; // eax
  int v48; // [rsp+1Ch] [rbp-44h]
  char v49[56]; // [rsp+28h] [rbp-38h] BYREF

  v13 = a9;
  (*(void (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 240) + 64LL))(*(_QWORD *)(a1 + 240));
  v14 = *(_QWORD *)(a1 + 240);
  v48 = (*(_BYTE *)(*(_QWORD *)(a1 + 8) + 8LL) & 1) == 0 ? 56 : 72;
  v15 = (*(_BYTE *)(*(_QWORD *)(a1 + 8) + 8LL) & 1) == 0 ? 1 : 25;
  v16 = _byteswap_ulong(v15);
  if ( (unsigned int)(*(_DWORD *)(a1 + 248) - 1) > 1 )
    v15 = v16;
  *(_DWORD *)v49 = v15;
  sub_16E7EE0(v14, v49, 4u);
  v17 = *(_QWORD *)(a1 + 240);
  v18 = v48 + a4 * ((*(_BYTE *)(*(_QWORD *)(a1 + 8) + 8LL) & 1) == 0 ? 68 : 80);
  v19 = _byteswap_ulong(v18);
  if ( (unsigned int)(*(_DWORD *)(a1 + 248) - 1) > 1 )
    v18 = v19;
  *(_DWORD *)v49 = v18;
  sub_16E7EE0(v17, v49, 4u);
  sub_3914570(a1, a2, a3, 16);
  v20 = a5;
  v21 = *(_QWORD *)(a1 + 240);
  v22 = *(_DWORD *)(a1 + 248);
  if ( (*(_BYTE *)(*(_QWORD *)(a1 + 8) + 8LL) & 1) != 0 )
  {
    v23 = _byteswap_uint64(a5);
    if ( (unsigned int)(v22 - 1) > 1 )
      v20 = v23;
    *(_QWORD *)v49 = v20;
    sub_16E7EE0(v21, v49, 8u);
    v24 = *(_QWORD *)(a1 + 240);
    v25 = _byteswap_uint64(a6);
    if ( (unsigned int)(*(_DWORD *)(a1 + 248) - 1) > 1 )
      a6 = v25;
    *(_QWORD *)v49 = a6;
    sub_16E7EE0(v24, v49, 8u);
    v26 = *(_QWORD *)(a1 + 240);
    v27 = _byteswap_uint64(a7);
    if ( (unsigned int)(*(_DWORD *)(a1 + 248) - 1) <= 1 )
      v27 = a7;
    *(_QWORD *)v49 = v27;
    sub_16E7EE0(v26, v49, 8u);
    v28 = *(_QWORD *)(a1 + 240);
    v29 = _byteswap_uint64(a8);
    if ( (unsigned int)(*(_DWORD *)(a1 + 248) - 1) <= 1 )
      v29 = a8;
    *(_QWORD *)v49 = v29;
    sub_16E7EE0(v28, v49, 8u);
  }
  else
  {
    v38 = _byteswap_ulong(a5);
    if ( (unsigned int)(v22 - 1) > 1 )
      LODWORD(v20) = v38;
    *(_DWORD *)v49 = v20;
    sub_16E7EE0(v21, v49, 4u);
    v39 = _byteswap_ulong(a6);
    v40 = *(_QWORD *)(a1 + 240);
    if ( (unsigned int)(*(_DWORD *)(a1 + 248) - 1) > 1 )
      LODWORD(a6) = v39;
    *(_DWORD *)v49 = a6;
    sub_16E7EE0(v40, v49, 4u);
    v41 = *(_QWORD *)(a1 + 240);
    v42 = _byteswap_ulong(a7);
    if ( (unsigned int)(*(_DWORD *)(a1 + 248) - 1) <= 1 )
      v42 = a7;
    *(_DWORD *)v49 = v42;
    sub_16E7EE0(v41, v49, 4u);
    v43 = *(_QWORD *)(a1 + 240);
    v44 = _byteswap_ulong(a8);
    if ( (unsigned int)(*(_DWORD *)(a1 + 248) - 1) <= 1 )
      v44 = a8;
    *(_DWORD *)v49 = v44;
    sub_16E7EE0(v43, v49, 4u);
  }
  v30 = *(_QWORD *)(a1 + 240);
  v31 = _byteswap_ulong(a9);
  if ( (unsigned int)(*(_DWORD *)(a1 + 248) - 1) > 1 )
    v13 = v31;
  *(_DWORD *)v49 = v13;
  sub_16E7EE0(v30, v49, 4u);
  v32 = *(_QWORD *)(a1 + 240);
  v33 = _byteswap_ulong(a10);
  if ( (unsigned int)(*(_DWORD *)(a1 + 248) - 1) <= 1 )
    v33 = a10;
  *(_DWORD *)v49 = v33;
  sub_16E7EE0(v32, v49, 4u);
  v34 = *(_QWORD *)(a1 + 240);
  v35 = _byteswap_ulong(a4);
  if ( (unsigned int)(*(_DWORD *)(a1 + 248) - 1) > 1 )
    a4 = v35;
  *(_DWORD *)v49 = a4;
  sub_16E7EE0(v34, v49, 4u);
  v36 = *(_QWORD *)(a1 + 240);
  *(_DWORD *)v49 = 0;
  return sub_16E7EE0(v36, v49, 4u);
}
