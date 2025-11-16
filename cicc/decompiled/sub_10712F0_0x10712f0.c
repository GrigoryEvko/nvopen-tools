// Function: sub_10712F0
// Address: 0x10712f0
//
__int64 __fastcall sub_10712F0(
        __int64 a1,
        unsigned __int8 *a2,
        size_t a3,
        unsigned int a4,
        unsigned __int64 a5,
        unsigned __int64 a6,
        unsigned __int64 a7,
        unsigned __int64 a8,
        unsigned int a9,
        unsigned int a10)
{
  __int64 v14; // rdi
  unsigned int v15; // eax
  __int64 v16; // rdi
  unsigned int v17; // eax
  __int64 v18; // rdi
  __int64 v19; // rdi
  __int64 v20; // rdi
  __int64 v21; // rdi
  __int64 v22; // rdi
  __int64 v23; // rdi
  __int64 v24; // rdi
  __int64 v25; // rdi
  __int64 v27; // rdi
  unsigned __int32 v28; // eax
  __int64 v29; // rdi
  unsigned __int32 v30; // eax
  unsigned __int32 v31; // eax
  __int64 v32; // rdi
  unsigned __int32 v33; // eax
  __int64 v34; // rdi
  int v35; // [rsp+Ch] [rbp-54h]
  unsigned __int8 v37[56]; // [rsp+28h] [rbp-38h] BYREF

  (*(void (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 2048) + 80LL))(*(_QWORD *)(a1 + 2048));
  v14 = *(_QWORD *)(a1 + 2048);
  v15 = (*(_BYTE *)(*(_QWORD *)(a1 + 104) + 8LL) & 1) == 0 ? 1 : 25;
  if ( *(_DWORD *)(a1 + 2056) != 1 )
    v15 = _byteswap_ulong(v15);
  v35 = (*(_BYTE *)(*(_QWORD *)(a1 + 104) + 8LL) & 1) == 0 ? 56 : 72;
  *(_DWORD *)v37 = v15;
  sub_CB6200(v14, v37, 4u);
  v16 = *(_QWORD *)(a1 + 2048);
  v17 = v35 + a4 * ((*(_BYTE *)(*(_QWORD *)(a1 + 104) + 8LL) & 1) == 0 ? 68 : 80);
  if ( *(_DWORD *)(a1 + 2056) != 1 )
    v17 = _byteswap_ulong(v17);
  *(_DWORD *)v37 = v17;
  sub_CB6200(v16, v37, 4u);
  sub_1071270(a1, a2, a3, 16);
  if ( (*(_BYTE *)(*(_QWORD *)(a1 + 104) + 8LL) & 1) != 0 )
  {
    v18 = *(_QWORD *)(a1 + 2048);
    if ( *(_DWORD *)(a1 + 2056) != 1 )
      a5 = _byteswap_uint64(a5);
    *(_QWORD *)v37 = a5;
    sub_CB6200(v18, v37, 8u);
    v19 = *(_QWORD *)(a1 + 2048);
    if ( *(_DWORD *)(a1 + 2056) != 1 )
      a6 = _byteswap_uint64(a6);
    *(_QWORD *)v37 = a6;
    sub_CB6200(v19, v37, 8u);
    v20 = *(_QWORD *)(a1 + 2048);
    if ( *(_DWORD *)(a1 + 2056) != 1 )
      a7 = _byteswap_uint64(a7);
    *(_QWORD *)v37 = a7;
    sub_CB6200(v20, v37, 8u);
    v21 = *(_QWORD *)(a1 + 2048);
    if ( *(_DWORD *)(a1 + 2056) != 1 )
      a8 = _byteswap_uint64(a8);
    *(_QWORD *)v37 = a8;
    sub_CB6200(v21, v37, 8u);
  }
  else
  {
    v27 = *(_QWORD *)(a1 + 2048);
    v28 = a5;
    if ( *(_DWORD *)(a1 + 2056) != 1 )
      v28 = _byteswap_ulong(a5);
    *(_DWORD *)v37 = v28;
    sub_CB6200(v27, v37, 4u);
    v29 = *(_QWORD *)(a1 + 2048);
    v30 = a6;
    if ( *(_DWORD *)(a1 + 2056) != 1 )
      v30 = _byteswap_ulong(a6);
    *(_DWORD *)v37 = v30;
    sub_CB6200(v29, v37, 4u);
    v31 = a7;
    v32 = *(_QWORD *)(a1 + 2048);
    if ( *(_DWORD *)(a1 + 2056) != 1 )
      v31 = _byteswap_ulong(a7);
    *(_DWORD *)v37 = v31;
    sub_CB6200(v32, v37, 4u);
    v33 = a8;
    v34 = *(_QWORD *)(a1 + 2048);
    if ( *(_DWORD *)(a1 + 2056) != 1 )
      v33 = _byteswap_ulong(a8);
    *(_DWORD *)v37 = v33;
    sub_CB6200(v34, v37, 4u);
  }
  v22 = *(_QWORD *)(a1 + 2048);
  if ( *(_DWORD *)(a1 + 2056) != 1 )
    a9 = _byteswap_ulong(a9);
  *(_DWORD *)v37 = a9;
  sub_CB6200(v22, v37, 4u);
  v23 = *(_QWORD *)(a1 + 2048);
  if ( *(_DWORD *)(a1 + 2056) != 1 )
    a10 = _byteswap_ulong(a10);
  *(_DWORD *)v37 = a10;
  sub_CB6200(v23, v37, 4u);
  v24 = *(_QWORD *)(a1 + 2048);
  if ( *(_DWORD *)(a1 + 2056) != 1 )
    a4 = _byteswap_ulong(a4);
  *(_DWORD *)v37 = a4;
  sub_CB6200(v24, v37, 4u);
  v25 = *(_QWORD *)(a1 + 2048);
  *(_DWORD *)v37 = 0;
  return sub_CB6200(v25, v37, 4u);
}
