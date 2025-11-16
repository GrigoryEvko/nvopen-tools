// Function: sub_124BE70
// Address: 0x124be70
//
__int64 __fastcall sub_124BE70(
        __int64 a1,
        unsigned int a2,
        unsigned int a3,
        unsigned __int64 a4,
        unsigned __int64 a5,
        unsigned __int64 a6,
        unsigned int a7,
        unsigned int a8,
        unsigned __int16 a9,
        unsigned __int64 a10)
{
  bool v14; // zf
  __int64 v15; // rdi
  __int64 v16; // rdi
  unsigned __int32 v17; // r11d
  __int64 v18; // rdi
  __int64 v19; // rdi
  int v20; // eax
  __int64 v21; // rdi
  unsigned __int64 v22; // rdx
  int v23; // eax
  __int64 v24; // rdi
  unsigned __int64 v25; // rdx
  __int64 v26; // rdi
  __int64 v27; // rdi
  __int64 v28; // rax
  int v29; // edx
  __int64 v30; // rdi
  unsigned __int64 v31; // rcx
  int v32; // eax
  __int64 v33; // rdi
  unsigned __int64 v34; // rdx
  unsigned __int32 v36; // r9d
  unsigned __int32 v37; // r8d
  __int64 v38; // rdi
  unsigned __int32 v39; // eax
  unsigned __int32 v40; // edx
  unsigned __int32 v41; // ecx
  unsigned __int8 v43[56]; // [rsp+18h] [rbp-38h] BYREF

  v14 = *(_DWORD *)(a1 + 16) == 1;
  v15 = *(_QWORD *)(a1 + 8);
  if ( !v14 )
    a2 = _byteswap_ulong(a2);
  *(_DWORD *)v43 = a2;
  sub_CB6200(v15, v43, 4u);
  v16 = *(_QWORD *)(a1 + 8);
  v17 = a3;
  if ( *(_DWORD *)(a1 + 16) != 1 )
    v17 = _byteswap_ulong(a3);
  *(_DWORD *)v43 = v17;
  sub_CB6200(v16, v43, 4u);
  if ( (*(_BYTE *)(*(_QWORD *)(*(_QWORD *)a1 + 112LL) + 12LL) & 2) != 0 )
  {
    v18 = *(_QWORD *)(a1 + 8);
    if ( *(_DWORD *)(a1 + 16) != 1 )
      a4 = _byteswap_uint64(a4);
    *(_QWORD *)v43 = a4;
    sub_CB6200(v18, v43, 8u);
  }
  else
  {
    v38 = *(_QWORD *)(a1 + 8);
    v39 = a4;
    if ( *(_DWORD *)(a1 + 16) != 1 )
      v39 = _byteswap_ulong(a4);
    *(_DWORD *)v43 = v39;
    sub_CB6200(v38, v43, 4u);
  }
  v19 = *(_QWORD *)(a1 + 8);
  if ( (*(_BYTE *)(*(_QWORD *)(*(_QWORD *)a1 + 112LL) + 12LL) & 2) != 0 )
  {
    *(_QWORD *)v43 = 0;
    sub_CB6200(v19, v43, 8u);
  }
  else
  {
    *(_DWORD *)v43 = 0;
    sub_CB6200(v19, v43, 4u);
  }
  v20 = *(_DWORD *)(a1 + 16);
  v21 = *(_QWORD *)(a1 + 8);
  if ( (*(_BYTE *)(*(_QWORD *)(*(_QWORD *)a1 + 112LL) + 12LL) & 2) != 0 )
  {
    v22 = _byteswap_uint64(a5);
    if ( v20 != 1 )
      a5 = v22;
    *(_QWORD *)v43 = a5;
    sub_CB6200(v21, v43, 8u);
  }
  else
  {
    v37 = _byteswap_ulong(a5);
    if ( v20 != 1 )
      LODWORD(a5) = v37;
    *(_DWORD *)v43 = a5;
    sub_CB6200(v21, v43, 4u);
  }
  v23 = *(_DWORD *)(a1 + 16);
  v24 = *(_QWORD *)(a1 + 8);
  if ( (*(_BYTE *)(*(_QWORD *)(*(_QWORD *)a1 + 112LL) + 12LL) & 2) != 0 )
  {
    v25 = _byteswap_uint64(a6);
    if ( v23 != 1 )
      a6 = v25;
    *(_QWORD *)v43 = a6;
    sub_CB6200(v24, v43, 8u);
  }
  else
  {
    v36 = _byteswap_ulong(a6);
    if ( v23 != 1 )
      LODWORD(a6) = v36;
    *(_DWORD *)v43 = a6;
    sub_CB6200(v24, v43, 4u);
  }
  v26 = *(_QWORD *)(a1 + 8);
  if ( *(_DWORD *)(a1 + 16) != 1 )
    a7 = _byteswap_ulong(a7);
  *(_DWORD *)v43 = a7;
  sub_CB6200(v26, v43, 4u);
  v27 = *(_QWORD *)(a1 + 8);
  if ( *(_DWORD *)(a1 + 16) != 1 )
    a8 = _byteswap_ulong(a8);
  *(_DWORD *)v43 = a8;
  sub_CB6200(v27, v43, 4u);
  v28 = 0;
  if ( HIBYTE(a9) )
    v28 = 1LL << a9;
  v29 = *(_DWORD *)(a1 + 16);
  v30 = *(_QWORD *)(a1 + 8);
  if ( (*(_BYTE *)(*(_QWORD *)(*(_QWORD *)a1 + 112LL) + 12LL) & 2) != 0 )
  {
    v31 = _byteswap_uint64(v28);
    if ( v29 != 1 )
      v28 = v31;
    *(_QWORD *)v43 = v28;
    sub_CB6200(v30, v43, 8u);
  }
  else
  {
    v41 = _byteswap_ulong(v28);
    if ( v29 != 1 )
      LODWORD(v28) = v41;
    *(_DWORD *)v43 = v28;
    sub_CB6200(v30, v43, 4u);
  }
  v32 = *(_DWORD *)(a1 + 16);
  v33 = *(_QWORD *)(a1 + 8);
  if ( (*(_BYTE *)(*(_QWORD *)(*(_QWORD *)a1 + 112LL) + 12LL) & 2) != 0 )
  {
    v34 = _byteswap_uint64(a10);
    if ( v32 == 1 )
      v34 = a10;
    *(_QWORD *)v43 = v34;
    return sub_CB6200(v33, v43, 8u);
  }
  else
  {
    v40 = _byteswap_ulong(a10);
    if ( v32 == 1 )
      v40 = a10;
    *(_DWORD *)v43 = v40;
    return sub_CB6200(v33, v43, 4u);
  }
}
