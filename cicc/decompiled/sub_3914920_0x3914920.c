// Function: sub_3914920
// Address: 0x3914920
//
__int64 __fastcall sub_3914920(__int64 a1, unsigned int a2, unsigned int a3, unsigned int a4, unsigned int a5)
{
  __int64 v9; // rdi
  __int64 v10; // rdi
  unsigned int v11; // r9d
  __int64 v12; // rdi
  unsigned __int32 v13; // edx
  __int64 v14; // rdi
  unsigned __int32 v15; // edx
  __int64 v16; // rdi
  unsigned __int32 v17; // edx
  __int64 v18; // rdi
  unsigned __int32 v19; // edx
  char v21[52]; // [rsp+1Ch] [rbp-34h] BYREF

  (*(void (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 240) + 64LL))(*(_QWORD *)(a1 + 240));
  v9 = *(_QWORD *)(a1 + 240);
  *(_DWORD *)v21 = (unsigned int)(*(_DWORD *)(a1 + 248) - 1) < 2 ? 2 : 0x2000000;
  sub_16E7EE0(v9, v21, 4u);
  v10 = *(_QWORD *)(a1 + 240);
  *(_DWORD *)v21 = (unsigned int)(*(_DWORD *)(a1 + 248) - 1) < 2 ? 24 : 402653184;
  sub_16E7EE0(v10, v21, 4u);
  v11 = a2;
  v12 = *(_QWORD *)(a1 + 240);
  v13 = _byteswap_ulong(a2);
  if ( (unsigned int)(*(_DWORD *)(a1 + 248) - 1) > 1 )
    v11 = v13;
  *(_DWORD *)v21 = v11;
  sub_16E7EE0(v12, v21, 4u);
  v14 = *(_QWORD *)(a1 + 240);
  v15 = _byteswap_ulong(a3);
  if ( (unsigned int)(*(_DWORD *)(a1 + 248) - 1) > 1 )
    a3 = v15;
  *(_DWORD *)v21 = a3;
  sub_16E7EE0(v14, v21, 4u);
  v16 = *(_QWORD *)(a1 + 240);
  v17 = _byteswap_ulong(a4);
  if ( (unsigned int)(*(_DWORD *)(a1 + 248) - 1) > 1 )
    a4 = v17;
  *(_DWORD *)v21 = a4;
  sub_16E7EE0(v16, v21, 4u);
  v18 = *(_QWORD *)(a1 + 240);
  v19 = _byteswap_ulong(a5);
  if ( (unsigned int)(*(_DWORD *)(a1 + 248) - 1) > 1 )
    a5 = v19;
  *(_DWORD *)v21 = a5;
  return sub_16E7EE0(v18, v21, 4u);
}
