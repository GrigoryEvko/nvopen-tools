// Function: sub_39157D0
// Address: 0x39157d0
//
__int64 __fastcall sub_39157D0(__int64 a1, unsigned int a2, unsigned int a3, unsigned int a4)
{
  unsigned int v4; // r14d
  __int64 v8; // rdi
  unsigned __int32 v9; // edx
  __int64 v10; // rdi
  __int64 v11; // rdi
  unsigned __int32 v12; // edx
  __int64 v13; // rdi
  unsigned __int32 v14; // edx
  char v16[36]; // [rsp+Ch] [rbp-24h] BYREF

  v4 = a2;
  (*(void (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 240) + 64LL))(*(_QWORD *)(a1 + 240));
  v8 = *(_QWORD *)(a1 + 240);
  v9 = _byteswap_ulong(a2);
  if ( (unsigned int)(*(_DWORD *)(a1 + 248) - 1) > 1 )
    v4 = v9;
  *(_DWORD *)v16 = v4;
  sub_16E7EE0(v8, v16, 4u);
  v10 = *(_QWORD *)(a1 + 240);
  *(_DWORD *)v16 = (unsigned int)(*(_DWORD *)(a1 + 248) - 1) < 2 ? 16 : 0x10000000;
  sub_16E7EE0(v10, v16, 4u);
  v11 = *(_QWORD *)(a1 + 240);
  v12 = _byteswap_ulong(a3);
  if ( (unsigned int)(*(_DWORD *)(a1 + 248) - 1) > 1 )
    a3 = v12;
  *(_DWORD *)v16 = a3;
  sub_16E7EE0(v11, v16, 4u);
  v13 = *(_QWORD *)(a1 + 240);
  v14 = _byteswap_ulong(a4);
  if ( (unsigned int)(*(_DWORD *)(a1 + 248) - 1) > 1 )
    a4 = v14;
  *(_DWORD *)v16 = a4;
  return sub_16E7EE0(v13, v16, 4u);
}
