// Function: sub_10723C0
// Address: 0x10723c0
//
__int64 __fastcall sub_10723C0(__int64 a1, unsigned int a2, unsigned int a3, unsigned int a4)
{
  unsigned __int32 v4; // r14d
  __int64 v8; // rdi
  int v9; // eax
  __int64 v10; // rdi
  __int64 v11; // rdi
  __int64 v12; // rdi
  unsigned __int8 v14[36]; // [rsp+Ch] [rbp-24h] BYREF

  v4 = a2;
  (*(void (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 2048) + 80LL))(*(_QWORD *)(a1 + 2048));
  v8 = *(_QWORD *)(a1 + 2048);
  if ( *(_DWORD *)(a1 + 2056) != 1 )
    v4 = _byteswap_ulong(a2);
  *(_DWORD *)v14 = v4;
  sub_CB6200(v8, v14, 4u);
  v9 = 16;
  v10 = *(_QWORD *)(a1 + 2048);
  if ( *(_DWORD *)(a1 + 2056) != 1 )
    v9 = 0x10000000;
  *(_DWORD *)v14 = v9;
  sub_CB6200(v10, v14, 4u);
  v11 = *(_QWORD *)(a1 + 2048);
  if ( *(_DWORD *)(a1 + 2056) != 1 )
    a3 = _byteswap_ulong(a3);
  *(_DWORD *)v14 = a3;
  sub_CB6200(v11, v14, 4u);
  v12 = *(_QWORD *)(a1 + 2048);
  if ( *(_DWORD *)(a1 + 2056) != 1 )
    a4 = _byteswap_ulong(a4);
  *(_DWORD *)v14 = a4;
  return sub_CB6200(v12, v14, 4u);
}
