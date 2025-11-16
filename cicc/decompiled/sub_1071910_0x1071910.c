// Function: sub_1071910
// Address: 0x1071910
//
__int64 __fastcall sub_1071910(__int64 a1, unsigned int a2, unsigned int a3, unsigned int a4, unsigned int a5)
{
  int v9; // eax
  __int64 v10; // rdi
  int v11; // eax
  __int64 v12; // rdi
  __int64 v13; // rdi
  unsigned __int32 v14; // r9d
  __int64 v15; // rdi
  __int64 v16; // rdi
  __int64 v17; // rdi
  unsigned __int8 v19[52]; // [rsp+1Ch] [rbp-34h] BYREF

  (*(void (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 2048) + 80LL))(*(_QWORD *)(a1 + 2048));
  v9 = 2;
  v10 = *(_QWORD *)(a1 + 2048);
  if ( *(_DWORD *)(a1 + 2056) != 1 )
    v9 = 0x2000000;
  *(_DWORD *)v19 = v9;
  sub_CB6200(v10, v19, 4u);
  v11 = 24;
  v12 = *(_QWORD *)(a1 + 2048);
  if ( *(_DWORD *)(a1 + 2056) != 1 )
    v11 = 402653184;
  *(_DWORD *)v19 = v11;
  sub_CB6200(v12, v19, 4u);
  v13 = *(_QWORD *)(a1 + 2048);
  v14 = a2;
  if ( *(_DWORD *)(a1 + 2056) != 1 )
    v14 = _byteswap_ulong(a2);
  *(_DWORD *)v19 = v14;
  sub_CB6200(v13, v19, 4u);
  v15 = *(_QWORD *)(a1 + 2048);
  if ( *(_DWORD *)(a1 + 2056) != 1 )
    a3 = _byteswap_ulong(a3);
  *(_DWORD *)v19 = a3;
  sub_CB6200(v15, v19, 4u);
  v16 = *(_QWORD *)(a1 + 2048);
  if ( *(_DWORD *)(a1 + 2056) != 1 )
    a4 = _byteswap_ulong(a4);
  *(_DWORD *)v19 = a4;
  sub_CB6200(v16, v19, 4u);
  v17 = *(_QWORD *)(a1 + 2048);
  if ( *(_DWORD *)(a1 + 2056) != 1 )
    a5 = _byteswap_ulong(a5);
  *(_DWORD *)v19 = a5;
  return sub_CB6200(v17, v19, 4u);
}
