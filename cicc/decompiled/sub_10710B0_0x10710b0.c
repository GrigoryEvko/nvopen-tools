// Function: sub_10710B0
// Address: 0x10710b0
//
__int64 __fastcall sub_10710B0(__int64 a1, unsigned int a2, unsigned int a3, unsigned int a4, char a5)
{
  unsigned __int32 v5; // r15d
  unsigned int v8; // ebx
  __int64 v9; // rdi
  unsigned int v10; // eax
  __int64 v11; // rdi
  unsigned int v12; // eax
  __int64 v13; // rdx
  unsigned int v14; // eax
  __int64 v15; // rdi
  __int64 v16; // rdi
  __int64 v17; // rdi
  __int64 v18; // rdi
  __int64 v19; // rdi
  __int64 result; // rax
  __int64 v21; // rdi
  unsigned __int8 v23[52]; // [rsp+1Ch] [rbp-34h] BYREF

  v5 = a2;
  v8 = (a5 != 0) << 13;
  (*(void (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 2048) + 80LL))(*(_QWORD *)(a1 + 2048));
  v9 = *(_QWORD *)(a1 + 2048);
  v10 = ((*(_BYTE *)(*(_QWORD *)(a1 + 104) + 8LL) & 1) != 0) - 17958194;
  if ( *(_DWORD *)(a1 + 2056) != 1 )
    v10 = _byteswap_ulong(v10);
  *(_DWORD *)v23 = v10;
  sub_CB6200(v9, v23, 4u);
  v11 = *(_QWORD *)(a1 + 2048);
  v12 = *(_DWORD *)(*(_QWORD *)(a1 + 104) + 12LL);
  if ( *(_DWORD *)(a1 + 2056) != 1 )
    v12 = _byteswap_ulong(v12);
  *(_DWORD *)v23 = v12;
  sub_CB6200(v11, v23, 4u);
  v13 = *(_QWORD *)(a1 + 104);
  v14 = *(_DWORD *)(v13 + 16);
  if ( *(_DWORD *)(v13 + 12) == 16777228 && v14 == 2 )
    v14 = -2147483646;
  v15 = *(_QWORD *)(a1 + 2048);
  if ( *(_DWORD *)(a1 + 2056) != 1 )
    v14 = _byteswap_ulong(v14);
  *(_DWORD *)v23 = v14;
  sub_CB6200(v15, v23, 4u);
  v16 = *(_QWORD *)(a1 + 2048);
  if ( *(_DWORD *)(a1 + 2056) != 1 )
    v5 = _byteswap_ulong(a2);
  *(_DWORD *)v23 = v5;
  sub_CB6200(v16, v23, 4u);
  v17 = *(_QWORD *)(a1 + 2048);
  if ( *(_DWORD *)(a1 + 2056) != 1 )
    a3 = _byteswap_ulong(a3);
  *(_DWORD *)v23 = a3;
  sub_CB6200(v17, v23, 4u);
  v18 = *(_QWORD *)(a1 + 2048);
  if ( *(_DWORD *)(a1 + 2056) != 1 )
    a4 = _byteswap_ulong(a4);
  *(_DWORD *)v23 = a4;
  sub_CB6200(v18, v23, 4u);
  v19 = *(_QWORD *)(a1 + 2048);
  if ( *(_DWORD *)(a1 + 2056) != 1 )
    v8 = _byteswap_ulong(v8);
  *(_DWORD *)v23 = v8;
  sub_CB6200(v19, v23, 4u);
  result = *(_QWORD *)(a1 + 104);
  if ( (*(_BYTE *)(result + 8) & 1) != 0 )
  {
    v21 = *(_QWORD *)(a1 + 2048);
    *(_DWORD *)v23 = 0;
    return sub_CB6200(v21, v23, 4u);
  }
  return result;
}
