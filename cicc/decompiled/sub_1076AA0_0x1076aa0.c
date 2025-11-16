// Function: sub_1076AA0
// Address: 0x1076aa0
//
__int64 __fastcall sub_1076AA0(__int64 a1)
{
  int v1; // eax
  bool v3; // zf
  __int64 v4; // rdi
  __int64 v5; // rdi
  unsigned int v6; // eax
  __int64 v7; // rdi
  int v8; // eax
  unsigned int v9; // eax
  __int64 v10; // rdi
  __int64 v11; // rdi
  unsigned __int8 v13[20]; // [rsp+Ch] [rbp-14h] BYREF

  v1 = 119734787;
  v3 = *(_DWORD *)(a1 + 112) == 1;
  v4 = *(_QWORD *)(a1 + 104);
  if ( !v3 )
    v1 = 50471687;
  *(_DWORD *)v13 = v1;
  sub_CB6200(v4, v13, 4u);
  v5 = *(_QWORD *)(a1 + 104);
  v6 = (*(_DWORD *)(a1 + 132) << 8) | (*(_DWORD *)(a1 + 128) << 16);
  if ( *(_DWORD *)(a1 + 112) != 1 )
    v6 = _byteswap_ulong(v6);
  *(_DWORD *)v13 = v6;
  sub_CB6200(v5, v13, 4u);
  v7 = *(_QWORD *)(a1 + 104);
  v8 = 2818068;
  if ( *(_DWORD *)(a1 + 112) != 1 )
    v8 = 335555328;
  *(_DWORD *)v13 = v8;
  sub_CB6200(v7, v13, 4u);
  v9 = *(_DWORD *)(a1 + 136);
  v10 = *(_QWORD *)(a1 + 104);
  if ( *(_DWORD *)(a1 + 112) != 1 )
    v9 = _byteswap_ulong(v9);
  *(_DWORD *)v13 = v9;
  sub_CB6200(v10, v13, 4u);
  v11 = *(_QWORD *)(a1 + 104);
  *(_DWORD *)v13 = 0;
  return sub_CB6200(v11, v13, 4u);
}
