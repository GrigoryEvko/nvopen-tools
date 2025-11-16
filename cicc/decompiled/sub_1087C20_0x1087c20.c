// Function: sub_1087C20
// Address: 0x1087c20
//
__int64 __fastcall sub_1087C20(__int64 a1, __int64 a2)
{
  bool v3; // zf
  unsigned int v4; // eax
  __int64 v5; // rdi
  unsigned int v6; // eax
  __int64 v7; // rdi
  __int16 v8; // ax
  __int64 v9; // rdi
  unsigned __int8 v11[36]; // [rsp+Ch] [rbp-24h] BYREF

  v3 = *(_DWORD *)(a1 + 16) == 1;
  v4 = *(_DWORD *)a2;
  v5 = *(_QWORD *)(a1 + 8);
  if ( !v3 )
    v4 = _byteswap_ulong(v4);
  *(_DWORD *)v11 = v4;
  sub_CB6200(v5, v11, 4u);
  v6 = *(_DWORD *)(a2 + 4);
  v7 = *(_QWORD *)(a1 + 8);
  if ( *(_DWORD *)(a1 + 16) != 1 )
    v6 = _byteswap_ulong(v6);
  *(_DWORD *)v11 = v6;
  sub_CB6200(v7, v11, 4u);
  v8 = *(_WORD *)(a2 + 8);
  v9 = *(_QWORD *)(a1 + 8);
  if ( *(_DWORD *)(a1 + 16) != 1 )
    v8 = __ROL2__(v8, 8);
  *(_WORD *)v11 = v8;
  return sub_CB6200(v9, v11, 2u);
}
