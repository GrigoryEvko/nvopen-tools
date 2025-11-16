// Function: sub_15FBDD0
// Address: 0x15fbdd0
//
bool __fastcall sub_15FBDD0(__int64 a1, __int64 a2, __int64 a3)
{
  char v5; // al
  char v6; // dl
  int v7; // r13d
  unsigned int v9; // r12d
  _DWORD *v10; // rdi
  __int64 v11; // rdx
  unsigned int v12; // eax
  int v13[9]; // [rsp+1Ch] [rbp-24h] BYREF

  v5 = *(_BYTE *)(a1 + 8);
  v6 = *(_BYTE *)(a2 + 8);
  if ( v5 == 15 )
  {
    if ( v6 == 11 )
    {
      v7 = *(_DWORD *)(a2 + 8) >> 8;
      if ( (unsigned int)sub_15A9570(a3, a1) != v7 )
        return 0;
      v10 = *(_DWORD **)(a3 + 408);
      v11 = *(unsigned int *)(a3 + 416);
      v12 = *(_DWORD *)(a1 + 8);
      goto LABEL_11;
    }
    return sub_15FBC60(a1, a2);
  }
  if ( v5 != 11 || v6 != 15 )
    return sub_15FBC60(a1, a2);
  v9 = *(_DWORD *)(a1 + 8);
  if ( (unsigned int)sub_15A9570(a3, a2) != v9 >> 8 )
    return 0;
  v10 = *(_DWORD **)(a3 + 408);
  v11 = *(unsigned int *)(a3 + 416);
  v12 = *(_DWORD *)(a2 + 8);
LABEL_11:
  v13[0] = v12 >> 8;
  return &v10[v11] == sub_15F4BA0(v10, (__int64)&v10[v11], v13);
}
