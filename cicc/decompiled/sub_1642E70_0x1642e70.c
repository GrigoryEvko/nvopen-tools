// Function: sub_1642E70
// Address: 0x1642e70
//
__int64 __fastcall sub_1642E70(__int64 a1, __int64 a2)
{
  __int64 v3; // rax
  __int64 v4; // rdx
  bool v5; // zf
  __int64 v6; // rax
  __int64 v7; // rdx
  _QWORD v8[2]; // [rsp+8h] [rbp-48h] BYREF
  __int64 v9; // [rsp+18h] [rbp-38h] BYREF
  __int64 v10; // [rsp+20h] [rbp-30h] BYREF
  unsigned int v11[10]; // [rsp+28h] [rbp-28h] BYREF

  *(_BYTE *)(a1 + 4) = 0;
  *(_BYTE *)(a1 + 16) = 0;
  v8[0] = a2;
  v9 = sub_1560340(v8, -1, "statepoint-id", 0xDu);
  if ( sub_155D3E0((__int64)&v9) )
  {
    v3 = sub_155D8B0(&v9);
    if ( !(unsigned __int8)sub_16D2B80(v3, v4, 10, v11) )
    {
      v5 = *(_BYTE *)(a1 + 16) == 0;
      *(_QWORD *)(a1 + 8) = *(_QWORD *)v11;
      if ( v5 )
        *(_BYTE *)(a1 + 16) = 1;
    }
  }
  v10 = sub_1560340(v8, -1, "statepoint-num-patch-bytes", 0x1Au);
  if ( sub_155D3E0((__int64)&v10) )
  {
    v6 = sub_155D8B0(&v10);
    if ( !(unsigned __int8)sub_16D2B80(v6, v7, 10, v11) && *(_QWORD *)v11 == v11[0] )
    {
      v5 = *(_BYTE *)(a1 + 4) == 0;
      *(_DWORD *)a1 = v11[0];
      if ( v5 )
        *(_BYTE *)(a1 + 4) = 1;
    }
  }
  return a1;
}
