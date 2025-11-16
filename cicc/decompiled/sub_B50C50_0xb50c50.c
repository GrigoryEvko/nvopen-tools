// Function: sub_B50C50
// Address: 0xb50c50
//
__int64 __fastcall sub_B50C50(__int64 a1, __int64 a2, __int64 a3)
{
  char v4; // al
  char v5; // dl
  int v6; // r13d
  unsigned int v8; // r12d
  unsigned int v9; // esi

  v4 = *(_BYTE *)(a1 + 8);
  v5 = *(_BYTE *)(a2 + 8);
  if ( v4 == 14 )
  {
    if ( v5 == 12 )
    {
      v6 = *(_DWORD *)(a2 + 8) >> 8;
      if ( (unsigned int)sub_AE43A0(a3, a1) != v6 )
        return 0;
      v9 = *(_DWORD *)(a1 + 8);
      return *((unsigned __int8 *)sub_AE2980(a3, v9 >> 8) + 16) ^ 1u;
    }
    return sub_B50B40(a1, a2);
  }
  if ( v4 != 12 || v5 != 14 )
    return sub_B50B40(a1, a2);
  v8 = *(_DWORD *)(a1 + 8);
  if ( (unsigned int)sub_AE43A0(a3, a2) != v8 >> 8 )
    return 0;
  v9 = *(_DWORD *)(a2 + 8);
  return *((unsigned __int8 *)sub_AE2980(a3, v9 >> 8) + 16) ^ 1u;
}
