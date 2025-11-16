// Function: sub_B326E0
// Address: 0xb326e0
//
bool __fastcall sub_B326E0(_BYTE *a1, __int64 a2, __int64 a3)
{
  char v3; // al
  bool result; // al
  __int64 v5; // rdx

  if ( (a1[34] & 1) != 0 && (*(_BYTE *)sub_B31490((__int64)a1, a2, a3) & 4) != 0 )
    return 0;
  v3 = a1[32];
  if ( (v3 & 0x30) != 0 || (v3 & 0xF) != 0 || sub_B2FC80((__int64)a1) || *a1 == 2 )
    return 0;
  v5 = sub_B326A0((__int64)a1);
  result = 1;
  if ( v5 )
    return *(_DWORD *)(v5 + 8) == 3;
  return result;
}
