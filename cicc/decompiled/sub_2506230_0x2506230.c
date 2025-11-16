// Function: sub_2506230
// Address: 0x2506230
//
__int64 __fastcall sub_2506230(__int64 a1, int *a2, __int64 a3)
{
  int v4; // esi
  _QWORD v6[3]; // [rsp+8h] [rbp-18h] BYREF

  v4 = *a2;
  v6[0] = a3;
  if ( (unsigned __int8)sub_A73170(v6, v4) )
  {
    **(_BYTE **)a1 |= **(_DWORD **)(a1 + 8) != *a2;
    **(_BYTE **)(a1 + 16) = 1;
  }
  return 0;
}
