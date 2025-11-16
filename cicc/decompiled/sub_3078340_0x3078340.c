// Function: sub_3078340
// Address: 0x3078340
//
signed __int64 __fastcall sub_3078340(__int64 a1, __int64 a2, unsigned int a3)
{
  int v5; // edi

  v5 = *(_DWORD *)(a2 + 16);
  if ( (unsigned int)(v5 - 9540) <= 6 && ((1LL << ((unsigned __int8)v5 - 68)) & 0x49) != 0 || sub_CEA4B0(v5) )
    return 4;
  else
    return sub_306A930(a1, a2, a3);
}
