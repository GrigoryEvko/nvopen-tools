// Function: sub_2E4F5F0
// Address: 0x2e4f5f0
//
__int64 __fastcall sub_2E4F5F0(__int64 a1, __int64 a2)
{
  int v2; // eax

  v2 = *(_DWORD *)(a2 + 44);
  if ( (v2 & 4) != 0 || (v2 & 8) == 0 )
    return (*(_QWORD *)(*(_QWORD *)(a2 + 16) + 24LL) >> 30) & 1LL;
  else
    return sub_2E88A90(a2, 0x40000000, 2);
}
