// Function: sub_3961900
// Address: 0x3961900
//
__int64 __fastcall sub_3961900(int *a1, __int64 a2)
{
  unsigned int v2; // r8d

  v2 = 0;
  if ( *a1 >= 0 && a1[1] >= 0 && *(_QWORD *)a1 && a1[2] <= *(_DWORD *)(a2 + 8) && a1[3] <= (unsigned int)dword_5055B20 )
    LOBYTE(v2) = a1[4] <= (unsigned int)dword_5055A40;
  return v2;
}
