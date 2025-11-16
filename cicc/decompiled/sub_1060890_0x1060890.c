// Function: sub_1060890
// Address: 0x1060890
//
__int16 __fastcall sub_1060890(__int64 a1, __int64 a2)
{
  __int16 result; // ax

  LOBYTE(result) = *(_DWORD *)(a2 + 8) >> 8 == 1;
  HIBYTE(result) = 1;
  return result;
}
