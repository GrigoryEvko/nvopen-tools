// Function: sub_32498C0
// Address: 0x32498c0
//
__int64 __fastcall sub_32498C0(__int64 *a1, __int64 a2, __int16 a3, __int64 a4)
{
  int v4; // eax
  __int16 v6; // cx

  v4 = *(_DWORD *)(a4 + 8);
  v6 = 10;
  if ( (v4 & 0xFFFFFF00) != 0 )
  {
    LOWORD(v4) = 0;
    v6 = 3 - ((v4 == 0) - 1);
  }
  return sub_3249790(a1, a2, a3, v6, (__int64 **)a4);
}
