// Function: sub_1643380
// Address: 0x1643380
//
__int64 __fastcall sub_1643380(__int64 a1, __int64 a2)
{
  unsigned int v2; // ecx

  v2 = *(_DWORD *)(a2 + 8) >> 8;
  *(_DWORD *)(a1 + 8) = v2;
  if ( v2 > 0x40 )
    sub_16A4EF0(a1, -1, 1);
  else
    *(_QWORD *)a1 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v2;
  return a1;
}
