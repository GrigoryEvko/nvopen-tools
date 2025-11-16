// Function: sub_C6EC80
// Address: 0xc6ec80
//
__int64 __fastcall sub_C6EC80(__int64 a1, unsigned int a2)
{
  unsigned int v2; // r12d
  unsigned int v4; // ebx

  v2 = a2 - 1;
  if ( a2 && (v2 & a2) == 0 )
  {
    _BitScanReverse(&a2, a2);
    return (unsigned int)sub_C44320((unsigned __int64 *)a1, 31 - (a2 ^ 0x1F), 0);
  }
  v4 = *(_DWORD *)(a1 + 8);
  if ( v4 <= 0x40 )
  {
    if ( (unsigned __int64)v2 >= *(_QWORD *)a1 )
      return (unsigned int)*(_QWORD *)a1;
    return v2;
  }
  else
  {
    if ( v4 - (unsigned int)sub_C444A0(a1) > 0x40 )
      return v2;
    if ( (unsigned __int64)v2 >= **(_QWORD **)a1 )
      return (unsigned int)**(_QWORD **)a1;
    return v2;
  }
}
