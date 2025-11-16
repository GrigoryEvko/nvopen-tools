// Function: sub_BCB300
// Address: 0xbcb300
//
__int64 __fastcall sub_BCB300(__int64 a1, __int64 a2)
{
  unsigned int v2; // eax
  unsigned __int64 v3; // rdx

  v2 = *(_DWORD *)(a2 + 8) >> 8;
  *(_DWORD *)(a1 + 8) = v2;
  if ( v2 > 0x40 )
  {
    sub_C43690(a1, -1, 1);
    return a1;
  }
  else
  {
    v3 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v2;
    if ( !v2 )
      v3 = 0;
    *(_QWORD *)a1 = v3;
    return a1;
  }
}
