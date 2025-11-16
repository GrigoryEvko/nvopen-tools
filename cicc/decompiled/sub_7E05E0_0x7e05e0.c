// Function: sub_7E05E0
// Address: 0x7e05e0
//
__int64 __fastcall sub_7E05E0(__int64 a1, __int64 a2)
{
  __int64 result; // rax
  __int64 i; // rdx

  for ( result = a1; result; result = *(_QWORD *)(result + 112) )
  {
    if ( *(_QWORD *)(result + 128) != a2 || *(_BYTE *)(result + 136) )
      continue;
    if ( (*(_BYTE *)(result + 144) & 4) != 0 )
    {
      if ( !*(_BYTE *)(result + 137) )
        continue;
    }
    else
    {
      for ( i = *(_QWORD *)(result + 120); *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
        ;
      if ( !*(_QWORD *)(i + 128) )
        continue;
    }
    if ( (*(_BYTE *)(result + 146) & 8) == 0 )
      return result;
  }
  return result;
}
