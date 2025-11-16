// Function: sub_76FF70
// Address: 0x76ff70
//
__int64 __fastcall sub_76FF70(__int64 a1)
{
  __int64 result; // rax

  for ( result = a1; result; result = *(_QWORD *)(result + 112) )
  {
    if ( *(_QWORD *)result )
    {
      if ( *(_QWORD *)(result + 8) )
        return result;
    }
    else
    {
      if ( (*(_BYTE *)(result + 145) & 1) == 0 )
        continue;
      if ( *(_QWORD *)(result + 8) )
        return result;
    }
    if ( (*(_BYTE *)(result + 144) & 4) == 0 || (*(_BYTE *)(result + 144) & 0x10) != 0 )
      return result;
  }
  return result;
}
