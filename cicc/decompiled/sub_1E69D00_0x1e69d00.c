// Function: sub_1E69D00
// Address: 0x1e69d00
//
__int64 __fastcall sub_1E69D00(__int64 a1, int a2)
{
  __int64 result; // rax

  if ( a2 < 0 )
    result = *(_QWORD *)(*(_QWORD *)(a1 + 24) + 16LL * (a2 & 0x7FFFFFFF) + 8);
  else
    result = *(_QWORD *)(*(_QWORD *)(a1 + 272) + 8LL * (unsigned int)a2);
  if ( result )
  {
    if ( (*(_BYTE *)(result + 3) & 0x10) != 0 )
      return *(_QWORD *)(result + 16);
    result = *(_QWORD *)(result + 32);
    if ( result )
    {
      if ( (*(_BYTE *)(result + 3) & 0x10) == 0 )
        return 0;
      return *(_QWORD *)(result + 16);
    }
  }
  return result;
}
