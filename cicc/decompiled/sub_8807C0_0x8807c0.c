// Function: sub_8807C0
// Address: 0x8807c0
//
__int64 __fastcall sub_8807C0(__int64 a1)
{
  __int64 result; // rax
  bool v2; // zf

  result = *(_QWORD *)(a1 + 64);
  if ( (*(_BYTE *)(a1 + 81) & 0x10) != 0 )
  {
    while ( 1 )
    {
      v2 = (*(_BYTE *)(result + 89) & 4) == 0;
      result = *(_QWORD *)(result + 40);
      if ( v2 )
        break;
      result = *(_QWORD *)(result + 32);
    }
    if ( result )
    {
      if ( *(_BYTE *)(result + 28) == 3 )
        return *(_QWORD *)(result + 32);
      else
        return 0;
    }
  }
  return result;
}
