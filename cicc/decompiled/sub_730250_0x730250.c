// Function: sub_730250
// Address: 0x730250
//
__int64 __fastcall sub_730250(__int64 a1)
{
  char v1; // dl
  __int64 result; // rax

  v1 = *(_BYTE *)(a1 + 48);
  if ( v1 == 2 )
    return *(_QWORD *)(a1 + 56);
  result = 0;
  if ( v1 == 3 )
  {
    result = *(_QWORD *)(a1 + 56);
    if ( result )
    {
      if ( *(_BYTE *)(result + 24) == 2 )
        return *(_QWORD *)(result + 56);
      else
        return 0;
    }
  }
  return result;
}
