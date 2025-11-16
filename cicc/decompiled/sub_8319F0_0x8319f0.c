// Function: sub_8319F0
// Address: 0x8319f0
//
__int64 __fastcall sub_8319F0(__int64 a1, _QWORD *a2)
{
  __int64 result; // rax
  __int64 v3; // rcx
  char v4; // dl

  result = 0;
  if ( *(_BYTE *)(a1 + 16) == 1 )
  {
    v3 = *(_QWORD *)(a1 + 144);
    v4 = *(_BYTE *)(v3 + 24);
    if ( v4 == 1 )
    {
      if ( *(_BYTE *)(v3 + 56) != 9 )
        return result;
      v3 = *(_QWORD *)(v3 + 72);
      v4 = *(_BYTE *)(v3 + 24);
    }
    result = 0;
    if ( (unsigned __int8)(v4 - 5) <= 1u )
    {
      result = 1;
      if ( a2 )
        *a2 = v3;
    }
  }
  return result;
}
