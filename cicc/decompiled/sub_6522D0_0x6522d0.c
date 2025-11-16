// Function: sub_6522D0
// Address: 0x6522d0
//
__int64 __fastcall sub_6522D0(_QWORD *a1)
{
  __int64 result; // rax
  char v2; // dl
  __int64 v3; // rdi

  result = a1[44];
  if ( result )
  {
    if ( *(_QWORD *)result )
    {
      v2 = *(_BYTE *)(*a1 + 80LL);
      if ( v2 != 18 )
      {
        if ( v2 != 7 && v2 != 9 )
          sub_721090(a1);
        v3 = *(_QWORD *)(*a1 + 88LL);
        if ( *(_BYTE *)(result + 16) == 7 )
          *(_BYTE *)(v3 + 175) |= 8u;
        else
          *(_BYTE *)(*(_QWORD *)(result + 24) + 57LL) |= 2u;
        return sub_869D70(v3, 7);
      }
    }
  }
  return result;
}
