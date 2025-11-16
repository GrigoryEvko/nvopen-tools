// Function: sub_1FE6580
// Address: 0x1fe6580
//
__int64 __fastcall sub_1FE6580(__int64 a1)
{
  unsigned int i; // r8d
  char v2; // al

  for ( i = *(_DWORD *)(a1 + 60); i; --i )
  {
    v2 = *(_BYTE *)(*(_QWORD *)(a1 + 40) + 16LL * (i - 1));
    if ( v2 != 111 )
    {
      if ( v2 == 1 )
        --i;
      return i;
    }
  }
  return i;
}
