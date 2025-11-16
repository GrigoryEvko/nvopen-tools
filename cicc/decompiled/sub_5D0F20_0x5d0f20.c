// Function: sub_5D0F20
// Address: 0x5d0f20
//
_BOOL8 __fastcall sub_5D0F20(__int64 a1, const char *a2)
{
  char v2; // al

  v2 = *(_BYTE *)(a1 + 80);
  if ( v2 == 7 )
    return strcmp(*(const char **)(*(_QWORD *)(a1 + 88) + 144LL), a2) == 0;
  if ( v2 != 11 )
    sub_721090(a1);
  return strcmp(*(const char **)(*(_QWORD *)(*(_QWORD *)(a1 + 88) + 256LL) + 40LL), a2) == 0;
}
