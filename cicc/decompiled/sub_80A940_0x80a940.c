// Function: sub_80A940
// Address: 0x80a940
//
__int64 __fastcall sub_80A940(const char *a1, __int64 a2)
{
  unsigned int v2; // r8d

  v2 = 0;
  if ( !a2 || *(_BYTE *)(a2 + 28) != 3 )
    return v2;
  if ( (*(_BYTE *)(*(_QWORD *)(a2 + 32) + 124LL) & 0x10) != 0 && a1 )
    return strcmp(a1, "basic_string") == 0;
  return 0;
}
