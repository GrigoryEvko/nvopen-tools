// Function: sub_CEFF50
// Address: 0xceff50
//
bool __fastcall sub_CEFF50(__int64 a1)
{
  __int64 v2; // rdi
  bool result; // al
  size_t v4; // rdx
  char *v5; // r13

  if ( *(_BYTE *)a1 <= 0x1Cu )
    return 0;
  v2 = *(_QWORD *)(*(_QWORD *)(a1 + 40) + 72LL);
  if ( !v2 || !(unsigned __int8)sub_CEF9A0(v2) )
    return 0;
  v4 = 0;
  v5 = off_4C5D0D8[0];
  if ( off_4C5D0D8[0] )
    v4 = strlen(off_4C5D0D8[0]);
  if ( *(_QWORD *)(a1 + 48) )
    return sub_B91F50(a1, v5, v4) != 0;
  result = 0;
  if ( (*(_BYTE *)(a1 + 7) & 0x20) != 0 )
    return sub_B91F50(a1, v5, v4) != 0;
  return result;
}
