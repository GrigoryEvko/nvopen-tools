// Function: sub_15E64D0
// Address: 0x15e64d0
//
const char *__fastcall sub_15E64D0(__int64 a1)
{
  char v1; // al
  const char *result; // rax

  if ( *(_BYTE *)(a1 + 16) == 1 )
  {
    a1 = sub_164A820(*(_QWORD *)(a1 - 24));
    v1 = *(_BYTE *)(a1 + 16);
    if ( v1 != 3 && v1 )
      return byte_3F871B3;
    if ( (*(_BYTE *)(a1 + 34) & 0x20) == 0 )
      return 0;
    return (const char *)sub_15E61A0(a1);
  }
  result = 0;
  if ( (*(_BYTE *)(a1 + 34) & 0x20) != 0 )
    return (const char *)sub_15E61A0(a1);
  return result;
}
