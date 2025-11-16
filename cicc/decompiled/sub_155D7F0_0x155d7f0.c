// Function: sub_155D7F0
// Address: 0x155d7f0
//
bool __fastcall sub_155D7F0(__int64 a1, const void *a2, size_t a3)
{
  bool result; // al
  const void *v5; // rdi
  __int64 v6; // rdx

  if ( *(_BYTE *)(a1 + 16) != 2 )
    return 0;
  v5 = (const void *)sub_155D7C0(a1);
  result = 0;
  if ( a3 == v6 )
  {
    result = 1;
    if ( a3 )
      return memcmp(v5, a2, a3) == 0;
  }
  return result;
}
