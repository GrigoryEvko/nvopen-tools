// Function: sub_25CD160
// Address: 0x25cd160
//
__int64 __fastcall sub_25CD160(__int64 a1, __int64 a2, const void *a3, __int64 a4)
{
  unsigned int v4; // r8d
  size_t v6; // rdx
  unsigned int v8; // r8d

  v4 = 0;
  if ( a2 == 1 || (*(_BYTE *)(a1 + 12) & 0xFu) - 7 > 1 )
    return v4;
  v6 = *(_QWORD *)(a1 + 32);
  if ( v6 != a4 )
    return 1;
  if ( !v6 )
    return v4;
  LOBYTE(v8) = memcmp(*(const void **)(a1 + 24), a3, v6) != 0;
  return v8;
}
