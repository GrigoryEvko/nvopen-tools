// Function: sub_149A7F0
// Address: 0x149a7f0
//
_BYTE *__fastcall sub_149A7F0(_BYTE *a1, __int64 a2)
{
  size_t v3; // rdx
  _BYTE *v4; // r14
  _BYTE *v5; // rax

  if ( !a2 )
    return 0;
  v3 = 0x7FFFFFFFFFFFFFFFLL;
  if ( a2 >= 0 )
    v3 = a2;
  v4 = a1;
  v5 = memchr(a1, 0, v3);
  if ( v5 && v5 - a1 != -1 )
    return 0;
  if ( *a1 == 1 )
    return a1 + 1;
  return v4;
}
