// Function: sub_1ECC890
// Address: 0x1ecc890
//
_QWORD *__fastcall sub_1ECC890(_QWORD *a1, unsigned __int64 a2)
{
  __int64 v3; // rdi
  void *v4; // rcx
  size_t v5; // rdx

  v3 = -1;
  if ( a2 <= 0x1FFFFFFFFFFFFFFELL )
    v3 = 4 * a2;
  v4 = (void *)sub_2207820(v3);
  if ( v4 && (__int64)(a2 - 1) >= 0 )
  {
    v5 = 4;
    if ( (__int64)(a2 - 2) >= -1 )
      v5 = 4 * a2;
    v4 = memset(v4, 0, v5);
  }
  *a1 = v4;
  return a1;
}
