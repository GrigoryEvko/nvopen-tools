// Function: sub_2EAFBF0
// Address: 0x2eafbf0
//
__int64 *__fastcall sub_2EAFBF0(__int64 a1, __int64 a2)
{
  __int64 *v2; // rdi
  __int64 v4; // [rsp-8h] [rbp-8h]

  v2 = *(__int64 **)(a1 + 8);
  if ( v2 )
    return sub_2E39F50(v2, a2);
  *((_BYTE *)&v4 - 8) = 0;
  return (__int64 *)*(&v4 - 2);
}
