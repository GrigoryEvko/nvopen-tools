// Function: sub_C88920
// Address: 0xc88920
//
unsigned __int64 __fastcall sub_C88920(_QWORD *a1)
{
  _QWORD *v1; // rax
  unsigned __int64 *v2; // rdx
  unsigned __int64 v3; // rsi
  unsigned __int64 v4; // rcx
  unsigned __int64 result; // rax
  unsigned __int64 v6; // rdx

  v1 = a1;
  v2 = a1;
  do
  {
    v3 = v2[156] ^ ((v2[1] & 0x7FFFFFFF | *v2 & 0xFFFFFFFF80000000LL) >> 1);
    if ( (v2[1] & 1) != 0 )
      v3 ^= 0xB5026F5AA96619E9LL;
    *v2++ = v3;
  }
  while ( a1 + 156 != v2 );
  do
  {
    v4 = *v1 ^ ((v1[157] & 0x7FFFFFFFLL | v1[156] & 0xFFFFFFFF80000000LL) >> 1);
    if ( (v1[157] & 1) != 0 )
      v4 ^= 0xB5026F5AA96619E9LL;
    v1[156] = v4;
    ++v1;
  }
  while ( a1 + 155 != v1 );
  result = *a1 & 0x7FFFFFFFLL | a1[311] & 0xFFFFFFFF80000000LL;
  v6 = a1[155] ^ (result >> 1);
  if ( (*(_BYTE *)a1 & 1) != 0 )
  {
    result = 0xB5026F5AA96619E9LL;
    v6 ^= 0xB5026F5AA96619E9LL;
  }
  a1[311] = v6;
  a1[312] = 0;
  return result;
}
