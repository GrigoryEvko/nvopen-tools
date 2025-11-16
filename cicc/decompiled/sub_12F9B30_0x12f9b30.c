// Function: sub_12F9B30
// Address: 0x12f9b30
//
char __fastcall sub_12F9B30(unsigned __int64 *a1, unsigned __int64 *a2)
{
  __int64 v2; // rdx
  unsigned __int64 v3; // rax
  __int64 v4; // rsi
  __int64 v5; // rcx
  unsigned __int64 v6; // rdi
  unsigned __int64 v7; // r8
  __int64 v8; // rdx
  bool v10; // al

  v2 = a2[1];
  v3 = *a2;
  v4 = a1[1];
  v5 = v2;
  v6 = *a1;
  v7 = v3;
  v8 = ~v4;
  if ( (~v4 & 0x7FFF000000000000LL) == 0 && v6 | v4 & 0xFFFFFFFFFFFFLL )
    return sub_12FB5C8(v6, v4, v8, v5, v3);
  v8 = 0x7FFF000000000000LL;
  if ( (~v5 & 0x7FFF000000000000LL) == 0 )
  {
    if ( v3 | v5 & 0xFFFFFFFFFFFFLL )
      return sub_12FB5C8(v6, v4, v8, v5, v3);
  }
  if ( v4 < 0 != v5 < 0 )
  {
    if ( v4 >= 0 )
      return (v6 | v3 | (v4 | v5) & 0x7FFFFFFFFFFFFFFFLL) == 0;
    return nullsub_2015(v6, v4, (unsigned __int64)v4 >> 63, v5, v3);
  }
  if ( v4 == v5 && v3 == v6 )
    return nullsub_2015(v6, v4, (unsigned __int64)v4 >> 63, v5, v3);
  v10 = 1;
  if ( v4 >= (unsigned __int64)v5 )
    v10 = v4 == v5 && v7 > v6;
  return (v4 < 0) ^ v10;
}
