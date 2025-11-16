// Function: sub_385F870
// Address: 0x385f870
//
bool __fastcall sub_385F870(__int64 a1, unsigned __int64 a2, __int64 a3)
{
  unsigned __int64 v4; // r8
  unsigned __int64 v5; // rbx
  unsigned __int64 v6; // rdi
  unsigned __int64 v7; // rcx
  bool result; // al

  v4 = 2 * a3;
  v6 = *(_QWORD *)(a1 + 200);
  v5 = v6;
  if ( a3 << 6 <= v6 )
    v6 = a3 << 6;
  if ( v4 > v6 )
    return 1;
  v7 = 2 * a3;
  while ( !(a2 % v7) || a2 / v7 >= 8 * a3 )
  {
    v7 *= 2LL;
    if ( v7 > v6 )
      goto LABEL_8;
  }
  result = 1;
  v6 = v7 >> 1;
  if ( v4 > v7 >> 1 )
    return result;
LABEL_8:
  result = a3 << 6 != v6 && v5 > v6;
  if ( result )
  {
    *(_QWORD *)(a1 + 200) = v6;
    return 0;
  }
  return result;
}
