// Function: sub_16A5260
// Address: 0x16a5260
//
__int64 __fastcall sub_16A5260(_QWORD *a1, unsigned int a2, unsigned int a3)
{
  char v3; // cl
  char v4; // al
  unsigned int v5; // esi
  unsigned int v6; // edx
  __int64 v7; // r9
  int v8; // eax
  unsigned __int64 v9; // r8
  __int64 result; // rax
  unsigned __int64 v11; // rdx

  v3 = a2;
  v4 = a3;
  v5 = a2 >> 6;
  v6 = a3 >> 6;
  v7 = -1LL << v3;
  v8 = v4 & 0x3F;
  if ( v8 )
  {
    v9 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v8);
    if ( v5 == v6 )
      v7 &= v9;
    else
      *(_QWORD *)(*a1 + 8LL * v6) |= v9;
  }
  *(_QWORD *)(*a1 + 8LL * v5) |= v7;
  result = v5 + 1;
  if ( v6 > (unsigned int)result )
  {
    result *= 8;
    v11 = 8 * (v5 + 2 + (unsigned __int64)(v6 - 2 - v5));
    do
    {
      *(_QWORD *)(*a1 + result) = -1;
      result += 8;
    }
    while ( v11 != result );
  }
  return result;
}
