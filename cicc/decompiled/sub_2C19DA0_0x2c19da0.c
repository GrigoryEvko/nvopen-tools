// Function: sub_2C19DA0
// Address: 0x2c19da0
//
__int64 __fastcall sub_2C19DA0(_QWORD *a1, __int64 a2, unsigned __int64 *a3)
{
  __int64 v3; // rax
  unsigned __int64 v4; // rsi
  __int64 result; // rax

  a1[10] = a2;
  v3 = a1[3];
  v4 = *a3;
  a1[4] = a3;
  v4 &= 0xFFFFFFFFFFFFFFF8LL;
  a1[3] = v4 | v3 & 7;
  *(_QWORD *)(v4 + 8) = a1 + 3;
  result = *a3 & 7;
  *a3 = result | (unsigned __int64)(a1 + 3);
  return result;
}
