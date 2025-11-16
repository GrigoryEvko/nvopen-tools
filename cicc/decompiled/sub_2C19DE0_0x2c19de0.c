// Function: sub_2C19DE0
// Address: 0x2c19de0
//
__int64 __fastcall sub_2C19DE0(_QWORD *a1, __int64 a2)
{
  unsigned __int64 *v2; // rdx
  __int64 v3; // rax
  unsigned __int64 v4; // rsi
  __int64 result; // rax

  v2 = *(unsigned __int64 **)(a2 + 32);
  a1[10] = *(_QWORD *)(a2 + 80);
  v3 = a1[3];
  v4 = *v2;
  a1[4] = v2;
  v4 &= 0xFFFFFFFFFFFFFFF8LL;
  a1[3] = v4 | v3 & 7;
  *(_QWORD *)(v4 + 8) = a1 + 3;
  result = *v2 & 7;
  *v2 = result | (unsigned __int64)(a1 + 3);
  return result;
}
