// Function: sub_2C19D60
// Address: 0x2c19d60
//
__int64 __fastcall sub_2C19D60(_QWORD *a1, __int64 a2)
{
  __int64 v2; // rcx
  __int64 result; // rax

  a1[10] = *(_QWORD *)(a2 + 80);
  v2 = *(_QWORD *)(a2 + 24);
  a1[4] = a2 + 24;
  v2 &= 0xFFFFFFFFFFFFFFF8LL;
  a1[3] = v2 | a1[3] & 7LL;
  *(_QWORD *)(v2 + 8) = a1 + 3;
  result = *(_QWORD *)(a2 + 24) & 7LL;
  *(_QWORD *)(a2 + 24) = result | (unsigned __int64)(a1 + 3);
  return result;
}
