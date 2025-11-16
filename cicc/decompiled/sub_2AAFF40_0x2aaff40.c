// Function: sub_2AAFF40
// Address: 0x2aaff40
//
__int64 __fastcall sub_2AAFF40(__int64 a1, _QWORD *a2, unsigned __int64 *a3)
{
  __int64 v3; // rax
  unsigned __int64 v4; // rdi
  __int64 result; // rax

  a2[10] = a1;
  v3 = a2[3];
  v4 = *a3;
  a2[4] = a3;
  v4 &= 0xFFFFFFFFFFFFFFF8LL;
  a2[3] = v4 | v3 & 7;
  *(_QWORD *)(v4 + 8) = a2 + 3;
  result = *a3 & 7;
  *a3 = result | (unsigned __int64)(a2 + 3);
  return result;
}
