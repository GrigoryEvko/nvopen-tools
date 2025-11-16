// Function: sub_1D172A0
// Address: 0x1d172a0
//
__int64 __fastcall sub_1D172A0(__int64 a1, __int64 a2)
{
  __int64 v2; // rcx
  __int64 result; // rax

  v2 = *(_QWORD *)(a1 + 192);
  *(_QWORD *)(a2 + 16) = a1 + 192;
  v2 &= 0xFFFFFFFFFFFFFFF8LL;
  *(_QWORD *)(a2 + 8) = v2 | *(_QWORD *)(a2 + 8) & 7LL;
  *(_QWORD *)(v2 + 8) = a2 + 8;
  result = *(_QWORD *)(a1 + 192) & 7LL;
  *(_QWORD *)(a1 + 192) = result | (a2 + 8);
  return result;
}
