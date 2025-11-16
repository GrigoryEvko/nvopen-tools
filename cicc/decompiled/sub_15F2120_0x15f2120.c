// Function: sub_15F2120
// Address: 0x15f2120
//
__int64 __fastcall sub_15F2120(__int64 a1, __int64 a2)
{
  __int64 v2; // rcx
  __int64 result; // rax

  sub_157E9D0(*(_QWORD *)(a2 + 40) + 40LL, a1);
  v2 = *(_QWORD *)(a2 + 24);
  *(_QWORD *)(a1 + 32) = a2 + 24;
  v2 &= 0xFFFFFFFFFFFFFFF8LL;
  *(_QWORD *)(a1 + 24) = v2 | *(_QWORD *)(a1 + 24) & 7LL;
  *(_QWORD *)(v2 + 8) = a1 + 24;
  result = *(_QWORD *)(a2 + 24) & 7LL;
  *(_QWORD *)(a2 + 24) = result | (a1 + 24);
  return result;
}
