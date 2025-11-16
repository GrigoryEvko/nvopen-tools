// Function: sub_AF4850
// Address: 0xaf4850
//
__int64 __fastcall sub_AF4850(__int64 a1, __int64 a2)
{
  __int64 v3; // rdi
  __int64 v4; // rdx
  __int64 v5; // rax
  __int64 result; // rax

  v3 = a1 + 8;
  v4 = *(_DWORD *)(a2 + 4) & 0x7FFFFFF;
  *(_QWORD *)(v3 - 8) = *(_QWORD *)(*(_QWORD *)(a2 + 32 * (1 - v4)) + 24LL);
  v5 = *(_QWORD *)(*(_QWORD *)(a2 + 32 * (2 - v4)) + 24LL);
  sub_AF47B0(v3, *(unsigned __int64 **)(v5 + 16), *(unsigned __int64 **)(v5 + 24));
  result = sub_B10D40(a2 + 48);
  *(_QWORD *)(a1 + 32) = result;
  return result;
}
