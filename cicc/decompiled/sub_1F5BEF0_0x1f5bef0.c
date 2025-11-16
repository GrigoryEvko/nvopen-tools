// Function: sub_1F5BEF0
// Address: 0x1f5bef0
//
__int64 __fastcall sub_1F5BEF0(__int64 a1, int a2)
{
  __int64 result; // rax

  result = sub_1F5BDD0(
             a1,
             *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 256) + 40LL) + 24LL) + 16LL * (a2 & 0x7FFFFFFF))
           & 0xFFFFFFFFFFFFFFF8LL);
  *(_DWORD *)(*(_QWORD *)(a1 + 288) + 4LL * (a2 & 0x7FFFFFFF)) = result;
  return result;
}
