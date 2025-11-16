// Function: sub_300C100
// Address: 0x300c100
//
__int64 __fastcall sub_300C100(__int64 a1, int a2)
{
  __int64 result; // rax

  result = sub_300BF60(
             a1,
             *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 24) + 32LL) + 56LL) + 16LL * (a2 & 0x7FFFFFFF))
           & 0xFFFFFFFFFFFFFFF8LL);
  *(_DWORD *)(*(_QWORD *)(a1 + 56) + 4LL * (a2 & 0x7FFFFFFF)) = result;
  return result;
}
