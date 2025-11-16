// Function: sub_2254970
// Address: 0x2254970
//
__int64 __fastcall sub_2254970(__int64 a1, __int64 a2)
{
  __int64 result; // rax

  *(_DWORD *)(a1 + 8) = a2 != 0;
  *(_QWORD *)a1 = off_4A07F18;
  result = sub_2208E60(a1, a2);
  *(_QWORD *)(a1 + 16) = result;
  return result;
}
