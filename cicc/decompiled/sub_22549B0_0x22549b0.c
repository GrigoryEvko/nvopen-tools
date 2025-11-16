// Function: sub_22549B0
// Address: 0x22549b0
//
__int64 __fastcall sub_22549B0(__int64 a1, __int64 a2)
{
  __int64 result; // rax

  *(_DWORD *)(a1 + 8) = a2 != 0;
  *(_QWORD *)a1 = off_4A07F70;
  result = sub_2208E60(a1, a2);
  *(_QWORD *)(a1 + 16) = result;
  return result;
}
