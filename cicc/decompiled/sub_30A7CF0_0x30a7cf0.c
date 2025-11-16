// Function: sub_30A7CF0
// Address: 0x30a7cf0
//
__int64 __fastcall sub_30A7CF0(__int64 a1, __int64 a2)
{
  __int64 result; // rax

  result = (unsigned int)dword_502E248;
  *(_QWORD *)a1 = a2;
  *(_DWORD *)(a1 + 8) = result;
  return result;
}
