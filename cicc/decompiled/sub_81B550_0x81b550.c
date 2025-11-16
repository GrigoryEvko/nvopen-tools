// Function: sub_81B550
// Address: 0x81b550
//
__int64 __fastcall sub_81B550(unsigned __int8 *a1)
{
  int v1; // eax
  __int64 result; // rax

  v1 = *a1;
  *((_QWORD *)a1 + 1) = 0;
  *((_QWORD *)a1 + 2) = 0;
  result = v1 & 0xFFFFFFE0 | 1;
  *a1 = result;
  return result;
}
