// Function: sub_6E2140
// Address: 0x6e2140
//
_DWORD *__fastcall sub_6E2140(unsigned __int8 a1, __int64 a2, int a3, int a4, __int64 a5)
{
  _DWORD *result; // rax

  result = sub_6E1E00(a1, a2, a3, a4);
  *(_QWORD *)(a2 + 144) = a5;
  return result;
}
