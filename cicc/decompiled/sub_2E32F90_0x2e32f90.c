// Function: sub_2E32F90
// Address: 0x2e32f90
//
_DWORD *__fastcall sub_2E32F90(__int64 a1, __int64 a2, int a3)
{
  _DWORD *result; // rax

  result = *(_DWORD **)(a1 + 152);
  if ( *(_DWORD **)(a1 + 144) != result )
  {
    result = (_DWORD *)sub_2E32F70(a1, a2);
    *result = a3;
  }
  return result;
}
