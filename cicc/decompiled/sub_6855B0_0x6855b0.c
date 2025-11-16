// Function: sub_6855B0
// Address: 0x6855b0
//
__int64 __fastcall sub_6855B0(unsigned int a1, FILE *a2, _QWORD *a3)
{
  _DWORD *v4; // rax
  _DWORD *v5; // rdi
  __int64 result; // rax

  v4 = sub_67D610(a1, a2, 2u);
  v5 = v4;
  if ( !a3 )
    return sub_6837D0((__int64)v4, a2);
  if ( !*a3 )
    *a3 = v4;
  result = a3[1];
  if ( result )
    *(_QWORD *)(result + 8) = v5;
  a3[1] = v5;
  return result;
}
