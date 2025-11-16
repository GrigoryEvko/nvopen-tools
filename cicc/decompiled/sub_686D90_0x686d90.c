// Function: sub_686D90
// Address: 0x686d90
//
__int64 __fastcall sub_686D90(unsigned int a1, FILE *a2, __int64 a3, __int64 *a4)
{
  _DWORD *v6; // rax
  __int64 v7; // r12
  __int64 result; // rax

  v6 = sub_67D610(a1, a2, 2u);
  v7 = (__int64)v6;
  if ( a3 )
  {
    a2 = (FILE *)a3;
    sub_67D780((__int64)v6, a3);
  }
  if ( !a4 )
    return sub_6837D0(v7, a2);
  if ( !*a4 )
    *a4 = v7;
  result = a4[1];
  if ( result )
    *(_QWORD *)(result + 8) = v7;
  a4[1] = v7;
  return result;
}
