// Function: sub_100AC90
// Address: 0x100ac90
//
__int64 __fastcall sub_100AC90(_QWORD **a1, int a2, unsigned __int8 *a3)
{
  __int64 result; // rax
  __int64 v5; // rdx

  if ( a2 + 29 != *a3 )
    return 0;
  result = sub_995B10(a1, *((_QWORD *)a3 - 8));
  if ( !(_BYTE)result )
    return 0;
  v5 = *((_QWORD *)a3 - 4);
  if ( !v5 )
    return 0;
  *a1[1] = v5;
  return result;
}
