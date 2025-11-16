// Function: sub_267FF40
// Address: 0x267ff40
//
__int64 __fastcall sub_267FF40(__int64 *a1, unsigned int *a2, char a3)
{
  __int64 result; // rax

  if ( *((_QWORD *)a2 + 15) )
  {
    sub_3122A50(a1 + 50, *a2);
    if ( a3 )
      sub_BD3960(*((_QWORD *)a2 + 15));
    return sub_267FDF0(a1, (__int64)a2);
  }
  return result;
}
