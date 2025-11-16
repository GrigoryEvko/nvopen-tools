// Function: sub_615C50
// Address: 0x615c50
//
__int64 __fastcall sub_615C50(__int64 a1, __int64 *a2, _QWORD **a3, int a4)
{
  __int64 result; // rax

  result = sub_822B10(24);
  *(_QWORD *)result = 0;
  *(_QWORD *)(result + 8) = a1;
  *(_DWORD *)(result + 16) = a4;
  if ( *a3 )
    **a3 = result;
  *a3 = (_QWORD *)result;
  if ( !*a2 )
    *a2 = result;
  return result;
}
