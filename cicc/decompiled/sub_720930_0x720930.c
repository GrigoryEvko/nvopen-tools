// Function: sub_720930
// Address: 0x720930
//
__int64 __fastcall sub_720930(__int64 a1, int a2, __int64 *a3, __int64 a4)
{
  __int64 result; // rax

  result = qword_4F07940;
  if ( qword_4F07940 )
    qword_4F07940 = *(_QWORD *)(qword_4F07940 + 16);
  else
    result = sub_822B10(24);
  *(_QWORD *)(result + 16) = 0;
  *(_QWORD *)result = a1;
  *(_DWORD *)(result + 8) = a2;
  if ( *a3 )
    *(_QWORD *)(*(_QWORD *)a4 + 16LL) = result;
  else
    *a3 = result;
  *(_QWORD *)a4 = result;
  return result;
}
