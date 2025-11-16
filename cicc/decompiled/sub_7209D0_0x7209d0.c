// Function: sub_7209D0
// Address: 0x7209d0
//
__int64 __fastcall sub_7209D0(__int64 a1, __int64 *a2, __int64 *a3)
{
  __int64 result; // rax
  __int64 v5; // rcx

  result = qword_4F07940;
  if ( qword_4F07940 )
    qword_4F07940 = *(_QWORD *)(qword_4F07940 + 16);
  else
    result = sub_822B10(24);
  *(_QWORD *)(result + 16) = 0;
  *(_DWORD *)(result + 8) = 0;
  *(_QWORD *)result = a1;
  v5 = *a2;
  *(_QWORD *)(result + 16) = *a2;
  if ( !v5 )
    *a3 = result;
  *a2 = result;
  return result;
}
