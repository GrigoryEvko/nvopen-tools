// Function: sub_7E2130
// Address: 0x7e2130
//
__int64 __fastcall sub_7E2130(__int64 a1)
{
  __int64 result; // rax

  result = qword_4F18A18;
  if ( qword_4F18A18 )
    qword_4F18A18 = *(_QWORD *)qword_4F18A18;
  else
    result = sub_823970(24);
  *(_QWORD *)(result + 8) = a1;
  *(_BYTE *)(result + 16) = 1;
  *(_QWORD *)result = qword_4D03F68[9];
  qword_4D03F68[9] = result;
  return result;
}
