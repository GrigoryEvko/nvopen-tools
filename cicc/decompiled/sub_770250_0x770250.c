// Function: sub_770250
// Address: 0x770250
//
__int64 __fastcall sub_770250(_QWORD *a1)
{
  _QWORD *v1; // rax
  __int64 v2; // rax
  __int64 result; // rax

  v1 = (_QWORD *)qword_4F082A0;
  if ( qword_4F082A0 )
    qword_4F082A0 = *(_QWORD *)(qword_4F082A0 + 8);
  else
    v1 = (_QWORD *)sub_823970(0x10000);
  *v1 = a1[1];
  a1[1] = v1;
  v1[1] = 0;
  v2 = a1[1];
  a1[2] = 0;
  result = v2 + 24;
  a1[4] = 0;
  *a1 = result;
  return result;
}
