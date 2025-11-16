// Function: sub_67D020
// Address: 0x67d020
//
__int64 __fastcall sub_67D020(__int64 a1, _QWORD *a2)
{
  char v2; // r13
  char v3; // r14
  __int64 v4; // rdi
  __int64 v5; // rax
  __int64 v6; // rax
  __int64 result; // rax

  sub_67B660();
  v2 = byte_4CFFE51;
  v3 = byte_4CFFE59;
  byte_4CFFE51 = 0;
  byte_4CFFE59 = 1;
  sub_74B930(a1, &qword_4CFFDC0);
  v4 = qword_4D039E8;
  v5 = *(_QWORD *)(qword_4D039E8 + 16);
  if ( (unsigned __int64)(v5 + 1) > *(_QWORD *)(qword_4D039E8 + 8) )
  {
    sub_823810(qword_4D039E8);
    v4 = qword_4D039E8;
    v5 = *(_QWORD *)(qword_4D039E8 + 16);
  }
  *(_BYTE *)(*(_QWORD *)(v4 + 32) + v5) = 0;
  v6 = *(_QWORD *)(v4 + 16);
  *(_QWORD *)(v4 + 16) = v6 + 1;
  *a2 = v6;
  result = *(_QWORD *)(v4 + 32);
  byte_4CFFE59 = v3;
  byte_4CFFE51 = v2;
  return result;
}
