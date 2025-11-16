// Function: sub_81B4A0
// Address: 0x81b4a0
//
__int64 __fastcall sub_81B4A0(unsigned __int64 a1, __int64 a2)
{
  __int64 v3; // r9
  unsigned __int64 v4; // rax
  unsigned __int64 v5; // rsi
  unsigned __int64 v6; // rbx
  unsigned __int64 v7; // rbx
  void *v8; // r12
  __int64 result; // rax

  v3 = a2 - (_QWORD)qword_4F19580;
  v4 = ~(a2 - (_QWORD)qword_4F19580);
  v5 = qword_4F19578 - (_QWORD)qword_4F19580;
  if ( v4 <= a1 )
    goto LABEL_6;
  v6 = a1 / 0xA + a1 - (qword_4F19578 - a2);
  if ( v5 >= v6 )
    v6 = qword_4F19578 - (_QWORD)qword_4F19580;
  v7 = v5 + v6;
  if ( v7 + 1 < v3 + a1 )
LABEL_6:
    sub_685240(0x6D9u);
  v8 = (void *)sub_822C60(qword_4F19580, v5 + 1, v7 + 1);
  result = sub_81A600((unsigned __int64)qword_4F19580, qword_4F19578, (__int64)v8, 1);
  qword_4F19580 = v8;
  qword_4F19578 = (__int64)v8 + v7;
  return result;
}
