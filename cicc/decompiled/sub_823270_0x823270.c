// Function: sub_823270
// Address: 0x823270
//
_QWORD *__fastcall sub_823270(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // rax
  char *v8; // rdi
  __int64 v9; // rdx
  char *v10; // rax

  v7 = qword_4F07398;
  v8 = (char *)qword_4F07380;
  if ( qword_4F07398 == qword_4F07390 )
  {
    qword_4F07390 = qword_4F07398 + 500;
    qword_4F07380 = (void *)sub_822C60(
                              qword_4F07380,
                              16 * (qword_4F07398 + 500) - 8000,
                              16 * (qword_4F07398 + 500),
                              (__int64)&qword_4F07390,
                              a5,
                              a6);
    v8 = (char *)qword_4F07380;
    v7 = qword_4F07398;
  }
  v9 = v7 + 1;
  qword_4F195C0 += a2;
  v10 = &v8[16 * v7];
  qword_4F07398 = v9;
  *((_QWORD *)v10 + 1) = a2;
  *(_QWORD *)v10 = a1;
  qword_4F07388 = qword_4F07398;
  return &qword_4F07388;
}
