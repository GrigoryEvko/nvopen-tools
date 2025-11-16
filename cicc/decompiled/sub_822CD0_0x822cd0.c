// Function: sub_822CD0
// Address: 0x822cd0
//
void *__fastcall sub_822CD0(size_t a1)
{
  __off_t v2; // rdx
  __int64 v3; // rdi
  __int64 v4; // r8
  __int64 v5; // r9
  void *v6; // r13
  __int64 v7; // rax
  char *v8; // rdi
  __int64 v9; // rdx
  char *v10; // rax

  if ( dword_4F195C8 )
  {
    v2 = qword_4F195B8;
    v3 = qword_4F195C0;
  }
  else
  {
    sub_7219B0();
    v3 = 0;
    v2 = 0;
    qword_4F195C0 = 0;
    dword_4F195C8 = 1;
    qword_4F195B8 = 0;
  }
  v6 = sub_721870(v3, a1, v2);
  if ( !v6 )
    sub_685240(0x277u);
  qword_4F195C0 += a1;
  qword_4F195B8 += a1;
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
                              v4,
                              v5);
    v8 = (char *)qword_4F07380;
    v7 = qword_4F07398;
  }
  v9 = v7 + 1;
  v10 = &v8[16 * v7];
  qword_4F07398 = v9;
  *((_QWORD *)v10 + 1) = a1;
  *(_QWORD *)v10 = v6;
  qword_4F07388 = qword_4F07398;
  return v6;
}
