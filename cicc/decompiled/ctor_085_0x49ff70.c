// Function: ctor_085
// Address: 0x49ff70
//
_QWORD *ctor_085()
{
  _QWORD *result; // rax
  _QWORD *v1; // rdx

  qword_4F8FDA8 = 6;
  qword_4F8FDA0 = (__int64)"erlang";
  qword_4F8FDB0 = (__int64)"erlang-compatible garbage collector";
  qword_4F8FDC0 = (__int64)sub_10608B0;
  qword_4F8FDD0 = (__int64)&qword_4F8FDA0;
  result = &qword_4F8A310;
  qword_4F8FDB8 = 35;
  qword_4F8FDC8 = 0;
  v1 = (_QWORD *)qword_4F8A310;
  if ( !qword_4F8A310 )
    v1 = &unk_4F8A318;
  *v1 = &qword_4F8FDC8;
  qword_4F8FD60 = (__int64)"ocaml";
  qword_4F8FD80 = (__int64)sub_1060900;
  qword_4F8FD70 = (__int64)"ocaml 3.10-compatible GC";
  qword_4F8FD90 = (__int64)&qword_4F8FD60;
  qword_4F8FD20 = (__int64)"shadow-stack";
  qword_4F8FDC8 = (__int64)&qword_4F8FD88;
  qword_4F8FD30 = (__int64)"Very portable GC for uncooperative code generators";
  qword_4F8FD40 = (__int64)sub_1060950;
  qword_4F8FD88 = (__int64)&qword_4F8FD48;
  qword_4F8FCE0 = (__int64)"statepoint-example";
  qword_4F8FD50 = (__int64)&qword_4F8FD20;
  qword_4F8FD00 = (__int64)sub_10609A0;
  qword_4F8FD48 = (__int64)&qword_4F8FD08;
  qword_4F8FCF0 = (__int64)"an example strategy for statepoint";
  qword_4F8FD10 = (__int64)&qword_4F8FCE0;
  qword_4F8FCB0 = (__int64)"CoreCLR-compatible GC";
  qword_4F8FD68 = 5;
  qword_4F8FD78 = 24;
  qword_4F8FD28 = 12;
  qword_4F8FD38 = 50;
  qword_4F8FCE8 = 18;
  qword_4F8FCF8 = 34;
  qword_4F8FCA0 = (__int64)"coreclr";
  qword_4F8FCA8 = 7;
  qword_4F8FCB8 = 21;
  qword_4F8FCC0 = (__int64)sub_10609F0;
  qword_4F8FCC8 = 0;
  qword_4F8FCD0 = (__int64)&qword_4F8FCA0;
  qword_4F8FD08 = (__int64)&qword_4F8FCC8;
  qword_4F8A310 = &qword_4F8FCC8;
  return result;
}
