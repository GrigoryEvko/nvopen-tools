// Function: ctor_647
// Address: 0x597140
//
_QWORD *ctor_647()
{
  _QWORD *result; // rax
  _QWORD *v1; // rcx

  qword_5036388 = 6;
  qword_5036380 = (__int64)"erlang";
  qword_5036390 = (__int64)"erlang-compatible garbage collector";
  qword_50363A0 = (__int64)sub_3216D20;
  qword_50363B0 = (__int64)&qword_5036380;
  result = &qword_503B138;
  qword_5036398 = 35;
  qword_50363A8 = 0;
  v1 = (_QWORD *)qword_503B138;
  if ( !qword_503B138 )
    v1 = &unk_503B140;
  *v1 = &qword_50363A8;
  qword_503B138 = &qword_50363A8;
  return result;
}
