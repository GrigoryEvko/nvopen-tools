// Function: ctor_649
// Address: 0x5985c0
//
_QWORD *ctor_649()
{
  _QWORD *result; // rax
  _QWORD *v1; // rcx

  qword_5037A88 = 5;
  qword_5037A80 = (__int64)"ocaml";
  qword_5037A90 = (__int64)"ocaml 3.10-compatible collector";
  qword_5037AA0 = (__int64)sub_3255870;
  qword_5037AB0 = (__int64)&qword_5037A80;
  result = &qword_503B138;
  qword_5037A98 = 31;
  qword_5037AA8 = 0;
  v1 = (_QWORD *)qword_503B138;
  if ( !qword_503B138 )
    v1 = &unk_503B140;
  *v1 = &qword_5037AA8;
  qword_503B138 = &qword_5037AA8;
  return result;
}
