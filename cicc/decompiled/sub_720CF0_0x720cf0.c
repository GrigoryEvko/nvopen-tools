// Function: sub_720CF0
// Address: 0x720cf0
//
_QWORD *__fastcall sub_720CF0(char *s, const char *a2, _QWORD *a3)
{
  _QWORD *v3; // r12
  size_t v4; // rax

  v3 = a3;
  if ( !a3 )
  {
    v3 = (_QWORD *)qword_4F07930;
    if ( !qword_4F07930 )
    {
      qword_4F07930 = sub_8237A0(256);
      v3 = (_QWORD *)qword_4F07930;
    }
  }
  sub_823800(v3);
  v4 = strlen(s);
  sub_8238B0(v3, s, v4);
  sub_720C20(v3, a2);
  return v3;
}
