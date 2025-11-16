// Function: ctor_021
// Address: 0x455b80
//
_QWORD *ctor_021()
{
  qword_4F81430[0] = 0;
  qword_4F81430[2] = &qword_4F81430[1];
  qword_4F81430[1] = (char *)&qword_4F81430[1] + 4;
  return qword_4F81430;
}
