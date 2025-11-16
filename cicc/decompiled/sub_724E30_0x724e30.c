// Function: sub_724E30
// Address: 0x724e30
//
_QWORD *__fastcall sub_724E30(__int64 a1)
{
  *(_QWORD *)(*(_QWORD *)a1 + 120LL) = qword_4F06BB8;
  qword_4F06BB8 = *(_QWORD *)a1;
  *(_QWORD *)a1 = 0;
  return &qword_4F06BB8;
}
