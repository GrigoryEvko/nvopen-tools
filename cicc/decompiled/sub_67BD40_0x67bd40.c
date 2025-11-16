// Function: sub_67BD40
// Address: 0x67bd40
//
__int64 sub_67BD40()
{
  __int64 v0; // rdi
  __int64 v1; // rax

  sub_8238B0(qword_4D039D8, *(_QWORD *)(qword_4D039E8 + 32), *(_QWORD *)(qword_4D039E8 + 16));
  v0 = qword_4D039D8;
  v1 = *(_QWORD *)(qword_4D039D8 + 16);
  if ( (unsigned __int64)(v1 + 1) > *(_QWORD *)(qword_4D039D8 + 8) )
  {
    sub_823810();
    v0 = qword_4D039D8;
    v1 = *(_QWORD *)(qword_4D039D8 + 16);
  }
  *(_BYTE *)(*(_QWORD *)(v0 + 32) + v1) = 10;
  ++*(_QWORD *)(v0 + 16);
  return sub_823800(qword_4D039E8);
}
