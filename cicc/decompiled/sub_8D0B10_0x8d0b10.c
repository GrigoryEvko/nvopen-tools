// Function: sub_8D0B10
// Address: 0x8d0b10
//
void sub_8D0B10()
{
  __int64 *v0; // rax
  __int64 v1; // rcx

  v0 = qword_4D03FF8;
  if ( (_QWORD *)qword_4D03FF8[1] != qword_4D03FD0 )
    --dword_4D03FC8[0];
  qword_4D03FF8 = (_QWORD *)*qword_4D03FF8;
  v1 = qword_4F60550;
  qword_4F60550 = (__int64)v0;
  *v0 = v1;
  if ( qword_4D03FF8 )
    sub_8D0910((_QWORD *)qword_4D03FF8[1]);
}
