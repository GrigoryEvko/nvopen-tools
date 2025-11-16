// Function: sub_1CFC0C0
// Address: 0x1cfc0c0
//
void __fastcall sub_1CFC0C0(_QWORD *a1)
{
  _QWORD *v1; // rax
  _QWORD **v2; // rbx

  v1 = (_QWORD *)qword_4FC1B10[0];
  if ( qword_4FC1B10[0] )
  {
    v2 = (_QWORD **)qword_4FC1B10;
    while ( a1 != v1 )
    {
      v2 = (_QWORD **)v1;
      v1 = (_QWORD *)*v1;
      if ( !v1 )
        return;
    }
    if ( qword_4FC1B10[2] )
      (*(void (__fastcall **)(_QWORD, _QWORD, _QWORD))(*(_QWORD *)qword_4FC1B10[2] + 32LL))(
        qword_4FC1B10[2],
        a1[1],
        a1[2]);
    *v2 = (_QWORD *)**v2;
  }
  else
  {
    nullsub_2032(qword_4FC1B10, a1);
  }
}
