// Function: sub_2F41140
// Address: 0x2f41140
//
void __fastcall sub_2F41140(_QWORD *a1)
{
  _QWORD *v1; // rax
  _QWORD **v2; // rdx
  _QWORD **v3; // rbx

  v1 = (_QWORD *)qword_5023860[0];
  if ( qword_5023860[0] )
  {
    if ( a1 == (_QWORD *)qword_5023860[0] )
    {
      v3 = (_QWORD **)qword_5023860;
LABEL_8:
      if ( qword_5023870 )
        (*(void (__fastcall **)(_QWORD *, _QWORD, _QWORD))(*qword_5023870 + 32LL))(qword_5023870, a1[1], a1[2]);
      *v3 = (_QWORD *)**v3;
    }
    else
    {
      while ( 1 )
      {
        v2 = (_QWORD **)v1;
        v1 = (_QWORD *)*v1;
        if ( !v1 )
          break;
        if ( a1 == v1 )
        {
          v3 = v2;
          goto LABEL_8;
        }
      }
    }
  }
}
