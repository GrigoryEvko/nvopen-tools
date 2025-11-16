// Function: sub_334CAC0
// Address: 0x334cac0
//
void __fastcall sub_334CAC0(_QWORD *a1)
{
  _QWORD *v1; // rax
  _QWORD **v2; // rdx
  _QWORD **v3; // rbx

  v1 = (_QWORD *)qword_5039AB0[0];
  if ( qword_5039AB0[0] )
  {
    if ( a1 == (_QWORD *)qword_5039AB0[0] )
    {
      v3 = (_QWORD **)qword_5039AB0;
LABEL_8:
      if ( unk_5039AC0 )
        (*(void (__fastcall **)(_QWORD, _QWORD, _QWORD))(*unk_5039AC0 + 32LL))(unk_5039AC0, a1[1], a1[2]);
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
