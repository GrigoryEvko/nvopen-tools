// Function: sub_82D8A0
// Address: 0x82d8a0
//
void __fastcall sub_82D8A0(_QWORD *a1)
{
  _QWORD *v1; // rdx
  _QWORD *v2; // rax

  if ( a1 )
  {
    v1 = (_QWORD *)unk_4D03C60;
    while ( 1 )
    {
      v2 = (_QWORD *)*a1;
      *a1 = v1;
      v1 = a1;
      unk_4D03C60 = a1;
      if ( !v2 )
        break;
      a1 = v2;
    }
  }
}
