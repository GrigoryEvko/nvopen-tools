// Function: sub_68B390
// Address: 0x68b390
//
void __fastcall sub_68B390(char *a1, int a2)
{
  if ( qword_4F06C50 != (_QWORD *)a1 )
    sub_7295A0(a1);
  if ( a2 )
  {
    if ( unk_4F06C40 )
    {
      if ( !*((_BYTE *)qword_4F06C50 + unk_4F06C40 - 1) )
        --unk_4F06C40;
    }
    sub_7295A0("\"");
  }
}
