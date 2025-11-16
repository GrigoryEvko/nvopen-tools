// Function: sub_720FF0
// Address: 0x720ff0
//
void __fastcall __noreturn sub_720FF0(unsigned __int8 a1)
{
  if ( dword_4D04198 )
    goto LABEL_4;
  if ( (unsigned __int8)(a1 - 9) <= 1u )
  {
    fwrite("Compilation terminated.\n", 1u, 0x18u, qword_4F07510);
    goto LABEL_10;
  }
  if ( a1 == 11 )
  {
    fwrite("Compilation aborted.\n", 1u, 0x15u, qword_4F07510);
  }
  else
  {
LABEL_4:
    if ( a1 > 5u )
    {
      if ( a1 == 8 )
        sub_7208E0(2);
    }
    else if ( a1 > 2u )
    {
      sub_7208E0(0);
    }
  }
LABEL_10:
  sub_7208E0(4);
}
