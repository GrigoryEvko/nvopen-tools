// Function: sub_5C62E0
// Address: 0x5c62e0
//
void sub_5C62E0()
{
  __int64 v0; // rax
  unsigned __int64 i; // rbx

  if ( !byte_4CF6E00 )
  {
    v0 = qword_4CF6E08;
    for ( i = &qword_496EDF8 - qword_496EDF0 - 1; qword_4CF6E08 < i; v0 = qword_4CF6E08 )
    {
      qword_4CF6E08 = v0 + 1;
      ((void (*)(void))qword_496EDF0[v0 + 1])();
    }
    sub_5C6270();
    byte_4CF6E00 = 1;
  }
}
