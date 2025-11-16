// Function: sub_721180
// Address: 0x721180
//
time_t __fastcall sub_721180(_QWORD *a1)
{
  time_t result; // rax

  *a1 = (unsigned int)(int)((double)(int)clock() * 1000.0 / 1000000.0);
  result = time(0);
  a1[1] = result;
  return result;
}
