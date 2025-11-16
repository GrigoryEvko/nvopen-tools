// Function: sub_C8C4D0
// Address: 0xc8c4d0
//
int sub_C8C4D0()
{
  int result; // eax
  struct sigaction *v1; // rbx
  struct sigaction *v2; // r12

  result = dword_4F84BA0;
  if ( dword_4F84BA0 )
  {
    v1 = &act;
    v2 = (struct sigaction *)((char *)&unk_4F84240 + 160 * (unsigned int)(dword_4F84BA0 - 1));
    do
    {
      result = sigaction((int)v1[1].sa_handler, v1, 0);
      _InterlockedSub(&dword_4F84BA0, 1u);
      v1 = (struct sigaction *)((char *)v1 + 160);
    }
    while ( v1 != v2 );
  }
  return result;
}
