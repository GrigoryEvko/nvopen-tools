// Function: sub_8539E0
// Address: 0x8539e0
//
char *sub_8539E0()
{
  size_t v0; // rax
  char *result; // rax

  sprintf(byte_4F5FB80, "EDG C/C++ version %s (%s %s)\n", "6.6", off_4B6EB10, off_4B6EB08[0]);
  v0 = strlen(byte_4F5FB80);
  qword_4F5FB70 = 0;
  qword_4F5FB68 = 0;
  size = v0 + 1;
  dword_4D03CAC = 0;
  unk_4D03CA8 = 0;
  qword_4F5FB50 = 0;
  dword_4D03CB0[0] = 0;
  qword_4F5FB48 = 0;
  qword_4F5FB40 = 0;
  qword_4F5F860 = 0;
  *(_QWORD *)&dword_4D03CA0 = *(_QWORD *)&dword_4F077C8;
  qword_4F5F868 = 0;
  dword_4D03C98[0] = 0;
  qword_4F5F870 = 0;
  unk_4D03C84 = 0;
  unk_4D03C80 = 0;
  dword_4D03C94 = 0;
  unk_4D03C88 = *(_QWORD *)&dword_4F077C8;
  dword_4D03C90 = 0;
  result = qword_4F076F0;
  if ( *qword_4F076F0 == 45 )
    dword_4D03CAC = qword_4F076F0[1] == 0;
  return result;
}
