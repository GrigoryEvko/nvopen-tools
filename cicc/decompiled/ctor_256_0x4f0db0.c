// Function: ctor_256
// Address: 0x4f0db0
//
int ctor_256()
{
  dword_4FBA468[0] = 0;
  *(_QWORD *)&dword_4FBA468[2] = 0;
  *(_QWORD *)&dword_4FBA468[4] = dword_4FBA468;
  *(_QWORD *)&dword_4FBA468[6] = dword_4FBA468;
  *(_QWORD *)&dword_4FBA468[8] = 0;
  return __cxa_atexit(sub_705D20, &unk_4FBA460, &qword_4A427C0);
}
