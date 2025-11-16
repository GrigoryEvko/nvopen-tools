// Function: sub_867E30
// Address: 0x867e30
//
_QWORD *sub_867E30()
{
  int v0; // eax
  _QWORD *v1; // rbx
  _QWORD *result; // rax
  _QWORD *v3; // rcx
  _QWORD *v4; // rdx

  qword_4F04C68[0] = 0;
  dword_4D049E0 = 0;
  dword_4F04C64 = -1;
  dword_4F04C60 = -1;
  dword_4F04C5C = -1;
  dword_4F04C58 = -1;
  qword_4F04C50 = 0;
  unk_4F04C48 = -1;
  dword_4F04C44 = -1;
  dword_4F04C40 = -1;
  dword_4F04C38 = 0;
  dword_4F04C34 = -1;
  dword_4F04C2C = -1;
  unk_4F04C28 = 0;
  qword_4F04C18 = 0;
  qword_4F5FD18 = 0;
  dword_4F04C3C = dword_4D03FE8[0] == 0;
  if ( dword_4F077C4 == 2 )
    v0 = sub_7E16F0();
  else
    v0 = sub_7D7670();
  dword_4F04C3C |= v0;
  dword_4F5FD28 = v0;
  v1 = (_QWORD *)sub_823970(16);
  result = &qword_4F04C10;
  qword_4F04C10 = v1;
  if ( v1 )
  {
    result = (_QWORD *)sub_823970(0x4000);
    v3 = result;
    v4 = result + 2048;
    do
    {
      if ( result )
        *result = 0;
      result += 2;
    }
    while ( result != v4 );
    *v1 = v3;
    v1[1] = 1023;
  }
  return result;
}
