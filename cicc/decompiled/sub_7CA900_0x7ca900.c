// Function: sub_7CA900
// Address: 0x7ca900
//
_QWORD *sub_7CA900()
{
  _QWORD *result; // rax
  _QWORD *v1; // rbx
  _QWORD *v2; // rcx
  _QWORD *v3; // rdx

  qword_4F08558 = 0;
  qword_4F08550 = 0;
  qword_4F06448 = 0;
  qword_4F08540 = 0;
  qword_4F06430 = 0;
  qword_4F08530 = 0;
  qword_4F064A0 = 0;
  qword_4F08528 = 0;
  unk_4F06428 = 0;
  qword_4F17F48 = 0;
  qword_4F06420 = 0;
  qword_4F08548 = 0;
  unk_4D03D28 = 0;
  dword_4F17F78 = 0;
  qword_4F17F68 = 0;
  qword_4F17F60 = 0;
  qword_4F061D0 = *(_QWORD *)&dword_4F077C8;
  qword_4F17F70 = *(_QWORD *)&dword_4F077C8;
  dword_4F04D80 = 0;
  sub_7461E0((__int64)&qword_4F083E0);
  byte_4F08468 = 1;
  qword_4F083E0 = (__int64)sub_729610;
  qword_4F04D90 = 0;
  dword_4F04D88[0] = 0;
  result = (_QWORD *)sub_823970(16);
  qword_4F08490 = (__int64)result;
  if ( result )
  {
    v1 = result;
    result = (_QWORD *)sub_823970(4096);
    v2 = result;
    v3 = result + 512;
    do
    {
      if ( result )
        *result = 0;
      result += 2;
    }
    while ( result != v3 );
    *v1 = v2;
    v1[1] = 255;
  }
  qword_4F084D0 = 0;
  qword_4F084C8 = 0;
  qword_4F084A8 = 0;
  qword_4F084A0 = 0;
  qword_4F08498 = 0;
  return result;
}
