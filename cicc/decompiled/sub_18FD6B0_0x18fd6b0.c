// Function: sub_18FD6B0
// Address: 0x18fd6b0
//
unsigned __int64 __fastcall sub_18FD6B0(_DWORD *a1, int *a2, _QWORD *a3, _QWORD *a4)
{
  int v6; // eax
  unsigned __int64 v8; // rax
  _QWORD v9[22]; // [rsp+0h] [rbp-B0h] BYREF

  if ( !byte_4F99930[0] && (unsigned int)sub_2207590(byte_4F99930) )
  {
    v8 = unk_4FA04C8;
    if ( !unk_4FA04C8 )
      v8 = 0xFF51AFD7ED558CCDLL;
    qword_4F99938 = v8;
    sub_2207640(byte_4F99930);
  }
  LODWORD(v9[0]) = *a1;
  v6 = *a2;
  v9[15] = qword_4F99938;
  HIDWORD(v9[0]) = v6;
  v9[1] = *a3;
  v9[2] = *a4;
  return sub_1593600(v9, 0x18u, qword_4F99938);
}
