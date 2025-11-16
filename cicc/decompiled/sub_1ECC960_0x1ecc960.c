// Function: sub_1ECC960
// Address: 0x1ecc960
//
unsigned __int64 __fastcall sub_1ECC960(_DWORD *a1, int *a2, _QWORD *a3)
{
  int v4; // eax
  unsigned __int64 v6; // rax
  _QWORD v7[22]; // [rsp+0h] [rbp-B0h] BYREF

  if ( !byte_4F99930[0] && (unsigned int)sub_2207590(byte_4F99930) )
  {
    v6 = unk_4FA04C8;
    if ( !unk_4FA04C8 )
      v6 = 0xFF51AFD7ED558CCDLL;
    qword_4F99938 = v6;
    sub_2207640(byte_4F99930);
  }
  LODWORD(v7[0]) = *a1;
  v4 = *a2;
  v7[15] = qword_4F99938;
  HIDWORD(v7[0]) = v4;
  v7[1] = *a3;
  return sub_1593600(v7, 0x10u, qword_4F99938);
}
