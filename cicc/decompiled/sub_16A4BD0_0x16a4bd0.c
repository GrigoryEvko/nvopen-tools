// Function: sub_16A4BD0
// Address: 0x16a4bd0
//
unsigned __int64 __fastcall sub_16A4BD0(__int64 *a1)
{
  __int64 v1; // rax
  unsigned __int64 v3; // rax
  _QWORD v4[20]; // [rsp+0h] [rbp-A0h] BYREF

  if ( !byte_4F99930[0] && (unsigned int)sub_2207590(byte_4F99930) )
  {
    v3 = unk_4FA04C8;
    if ( !unk_4FA04C8 )
      v3 = 0xFF51AFD7ED558CCDLL;
    qword_4F99938 = v3;
    sub_2207640(byte_4F99930);
  }
  v1 = *a1;
  v4[15] = qword_4F99938;
  v4[0] = v1;
  return sub_1593600(v4, 8u, qword_4F99938);
}
