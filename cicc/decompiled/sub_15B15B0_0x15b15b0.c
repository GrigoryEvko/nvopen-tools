// Function: sub_15B15B0
// Address: 0x15b15b0
//
unsigned __int64 __fastcall sub_15B15B0(_QWORD *a1, __int64 *a2)
{
  __int64 v2; // rax
  unsigned __int64 v4; // rax
  _QWORD v5[20]; // [rsp+0h] [rbp-A0h] BYREF

  if ( !byte_4F99930[0] && (unsigned int)sub_2207590(byte_4F99930) )
  {
    v4 = unk_4FA04C8;
    if ( !unk_4FA04C8 )
      v4 = 0xFF51AFD7ED558CCDLL;
    qword_4F99938 = v4;
    sub_2207640(byte_4F99930);
  }
  v5[0] = *a1;
  v2 = *a2;
  v5[15] = qword_4F99938;
  v5[1] = v2;
  return sub_1593600(v5, 0x10u, qword_4F99938);
}
