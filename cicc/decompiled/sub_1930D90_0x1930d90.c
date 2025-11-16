// Function: sub_1930D90
// Address: 0x1930d90
//
unsigned __int64 __fastcall sub_1930D90(_QWORD *a1, __int64 *a2, _QWORD *a3)
{
  __int64 v4; // rax
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
  v7[0] = *a1;
  v4 = *a2;
  v7[15] = qword_4F99938;
  v7[1] = v4;
  v7[2] = *a3;
  return sub_1593600(v7, 0x18u, qword_4F99938);
}
