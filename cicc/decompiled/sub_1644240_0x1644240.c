// Function: sub_1644240
// Address: 0x1644240
//
unsigned __int64 __fastcall sub_1644240(_QWORD *a1, __int64 *a2, char *a3)
{
  __int64 v4; // rax
  unsigned __int64 v6; // rax
  _QWORD v7[2]; // [rsp+0h] [rbp-B0h] BYREF
  char v8; // [rsp+10h] [rbp-A0h]
  __int64 v9; // [rsp+78h] [rbp-38h]

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
  v9 = qword_4F99938;
  v7[1] = v4;
  v8 = *a3;
  return sub_1593600(v7, 0x11u, qword_4F99938);
}
