// Function: sub_16A4E10
// Address: 0x16a4e10
//
unsigned __int64 __fastcall sub_16A4E10(__int64 *a1)
{
  __int64 v2; // rdi
  unsigned __int64 v3; // rbx
  unsigned __int64 v4; // r12
  unsigned __int64 v6; // rax
  _QWORD v7[20]; // [rsp+0h] [rbp-A0h] BYREF

  v2 = a1[1];
  if ( !v2 )
    return sub_16A4BD0(a1);
  v3 = sub_16A4DD0(v2 + 32);
  v4 = sub_16A4DD0(a1[1]);
  if ( !byte_4F99930[0] && (unsigned int)sub_2207590(byte_4F99930) )
  {
    v6 = unk_4FA04C8;
    if ( !unk_4FA04C8 )
      v6 = 0xFF51AFD7ED558CCDLL;
    qword_4F99938 = v6;
    sub_2207640(byte_4F99930);
  }
  v7[0] = v4;
  v7[1] = v3;
  v7[15] = qword_4F99938;
  return sub_1593600(v7, 0x10u, qword_4F99938);
}
