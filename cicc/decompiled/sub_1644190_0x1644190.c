// Function: sub_1644190
// Address: 0x1644190
//
unsigned __int64 __fastcall sub_1644190(__int64 *a1, char *a2)
{
  char v2; // al
  unsigned __int64 v4; // rax
  __int64 v5; // [rsp+0h] [rbp-A0h] BYREF
  char v6; // [rsp+8h] [rbp-98h]
  __int64 v7; // [rsp+78h] [rbp-28h]

  if ( !byte_4F99930[0] && (unsigned int)sub_2207590(byte_4F99930) )
  {
    v4 = unk_4FA04C8;
    if ( !unk_4FA04C8 )
      v4 = 0xFF51AFD7ED558CCDLL;
    qword_4F99938 = v4;
    sub_2207640(byte_4F99930);
  }
  v5 = *a1;
  v2 = *a2;
  v7 = qword_4F99938;
  v6 = v2;
  return sub_1593600(&v5, 9u, qword_4F99938);
}
