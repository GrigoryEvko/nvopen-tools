// Function: sub_88B7B0
// Address: 0x88b7b0
//
void *sub_88B7B0()
{
  __int64 v0; // rdx
  __int64 v1; // rcx
  __int64 v2; // r8
  __int64 v3; // r9

  if ( unk_4D04508 )
    sub_8539C0((__int64)&off_4B7DB00);
  sub_8D0840(&qword_4D03FB8, 8, 0);
  sub_8D0840(&qword_4F600F8, 8, 0);
  qword_4D03FB8 = 0;
  qword_4F600F8 = 0;
  qword_4D03FB0 = (void *)sub_822B10(24LL * unk_4A598C8, 8, v0, v1, v2, v3);
  return memset(qword_4D03FB0, 0, 24LL * unk_4A598C8);
}
