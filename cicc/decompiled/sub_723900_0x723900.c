// Function: sub_723900
// Address: 0x723900
//
__int64 sub_723900()
{
  char *v0; // rax

  sub_8D0840(&qword_4F076A8, 8, 0);
  sub_8D0840(&unk_4F07688, 8, 0);
  sub_7209D0(unk_4F076B0, &qword_4F07698, &qword_4F07690);
  v0 = getenv("EDG_MODULES_PATH");
  if ( v0 )
    sub_720930((__int64)v0, 0, &qword_4F07698, (__int64)&qword_4F07690);
  return sub_8D0840(&qword_4F078E0, 8, 392);
}
