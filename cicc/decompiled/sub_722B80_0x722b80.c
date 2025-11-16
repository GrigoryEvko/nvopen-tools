// Function: sub_722B80
// Address: 0x722b80
//
int __fastcall sub_722B80(unsigned __int8 *a1, unsigned __int8 *a2, int a3)
{
  char *v4; // r13
  char *v5; // r12
  int result; // eax
  __dev_t v7[2]; // [rsp+0h] [rbp-40h] BYREF
  __dev_t v8[6]; // [rsp+10h] [rbp-30h] BYREF

  if ( !qword_4F078C8 )
    qword_4F078C8 = sub_8237A0(128);
  if ( !qword_4F078C0 )
    qword_4F078C0 = sub_8237A0(128);
  v4 = (char *)sub_721FB0(a1, (_QWORD *)qword_4F078C8, a3);
  v5 = (char *)sub_721FB0(a2, (_QWORD *)qword_4F078C0, a3);
  result = strcmp(v4, v5);
  if ( result )
  {
    if ( !a3 )
    {
      sub_7217C0(v4, v7);
      sub_7217C0(v5, v8);
      return !sub_721820((__int64)v7, v8);
    }
  }
  return result;
}
