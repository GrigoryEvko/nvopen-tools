// Function: sub_CEA700
// Address: 0xcea700
//
char __fastcall sub_CEA700(char a1)
{
  char *v1; // rax
  char result; // al

  v1 = (char *)sub_C94E20((__int64)qword_4F863F0);
  if ( v1 )
    result = *v1;
  else
    result = qword_4F863F0[2];
  if ( !result )
    return a1 == 1;
  return result;
}
