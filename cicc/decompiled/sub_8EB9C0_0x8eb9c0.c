// Function: sub_8EB9C0
// Address: 0x8eb9c0
//
char *__fastcall sub_8EB9C0(unsigned __int8 *a1, unsigned int a2, int a3, __int64 a4)
{
  char *v6; // r14

  v6 = sub_8E9FF0((__int64)a1, 0, 0, 0, a2, a4);
  if ( a3 && !*(_QWORD *)(a4 + 32) )
    sub_8E5790((unsigned __int8 *)"...", a4);
  sub_8EB260(a1, 0, 0, a4);
  return v6;
}
