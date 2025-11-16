// Function: sub_16E8000
// Address: 0x16e8000
//
__int64 *__fastcall sub_16E8000(__int64 *a1, int a2, unsigned __int8 a3, unsigned __int8 a4)
{
  char *v6; // r13
  size_t v7; // rbx

  if ( (unsigned __int8)sub_16C6BD0() && a1[3] != a1[1] )
    sub_16E7BA0(a1);
  if ( a2 == 8 )
    v6 = (char *)sub_16C6C10();
  else
    v6 = sub_16C6BE0(a2, a3, a4);
  if ( v6 )
  {
    v7 = strlen(v6);
    sub_16E7EE0((__int64)a1, v6, v7);
    a1[8] -= v7;
  }
  return a1;
}
