// Function: sub_16E8100
// Address: 0x16e8100
//
__int64 *__fastcall sub_16E8100(__int64 *a1)
{
  const char *v1; // rax
  char *v2; // r13
  size_t v3; // rbx

  if ( (unsigned __int8)sub_16C6BD0() && a1[3] != a1[1] )
    sub_16E7BA0(a1);
  v1 = sub_16C6C20();
  v2 = (char *)v1;
  if ( v1 )
  {
    v3 = strlen(v1);
    sub_16E7EE0((__int64)a1, v2, v3);
    a1[8] -= v3;
  }
  return a1;
}
