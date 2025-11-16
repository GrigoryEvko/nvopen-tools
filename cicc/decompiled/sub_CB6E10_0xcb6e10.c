// Function: sub_CB6E10
// Address: 0xcb6e10
//
__int64 *__fastcall sub_CB6E10(__int64 *a1)
{
  const char *v2; // rax
  unsigned __int8 *v3; // r13
  size_t v4; // rax

  if ( !(unsigned __int8)sub_CB6CE0(a1) )
    return a1;
  v2 = sub_C86460();
  v3 = (unsigned __int8 *)v2;
  if ( !v2 )
    return a1;
  v4 = strlen(v2);
  sub_CB6200((__int64)a1, v3, v4);
  return a1;
}
