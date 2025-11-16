// Function: sub_CB6D50
// Address: 0xcb6d50
//
__int64 *__fastcall sub_CB6D50(__int64 *a1, int a2, unsigned __int8 a3, unsigned __int8 a4)
{
  char *v6; // r13
  size_t v7; // rax

  if ( (unsigned __int8)sub_CB6CE0(a1) )
  {
    if ( a2 == 16 )
      v6 = (char *)sub_C86450();
    else
      v6 = sub_C86410(a2, a3, a4);
    if ( v6 )
    {
      v7 = strlen(v6);
      sub_CB6200((__int64)a1, (unsigned __int8 *)v6, v7);
    }
  }
  return a1;
}
