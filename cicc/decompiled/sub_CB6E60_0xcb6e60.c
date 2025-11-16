// Function: sub_CB6E60
// Address: 0xcb6e60
//
__int64 *__fastcall sub_CB6E60(__int64 *a1, __int64 a2)
{
  __int64 v2; // rax
  void (*v4)(void); // rax
  const char *v5; // rax
  unsigned __int8 *v6; // r13
  size_t v7; // rax

  v2 = *a1;
  if ( (_DWORD)a2 == 17 )
  {
    v4 = *(void (**)(void))(v2 + 32);
    if ( (char *)v4 == (char *)sub_CB6DC0 )
    {
      if ( (unsigned __int8)sub_CB6CE0(a1) )
      {
        v5 = sub_C86470();
        v6 = (unsigned __int8 *)v5;
        if ( v5 )
        {
          v7 = strlen(v5);
          sub_CB6200((__int64)a1, v6, v7);
        }
      }
    }
    else
    {
      v4();
    }
  }
  else
  {
    (*(void (__fastcall **)(__int64 *, __int64, _QWORD, _QWORD))(v2 + 24))(a1, a2, 0, 0);
  }
  return a1;
}
