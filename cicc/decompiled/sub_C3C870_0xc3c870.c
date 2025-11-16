// Function: sub_C3C870
// Address: 0xc3c870
//
void *__fastcall sub_C3C870(void **a1, void **a2)
{
  void *v2; // rbx
  void *result; // rax

  v2 = sub_C33340();
  result = *a2;
  if ( *a1 == v2 )
  {
    if ( v2 == result )
    {
      if ( a2 != a1 )
      {
        sub_969EE0((__int64)a1);
        return sub_C3C840(a1, a2);
      }
    }
    else if ( a1 != a2 )
    {
      sub_969EE0((__int64)a1);
LABEL_6:
      if ( v2 != *a2 )
        return (void *)sub_C338E0((__int64)a1, (__int64)a2);
      return sub_C3C840(a1, a2);
    }
  }
  else
  {
    if ( v2 != result )
      return (void *)sub_C33870((__int64)a1, (__int64)a2);
    if ( a1 != a2 )
    {
      sub_C338F0((__int64)a1);
      goto LABEL_6;
    }
  }
  return result;
}
