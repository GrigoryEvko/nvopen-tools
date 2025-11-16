// Function: sub_67B720
// Address: 0x67b720
//
void *__fastcall sub_67B720(const void *a1)
{
  void *result; // rax
  size_t v2; // rdx
  const void *v3; // rsi

  result = (void *)qword_4CFFD80;
  if ( qword_4CFFD80 )
  {
    v2 = *(_QWORD *)(qword_4CFFD80 + 16);
    if ( v2 )
    {
      v3 = *(const void **)qword_4CFFD80;
      qword_4CFDE98 = 0;
      result = bsearch(a1, v3, v2, 0x18u, compar);
      if ( !result )
        return (void *)qword_4CFDE98;
    }
    else
    {
      return 0;
    }
  }
  return result;
}
