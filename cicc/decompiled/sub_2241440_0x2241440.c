// Function: sub_2241440
// Address: 0x2241440
//
unsigned __int64 *__fastcall sub_2241440(unsigned __int64 *a1, size_t a2, char *a3)
{
  size_t v4; // rax
  size_t v5; // rcx

  v4 = strlen(a3);
  v5 = a1[1];
  if ( a2 > v5 )
    sub_222CF80("%s: __pos (which is %zu) > this->size() (which is %zu)", "basic_string::replace", a2, v5);
  return sub_2241130(a1, a2, 0, a3, v4);
}
