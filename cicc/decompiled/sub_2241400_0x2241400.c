// Function: sub_2241400
// Address: 0x2241400
//
unsigned __int64 *__fastcall sub_2241400(unsigned __int64 *a1, size_t a2, unsigned __int64 a3, _BYTE *a4, size_t a5)
{
  size_t v5; // r9

  v5 = a1[1];
  if ( v5 - a2 <= a3 )
    a3 = a1[1] - a2;
  if ( a2 > v5 )
    sub_222CF80("%s: __pos (which is %zu) > this->size() (which is %zu)", "basic_string::replace", a2, v5);
  return sub_2241130(a1, a2, a3, a4, a5);
}
