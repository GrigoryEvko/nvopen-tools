// Function: sub_2254D80
// Address: 0x2254d80
//
_BOOL8 sub_2254D80()
{
  size_t mb_cur_max; // rbx

  __uselocale();
  mb_cur_max = __ctype_get_mb_cur_max();
  __uselocale();
  return mb_cur_max == 1;
}
