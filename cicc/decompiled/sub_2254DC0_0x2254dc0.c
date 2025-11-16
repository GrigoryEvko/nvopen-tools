// Function: sub_2254DC0
// Address: 0x2254dc0
//
__int64 sub_2254DC0()
{
  unsigned int mb_cur_max; // ebx

  __uselocale();
  mb_cur_max = __ctype_get_mb_cur_max();
  __uselocale();
  return mb_cur_max;
}
