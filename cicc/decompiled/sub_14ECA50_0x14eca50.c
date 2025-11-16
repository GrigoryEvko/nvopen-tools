// Function: sub_14ECA50
// Address: 0x14eca50
//
__int64 *__fastcall sub_14ECA50(__int64 *a1, _QWORD *a2)
{
  __int64 v2; // rax

  v2 = *a1;
  if ( (*a1 & 1) != 0 || (v2 & 0xFFFFFFFFFFFFFFFELL) != 0 )
    sub_16BCAE0(a1);
  *a1 = *a2 | v2 | 1;
  *a2 = 0;
  return a1;
}
