// Function: sub_9C6670
// Address: 0x9c6670
//
__int64 *__fastcall sub_9C6670(__int64 *a1, _QWORD *a2)
{
  __int64 v2; // rax

  v2 = *a1;
  if ( (*a1 & 1) != 0 || (v2 & 0xFFFFFFFFFFFFFFFELL) != 0 )
    sub_C63C30(a1);
  *a1 = *a2 | v2 | 1;
  *a2 = 0;
  return a1;
}
