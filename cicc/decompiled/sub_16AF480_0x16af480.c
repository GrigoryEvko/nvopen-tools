// Function: sub_16AF480
// Address: 0x16af480
//
unsigned __int64 __fastcall sub_16AF480(__int64 *a1)
{
  __int64 v1; // rax

  v1 = *((unsigned int *)a1 + 2);
  if ( (unsigned int)v1 > 0x40 )
    return sub_16AF040((_QWORD *)*a1, *a1 + 8 * ((unsigned __int64)(v1 + 63) >> 6));
  else
    return sub_16AEFA0(a1);
}
