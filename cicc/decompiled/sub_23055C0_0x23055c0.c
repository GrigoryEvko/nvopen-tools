// Function: sub_23055C0
// Address: 0x23055c0
//
__int64 __fastcall sub_23055C0(__int64 *a1, _QWORD *a2)
{
  __int64 v2; // rax
  __int64 result; // rax

  v2 = *a1;
  if ( (*a1 & 1) != 0 || (v2 & 0xFFFFFFFFFFFFFFFELL) != 0 )
    sub_C63C30(a1, (__int64)a2);
  result = *a2 | v2 | 1;
  *a1 = result;
  *a2 = 0;
  return result;
}
