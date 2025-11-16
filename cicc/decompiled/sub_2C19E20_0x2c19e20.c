// Function: sub_2C19E20
// Address: 0x2c19e20
//
__int64 __fastcall sub_2C19E20(_QWORD *a1)
{
  __int64 *v1; // rcx
  unsigned __int64 v2; // rdx
  __int64 result; // rax

  v1 = (__int64 *)a1[4];
  v2 = a1[3] & 0xFFFFFFFFFFFFFFF8LL;
  result = v2 | *v1 & 7;
  *v1 = result;
  *(_QWORD *)(v2 + 8) = v1;
  a1[4] = 0;
  a1[3] &= 7uLL;
  a1[10] = 0;
  return result;
}
