// Function: sub_1444C30
// Address: 0x1444c30
//
_QWORD *__fastcall sub_1444C30(__int64 *a1)
{
  __int64 *v2; // rbx
  __int64 *v3; // r13
  __int64 v4; // rdi

  v2 = (__int64 *)a1[5];
  v3 = (__int64 *)a1[6];
  while ( v3 != v2 )
  {
    v4 = *v2++;
    sub_1444C30(v4);
  }
  return sub_1444BB0(a1);
}
