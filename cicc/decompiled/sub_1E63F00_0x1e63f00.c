// Function: sub_1E63F00
// Address: 0x1e63f00
//
_QWORD *__fastcall sub_1E63F00(__int64 *a1)
{
  __int64 *v2; // rbx
  __int64 *v3; // r13
  __int64 v4; // rdi

  v2 = (__int64 *)a1[5];
  v3 = (__int64 *)a1[6];
  while ( v3 != v2 )
  {
    v4 = *v2++;
    sub_1E63F00(v4);
  }
  return sub_1E63E80(a1);
}
