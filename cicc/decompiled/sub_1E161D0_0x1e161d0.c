// Function: sub_1E161D0
// Address: 0x1e161d0
//
_QWORD *__fastcall sub_1E161D0(_QWORD *a1)
{
  unsigned __int64 *v1; // rcx
  unsigned __int64 v2; // rdx

  sub_1DD5BC0(a1[3] + 16LL, (__int64)a1);
  v1 = (unsigned __int64 *)a1[1];
  v2 = *a1 & 0xFFFFFFFFFFFFFFF8LL;
  *v1 = v2 | *v1 & 7;
  *(_QWORD *)(v2 + 8) = v1;
  a1[1] = 0;
  *a1 &= 7uLL;
  return a1;
}
