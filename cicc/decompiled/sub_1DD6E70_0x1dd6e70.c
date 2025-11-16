// Function: sub_1DD6E70
// Address: 0x1dd6e70
//
__int64 __fastcall sub_1DD6E70(_QWORD *a1)
{
  __int64 v1; // r13
  unsigned __int64 *v2; // rcx
  unsigned __int64 v3; // rdx

  v1 = a1[7] + 320LL;
  sub_1DD5B80(v1, (__int64)a1);
  v2 = (unsigned __int64 *)a1[1];
  v3 = *a1 & 0xFFFFFFFFFFFFFFF8LL;
  *v2 = v3 | *v2 & 7;
  *(_QWORD *)(v3 + 8) = v2;
  *a1 &= 7uLL;
  a1[1] = 0;
  return sub_1E0A230(v1, a1);
}
