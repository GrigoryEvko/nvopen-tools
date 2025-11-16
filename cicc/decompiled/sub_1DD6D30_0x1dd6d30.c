// Function: sub_1DD6D30
// Address: 0x1dd6d30
//
__int64 __fastcall sub_1DD6D30(__int64 a1, _QWORD *a2)
{
  __int64 v2; // r14
  unsigned __int64 *v3; // rcx
  unsigned __int64 v4; // rdx

  sub_1DD4CE0((__int64)a2);
  v2 = a2[1];
  sub_1DD5BC0(a1 + 16, (__int64)a2);
  v3 = (unsigned __int64 *)a2[1];
  v4 = *a2 & 0xFFFFFFFFFFFFFFF8LL;
  *v3 = v4 | *v3 & 7;
  *(_QWORD *)(v4 + 8) = v3;
  *a2 &= 7uLL;
  a2[1] = 0;
  sub_1DD5C20(a1 + 16);
  return v2;
}
