// Function: sub_B44590
// Address: 0xb44590
//
__int64 __fastcall sub_B44590(__int64 a1, _QWORD *a2)
{
  unsigned __int64 *v2; // rcx
  unsigned __int64 v3; // rdx

  v2 = (unsigned __int64 *)a2[1];
  v3 = *a2 & 0xFFFFFFFFFFFFFFF8LL;
  *v2 = v3 | *v2 & 7;
  *(_QWORD *)(v3 + 8) = v2;
  *a2 &= 7uLL;
  a2[1] = 0;
  return sub_B12320((__int64)a2);
}
