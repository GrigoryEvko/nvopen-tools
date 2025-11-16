// Function: sub_1DD6DB0
// Address: 0x1dd6db0
//
__int64 __fastcall sub_1DD6DB0(__int64 a1, __int64 a2)
{
  unsigned __int64 *v2; // rcx
  unsigned __int64 v3; // rdx

  sub_1DD4CE0(a2);
  *(_WORD *)(a2 + 46) &= 0xFFF3u;
  sub_1DD5BC0(a1 + 16, a2);
  v2 = *(unsigned __int64 **)(a2 + 8);
  v3 = *(_QWORD *)a2 & 0xFFFFFFFFFFFFFFF8LL;
  *v2 = v3 | *v2 & 7;
  *(_QWORD *)(v3 + 8) = v2;
  *(_QWORD *)a2 &= 7uLL;
  *(_QWORD *)(a2 + 8) = 0;
  return a2;
}
