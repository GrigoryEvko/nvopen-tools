// Function: sub_1DD6E10
// Address: 0x1dd6e10
//
__int64 __fastcall sub_1DD6E10(__int64 a1, __int64 *a2, __int64 a3)
{
  __int64 v4; // rdx
  __int64 v5; // rax

  if ( a2 != (__int64 *)(a1 + 24) && (*((_BYTE *)a2 + 46) & 4) != 0 )
    *(_WORD *)(a3 + 46) |= 0xCu;
  sub_1DD5BA0((__int64 *)(a1 + 16), a3);
  v4 = *a2;
  v5 = *(_QWORD *)a3;
  *(_QWORD *)(a3 + 8) = a2;
  v4 &= 0xFFFFFFFFFFFFFFF8LL;
  *(_QWORD *)a3 = v4 | v5 & 7;
  *(_QWORD *)(v4 + 8) = a3;
  *a2 = a3 | *a2 & 7;
  return a3;
}
