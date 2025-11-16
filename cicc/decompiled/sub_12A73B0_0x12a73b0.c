// Function: sub_12A73B0
// Address: 0x12a73b0
//
__int64 __fastcall sub_12A73B0(__int64 a1, __int64 a2, __int64 a3, __int64 *a4)
{
  __int64 v5; // rcx
  __int64 v6; // rax

  if ( a3 )
  {
    sub_157E9D0(a3 + 40, a1);
    v5 = *a4;
    v6 = *(_QWORD *)(a1 + 24);
    *(_QWORD *)(a1 + 32) = a4;
    v5 &= 0xFFFFFFFFFFFFFFF8LL;
    *(_QWORD *)(a1 + 24) = v5 | v6 & 7;
    *(_QWORD *)(v5 + 8) = a1 + 24;
    *a4 = *a4 & 7 | (a1 + 24);
  }
  return sub_164B780(a1, a2);
}
