// Function: sub_B8C820
// Address: 0xb8c820
//
__int64 __fastcall sub_B8C820(__int64 *a1, __int64 a2, __int64 a3)
{
  __int64 v4; // r13
  __int64 v5; // r15
  __int64 v6; // rax
  __int64 v7; // rcx

  v4 = sub_BCCE00(*a1, *(unsigned int *)(a2 + 8));
  v5 = sub_AD8D80(v4, a3);
  v6 = sub_AD8D80(v4, a2);
  return sub_B8C7C0(a1, v6, v5, v7);
}
