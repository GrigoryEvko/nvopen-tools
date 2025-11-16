// Function: sub_6F7220
// Address: 0x6f7220
//
__int64 __fastcall sub_6F7220(__m128i *a1, __int64 a2)
{
  __int64 v2; // rdx
  __int64 v3; // rcx
  __int64 v4; // r8
  __int64 v5; // r9
  __int64 v6; // rax
  __int64 *v7; // rax

  sub_6F6BD0(a1->m128i_i64, 0);
  v6 = sub_6F7180(a1, 0, v2, v3, v4, v5);
  v7 = (__int64 *)sub_73DBF0(5, a2, v6);
  sub_6E70E0(v7, (__int64)a1);
  return sub_6E26D0(1, (__int64)a1);
}
