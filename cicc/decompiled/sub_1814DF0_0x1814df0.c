// Function: sub_1814DF0
// Address: 0x1814df0
//
__int64 __fastcall sub_1814DF0(__int64 a1)
{
  __int64 v1; // r13

  *(_QWORD *)a1 = off_49F0998;
  sub_1814D10(*(_QWORD **)(a1 + 456));
  j___libc_free_0(*(_QWORD *)(a1 + 408));
  v1 = *(_QWORD *)(a1 + 392);
  if ( v1 )
  {
    sub_39479B0(*(_QWORD *)(a1 + 392));
    j_j___libc_free_0(v1, 24);
  }
  sub_1636790((_QWORD *)a1);
  return j_j___libc_free_0(a1, 536);
}
