// Function: sub_B970B0
// Address: 0xb970b0
//
void __fastcall sub_B970B0(__int64 a1, unsigned int a2, __int64 a3)
{
  __int64 *v4; // r13

  v4 = (__int64 *)(**(_QWORD **)(a1 + 56) + 8LL * a2);
  if ( *v4 )
    sub_B91220((__int64)v4, *v4);
  *v4 = a3;
  if ( a3 )
    sub_B96E90((__int64)v4, a3, 1);
}
