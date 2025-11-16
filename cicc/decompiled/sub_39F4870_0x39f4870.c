// Function: sub_39F4870
// Address: 0x39f4870
//
void __fastcall sub_39F4870(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v5; // rsi
  __int64 v6; // r14

  v5 = *(_QWORD *)(a2 + 176);
  v6 = *(_QWORD *)(a1 + 264);
  if ( v5 )
    sub_390D5F0(*(_QWORD *)(a1 + 264), v5, 0);
  sub_38D59C0(a1, a2, a3);
  sub_390D5F0(v6, *(_QWORD *)(a2 + 8), 0);
}
