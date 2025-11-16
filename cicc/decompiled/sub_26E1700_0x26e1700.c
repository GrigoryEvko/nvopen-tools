// Function: sub_26E1700
// Address: 0x26e1700
//
void __fastcall sub_26E1700(__int64 a1, __int64 a2, __int64 a3, unsigned __int64 *a4, unsigned __int64 *a5)
{
  __int64 i; // r12
  __int64 v9; // r12
  __int64 j; // rbx

  for ( i = *(_QWORD *)(a2 + 24); a2 + 8 != i; i = sub_220EF30(i) )
  {
    if ( *(_QWORD *)(i + 40) && *(_QWORD *)(i + 48) )
      sub_26E16C0(a4, i + 32);
  }
  v9 = *(_QWORD *)(a3 + 24);
  for ( j = a3 + 8; j != v9; v9 = sub_220EF30(v9) )
    sub_26E16C0(a5, v9 + 32);
}
