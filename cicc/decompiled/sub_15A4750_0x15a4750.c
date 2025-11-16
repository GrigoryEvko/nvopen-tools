// Function: sub_15A4750
// Address: 0x15a4750
//
__int64 __fastcall sub_15A4750(__int64 ***a1, __int64 **a2, char a3)
{
  unsigned int v5; // ebx
  unsigned int v6; // eax
  int v7; // edi

  v5 = sub_16431D0(*a1);
  v6 = sub_16431D0(a2);
  v7 = 47;
  if ( v5 != v6 )
  {
    v7 = 36;
    if ( v5 <= v6 )
      v7 = 37 - ((a3 == 0) - 1);
  }
  return sub_15A46C0(v7, a1, a2, 0);
}
