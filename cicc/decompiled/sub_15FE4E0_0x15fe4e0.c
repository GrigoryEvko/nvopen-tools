// Function: sub_15FE4E0
// Address: 0x15fe4e0
//
__int64 __fastcall sub_15FE4E0(_QWORD *a1, __int64 a2, char a3, __int64 a4, __int64 a5)
{
  unsigned int v8; // r15d
  unsigned int v9; // eax
  int v10; // edi

  v8 = sub_16431D0(*a1);
  v9 = sub_16431D0(a2);
  v10 = 47;
  if ( v8 != v9 )
  {
    v10 = 36;
    if ( v8 <= v9 )
      v10 = 37 - ((a3 == 0) - 1);
  }
  return sub_15FE240(v10, (__int64)a1, a2, a4, a5);
}
