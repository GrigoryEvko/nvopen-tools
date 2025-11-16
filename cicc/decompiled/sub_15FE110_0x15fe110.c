// Function: sub_15FE110
// Address: 0x15fe110
//
__int64 __fastcall sub_15FE110(_QWORD *a1, __int64 a2, __int64 a3, __int64 a4)
{
  unsigned int v7; // r15d
  unsigned int v8; // eax
  int v9; // edi

  v7 = sub_16431D0(*a1);
  v8 = sub_16431D0(a2);
  v9 = 47;
  if ( v7 != v8 )
    v9 = (v7 <= v8) + 43;
  return sub_15FDBD0(v9, (__int64)a1, a2, a3, a4);
}
