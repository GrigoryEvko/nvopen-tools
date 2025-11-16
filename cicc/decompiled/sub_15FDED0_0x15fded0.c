// Function: sub_15FDED0
// Address: 0x15fded0
//
__int64 __fastcall sub_15FDED0(_QWORD *a1, __int64 a2, __int64 a3, __int64 a4)
{
  int v6; // r15d
  int v7; // eax
  int v10; // edi

  v6 = sub_16431D0(*a1);
  v7 = sub_16431D0(a2);
  v10 = 47;
  if ( v6 != v7 )
    v10 = 38;
  return sub_15FDBD0(v10, (__int64)a1, a2, a3, a4);
}
