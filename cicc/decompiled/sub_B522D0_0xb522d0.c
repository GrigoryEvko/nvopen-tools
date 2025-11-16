// Function: sub_B522D0
// Address: 0xb522d0
//
__int64 __fastcall sub_B522D0(__int64 a1, __int64 a2, char a3, __int64 a4, __int64 a5, unsigned __int16 a6)
{
  unsigned int v9; // ebx
  unsigned int v10; // eax
  int v11; // edi

  v9 = sub_BCB060(*(_QWORD *)(a1 + 8));
  v10 = sub_BCB060(a2);
  v11 = 49;
  if ( v9 != v10 )
  {
    v11 = 38;
    if ( v9 <= v10 )
      v11 = 39 - ((a3 == 0) - 1);
  }
  return sub_B51D30(v11, a1, a2, a4, a5, a6);
}
