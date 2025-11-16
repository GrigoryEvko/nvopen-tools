// Function: sub_B52350
// Address: 0xb52350
//
__int64 __fastcall sub_B52350(__int64 a1, __int64 a2, __int64 a3, __int64 a4, unsigned __int16 a5)
{
  unsigned int v8; // r15d
  unsigned int v9; // eax
  int v10; // edi

  v8 = sub_BCB060(*(_QWORD *)(a1 + 8));
  v9 = sub_BCB060(a2);
  v10 = 49;
  if ( v8 != v9 )
    v10 = (v8 <= v9) + 45;
  return sub_B51D30(v10, a1, a2, a3, a4, a5);
}
