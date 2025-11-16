// Function: sub_B520B0
// Address: 0xb520b0
//
__int64 __fastcall sub_B520B0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, unsigned __int16 a5)
{
  int v8; // r15d
  int v9; // edi

  v8 = sub_BCB060(*(_QWORD *)(a1 + 8));
  v9 = 49;
  if ( v8 != (unsigned int)sub_BCB060(a2) )
    v9 = 39;
  return sub_B51D30(v9, a1, a2, a3, a4, a5);
}
