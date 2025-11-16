// Function: sub_1E0A860
// Address: 0x1e0a860
//
__int64 __fastcall sub_1E0A860(__int64 a1, __int64 a2, __int64 a3)
{
  unsigned __int64 v3; // r12
  __int64 i; // rbx

  v3 = 0xAAAAAAAAAAAAAAABLL * ((__int64)(*(_QWORD *)(a1 + 16) - *(_QWORD *)(a1 + 8)) >> 3);
  if ( v3 )
  {
    for ( i = 0; i != v3; ++i )
      sub_1E0A7F0(a1, i, a2, a3);
  }
  return 0;
}
