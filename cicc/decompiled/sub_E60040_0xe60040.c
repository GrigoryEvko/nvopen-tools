// Function: sub_E60040
// Address: 0xe60040
//
__int64 __fastcall sub_E60040(__int64 a1, unsigned __int64 a2, unsigned __int64 a3)
{
  __int64 v4; // rcx

  if ( a3 <= a2 )
    return 0;
  v4 = *(_QWORD *)(a1 + 256);
  if ( a2 < 0xAAAAAAAAAAAAAAABLL * ((*(_QWORD *)(a1 + 264) - v4) >> 3) )
    return v4 + 24 * a2;
  else
    return 0;
}
