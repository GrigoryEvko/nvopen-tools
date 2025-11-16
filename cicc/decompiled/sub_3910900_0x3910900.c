// Function: sub_3910900
// Address: 0x3910900
//
__int64 __fastcall sub_3910900(__int64 a1, unsigned __int64 a2, unsigned __int64 a3)
{
  __int64 v4; // rcx

  if ( a3 <= a2 )
    return 0;
  v4 = *(_QWORD *)(a1 + 264);
  if ( a2 < 0xAAAAAAAAAAAAAAABLL * ((*(_QWORD *)(a1 + 272) - v4) >> 3) )
    return v4 + 24 * a2;
  else
    return 0;
}
