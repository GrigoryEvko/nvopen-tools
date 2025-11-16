// Function: sub_39CAC10
// Address: 0x39cac10
//
__int64 __fastcall sub_39CAC10(__int64 *a1, __int64 a2, __int64 a3, __int64 *a4)
{
  __int64 v5; // r13

  v5 = sub_39CABF0(a1, a2, *(_BYTE *)(a3 + 24));
  if ( (*(_BYTE *)(*(_QWORD *)a2 + 37LL) & 4) == 0 && (*(_BYTE *)(sub_3988770(a2) + 29) & 4) == 0 )
    return v5;
  *a4 = v5;
  return v5;
}
