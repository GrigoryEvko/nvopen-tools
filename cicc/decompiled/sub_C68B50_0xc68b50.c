// Function: sub_C68B50
// Address: 0xc68b50
//
__int64 __fastcall sub_C68B50(__int64 a1, __int64 a2)
{
  __int64 v3; // rdx

  if ( !*(_DWORD *)(a1 + 8) )
    return sub_CB59F0(a2, *(_QWORD *)a1);
  v3 = *(_QWORD *)(a2 + 32);
  if ( (unsigned __int64)(*(_QWORD *)(a2 + 24) - v3) <= 6 )
    return sub_CB6200(a2, "Invalid", 7);
  *(_DWORD *)v3 = 1635151433;
  *(_WORD *)(v3 + 4) = 26988;
  *(_BYTE *)(v3 + 6) = 100;
  *(_QWORD *)(a2 + 32) += 7LL;
  return 26988;
}
