// Function: sub_D136E0
// Address: 0xd136e0
//
__int64 __fastcall sub_D136E0(__int64 a1, __int64 a2)
{
  _BYTE *v2; // rdx
  __int64 v3; // rsi

  v2 = *(_BYTE **)(a2 + 24);
  if ( *v2 == 30 && !*(_BYTE *)(a1 + 24) )
    return 2;
  v3 = *(_QWORD *)(a1 + 8);
  if ( v3 )
    *(_QWORD *)(a1 + 8) = sub_B1A110(*(_QWORD *)(a1 + 16), v3, (__int64)v2);
  else
    *(_QWORD *)(a1 + 8) = v2;
  *(_BYTE *)(a1 + 25) = 1;
  return 2;
}
