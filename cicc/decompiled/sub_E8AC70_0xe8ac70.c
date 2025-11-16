// Function: sub_E8AC70
// Address: 0xe8ac70
//
__int64 __fastcall sub_E8AC70(__int64 *a1)
{
  unsigned __int64 v1; // rsi
  __int64 v2; // rdx
  __int64 v3; // rcx
  __int64 v4; // r8
  __int64 v5; // r9

  sub_E655A0(a1[1]);
  if ( *(_BYTE *)(a1[1] + 1793) )
    sub_E7A190(a1);
  v1 = *(unsigned __int16 *)(a1[37] + 72) | ((unsigned __int64)*(unsigned __int8 *)(a1[37] + 74) << 16);
  sub_E77660(a1, v1);
  sub_E918E0(a1);
  sub_E8A8A0((__int64)a1, v1, v2, v3, v4, v5);
  return sub_E5F540(a1[37]);
}
