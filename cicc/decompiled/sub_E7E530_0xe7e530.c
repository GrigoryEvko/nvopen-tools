// Function: sub_E7E530
// Address: 0xe7e530
//
__int64 __fastcall sub_E7E530(__int64 a1)
{
  __int64 v1; // r12

  v1 = *(_QWORD *)(*(_QWORD *)(a1 + 288) + 8LL);
  if ( !*(_DWORD *)(*(_QWORD *)(a1 + 296) + 368LL) )
    sub_C64ED0(".bundle_unlock forbidden when bundling is disabled", 1u);
  if ( !sub_E7E4B0(a1) )
    sub_C64ED0(".bundle_unlock without matching lock", 1u);
  if ( (*(_BYTE *)(v1 + 48) & 1) != 0 )
    sub_C64ED0("Empty bundle-locked group is forbidden", 1u);
  return sub_E92900(v1, 0);
}
