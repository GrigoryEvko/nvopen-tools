// Function: sub_E7E4D0
// Address: 0xe7e4d0
//
__int64 __fastcall sub_E7E4D0(__int64 a1, char a2)
{
  __int64 v2; // r12

  v2 = *(_QWORD *)(*(_QWORD *)(a1 + 288) + 8LL);
  if ( !*(_DWORD *)(*(_QWORD *)(a1 + 296) + 368LL) )
    sub_C64ED0(".bundle_lock forbidden when bundling is disabled", 1u);
  if ( !sub_E7E4B0(a1) )
    *(_BYTE *)(v2 + 48) |= 1u;
  return sub_E92900(v2, 1 - ((unsigned int)(a2 == 0) - 1));
}
