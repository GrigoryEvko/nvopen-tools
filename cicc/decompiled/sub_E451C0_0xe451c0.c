// Function: sub_E451C0
// Address: 0xe451c0
//
__int64 __fastcall sub_E451C0(__int64 a1, __int64 a2)
{
  __int64 v2; // rdx

  v2 = *(_QWORD *)(a2 + 32);
  if ( (unsigned __int64)(*(_QWORD *)(a2 + 24) - v2) <= 8 )
    return sub_CB6200(a2, "<nullptr>", 9u);
  *(_BYTE *)(v2 + 8) = 62;
  *(_QWORD *)v2 = 0x7274706C6C756E3CLL;
  *(_QWORD *)(a2 + 32) += 9LL;
  return 0x7274706C6C756E3CLL;
}
