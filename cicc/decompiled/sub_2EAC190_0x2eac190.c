// Function: sub_2EAC190
// Address: 0x2eac190
//
__int64 __fastcall sub_2EAC190(__int64 a1, int a2)
{
  _QWORD *v3; // rdx

  if ( a2 != -1 )
    return sub_CB59F0(a1, a2);
  v3 = *(_QWORD **)(a1 + 32);
  if ( *(_QWORD *)(a1 + 24) - (_QWORD)v3 <= 7u )
    return sub_CB6200(a1, "<badref>", 8u);
  *v3 = 0x3E6665726461623CLL;
  *(_QWORD *)(a1 + 32) += 8LL;
  return 0x3E6665726461623CLL;
}
