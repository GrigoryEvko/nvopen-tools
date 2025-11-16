// Function: sub_1E32200
// Address: 0x1e32200
//
__int64 __fastcall sub_1E32200(__int64 a1, int a2)
{
  _QWORD *v3; // rdx

  if ( a2 != -1 )
    return sub_16E7AB0(a1, a2);
  v3 = *(_QWORD **)(a1 + 24);
  if ( *(_QWORD *)(a1 + 16) - (_QWORD)v3 <= 7u )
    return sub_16E7EE0(a1, "<badref>", 8u);
  *v3 = 0x3E6665726461623CLL;
  *(_QWORD *)(a1 + 24) += 8LL;
  return 0x3E6665726461623CLL;
}
