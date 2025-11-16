// Function: sub_1E61F30
// Address: 0x1e61f30
//
__int64 __fastcall sub_1E61F30(__int64 a1, __int64 a2)
{
  __int64 v2; // rdi

  v2 = *(_QWORD *)(a1 + 232);
  *(_QWORD *)(v2 + 88) = a2;
  sub_1E61C40(v2, 0);
  return 0;
}
