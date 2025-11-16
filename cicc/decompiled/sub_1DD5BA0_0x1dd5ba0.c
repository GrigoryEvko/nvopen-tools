// Function: sub_1DD5BA0
// Address: 0x1dd5ba0
//
__int64 __fastcall sub_1DD5BA0(__int64 *a1, __int64 a2)
{
  __int64 v2; // rax

  v2 = *a1;
  *(_QWORD *)(a2 + 24) = *a1;
  return sub_1E15C30(a2, *(_QWORD *)(*(_QWORD *)(v2 + 56) + 40LL));
}
