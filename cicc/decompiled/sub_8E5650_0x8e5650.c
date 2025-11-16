// Function: sub_8E5650
// Address: 0x8e5650
//
__int64 __fastcall sub_8E5650(__int128 a1)
{
  unsigned __int64 v1; // rax
  __int64 v2; // rcx
  __int64 v3; // r8
  __int64 v4; // r9

  v1 = sub_5EB9D0(*((__int64 *)&a1 + 1), a1);
  *((_QWORD *)&a1 + 1) = *(_QWORD *)(*((_QWORD *)&a1 + 1) + 56LL);
  return sub_8E5310(a1, v1, v2, v3, v4);
}
