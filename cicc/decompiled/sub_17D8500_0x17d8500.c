// Function: sub_17D8500
// Address: 0x17d8500
//
unsigned __int64 __fastcall sub_17D8500(__int128 a1)
{
  __int64 v1; // r12
  __int64 v2; // rax
  _QWORD *v3; // rax
  __int64 *v4; // rax
  __int64 v5; // rdx
  __int64 v6; // rcx
  const char **v7; // rax
  __int64 v8; // rax
  unsigned __int64 result; // rax
  __int64 v10[14]; // [rsp+0h] [rbp-70h] BYREF

  v1 = *((_QWORD *)&a1 + 1);
  sub_17CE510((__int64)v10, *((__int64 *)&a1 + 1), 0, 0, 0);
  if ( (*(_BYTE *)(*((_QWORD *)&a1 + 1) + 23LL) & 0x40) != 0 )
    v2 = *(_QWORD *)(*((_QWORD *)&a1 + 1) - 8LL);
  else
    v2 = *((_QWORD *)&a1 + 1) - 24LL * (*(_DWORD *)(*((_QWORD *)&a1 + 1) + 20LL) & 0xFFFFFFF);
  *((_QWORD *)&a1 + 1) = *(_QWORD *)(v2 + 24);
  sub_17D5820(a1, v1);
  if ( (*(_BYTE *)(v1 + 23) & 0x40) != 0 )
    v3 = *(_QWORD **)(v1 - 8);
  else
    v3 = (_QWORD *)(v1 - 24LL * (*(_DWORD *)(v1 + 20) & 0xFFFFFFF));
  *((_QWORD *)&a1 + 1) = *v3;
  v4 = sub_17D4DA0(a1);
  sub_17D4920(a1, (__int64 *)v1, (__int64)v4);
  if ( (*(_BYTE *)(v1 + 23) & 0x40) != 0 )
  {
    v7 = *(const char ***)(v1 - 8);
  }
  else
  {
    v5 = 24LL * (*(_DWORD *)(v1 + 20) & 0xFFFFFFF);
    v7 = (const char **)(v1 - v5);
  }
  v8 = sub_17D4880(a1, *v7, v5, v6);
  result = sub_17D4B80(a1, v1, v8);
  if ( v10[0] )
    return sub_161E7C0((__int64)v10, v10[0]);
  return result;
}
