// Function: sub_2B2CF10
// Address: 0x2b2cf10
//
__int64 __fastcall sub_2B2CF10(__int64 *a1, __int64 a2)
{
  __int64 v2; // rax
  unsigned int v4; // r10d
  __int64 v5; // rdi
  __int64 result; // rax
  __int64 v7; // rax
  bool v8; // of

  v2 = *a1;
  if ( *(_DWORD *)(*a1 + 104) == 2 )
  {
    sub_2B08900(a1[5], *((unsigned int *)a1 + 12));
    v7 = sub_DFD5B0(*(__int64 **)(a1[23] + 3296), 33, a1[24], *(_QWORD *)(a1[25] - 32), 0);
  }
  else
  {
    v4 = *(_DWORD *)(v2 + 432);
    v5 = a1[23];
    if ( v4 )
      return sub_DFD610(*(__int64 **)(v5 + 3296), 33, a1[24], v4, 0);
    sub_2B2BBE0(v5, **(char ***)(v2 + 240), *(unsigned int *)(*(_QWORD *)(v2 + 240) + 8LL));
    v7 = sub_DFD4A0(*(__int64 **)(a1[23] + 3296));
  }
  v8 = __OFADD__(a2, v7);
  result = a2 + v7;
  if ( v8 )
  {
    result = 0x8000000000000000LL;
    if ( a2 > 0 )
      return 0x7FFFFFFFFFFFFFFFLL;
  }
  return result;
}
