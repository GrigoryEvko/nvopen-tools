// Function: sub_2B15170
// Address: 0x2b15170
//
unsigned __int64 __fastcall sub_2B15170(__int64 **a1, __int64 a2, int a3)
{
  __int64 v5; // rdx
  unsigned int v6; // eax
  unsigned int v7; // r11d
  __int64 *v8; // rdi
  __int64 v9; // rax
  bool v10; // of
  unsigned __int64 result; // rax
  __int64 v12; // r10
  __int64 v13; // r10

  v5 = **a1;
  v6 = *(_DWORD *)(v5 + 104);
  if ( v6 == 1 )
  {
    sub_2B08900(a1[5][4], *((unsigned int *)a1[5] + 10));
    v9 = sub_DFD550(
           *(__int64 **)(*(_QWORD *)(v13 + 8) + 3296LL),
           32,
           **(_QWORD **)(v13 + 16),
           *(_QWORD *)(**(_QWORD **)(v13 + 24) - 32LL),
           0);
  }
  else if ( v6 > 1 )
  {
    if ( v6 == 2 )
    {
      sub_2B08900(a1[5][4], *((unsigned int *)a1[5] + 10));
      v9 = sub_DFD5B0(
             *(__int64 **)(*(_QWORD *)(v12 + 8) + 3296LL),
             32,
             **(_QWORD **)(v12 + 16),
             *(_QWORD *)(**(_QWORD **)(v12 + 24) - 32LL),
             0);
    }
    else
    {
      if ( v6 - 3 <= 2 )
        BUG();
      if ( a3 == 1 )
        return a2;
      v9 = 0;
    }
  }
  else
  {
    v7 = *(_DWORD *)(v5 + 432);
    v8 = (__int64 *)a1[1][412];
    if ( v7 )
      v9 = sub_DFD610(v8, 32, *a1[2], v7, 0);
    else
      v9 = sub_DFD4A0(v8);
  }
  v10 = __OFADD__(a2, v9);
  result = a2 + v9;
  if ( v10 )
  {
    result = 0x7FFFFFFFFFFFFFFFLL;
    if ( a2 <= 0 )
      return 0x8000000000000000LL;
  }
  return result;
}
