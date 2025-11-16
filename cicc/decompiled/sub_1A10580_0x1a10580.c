// Function: sub_1A10580
// Address: 0x1a10580
//
__int64 __fastcall sub_1A10580(__int64 a1, __int64 *a2, __int64 a3, __int64 a4, __int64 a5, int a6)
{
  unsigned int v6; // r8d
  __int64 v8; // rdx
  __int64 v9; // rax
  unsigned __int64 v10; // rcx
  __int64 v11; // rax
  __int64 v13; // rax

  v6 = 0;
  v8 = *a2;
  v9 = (*a2 >> 1) & 3;
  if ( v9 == 3 || ((a4 >> 1) & 3) == 0 )
    return v6;
  if ( ((a4 >> 1) & 3) == 3 )
    goto LABEL_6;
  v10 = a4 & 0xFFFFFFFFFFFFFFF8LL;
  if ( v9 )
  {
    if ( v10 == (v8 & 0xFFFFFFFFFFFFFFF8LL) )
      return v6;
LABEL_6:
    *a2 = v8 | 6;
    v11 = *(unsigned int *)(a1 + 824);
    if ( (unsigned int)v11 >= *(_DWORD *)(a1 + 828) )
    {
      sub_16CD150(a1 + 816, (const void *)(a1 + 832), 0, 8, 0, a6);
      v11 = *(unsigned int *)(a1 + 824);
    }
    v6 = 1;
    *(_QWORD *)(*(_QWORD *)(a1 + 816) + 8 * v11) = a3;
    ++*(_DWORD *)(a1 + 824);
    return v6;
  }
  *a2 = *a2 & 1 | v10 | 2;
  v13 = *(unsigned int *)(a1 + 1352);
  if ( (unsigned int)v13 >= *(_DWORD *)(a1 + 1356) )
  {
    sub_16CD150(a1 + 1344, (const void *)(a1 + 1360), 0, 8, 0, a6);
    v13 = *(unsigned int *)(a1 + 1352);
  }
  v6 = 1;
  *(_QWORD *)(*(_QWORD *)(a1 + 1344) + 8 * v13) = a3;
  ++*(_DWORD *)(a1 + 1352);
  return v6;
}
