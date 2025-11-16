// Function: sub_385DAA0
// Address: 0x385daa0
//
__int64 __fastcall sub_385DAA0(__int64 a1, unsigned int a2)
{
  unsigned __int64 v2; // rax
  __int64 v3; // r14
  __int64 v4; // r15
  __int64 v5; // rax
  __int64 v6; // r13
  __int64 v7; // rax
  int v8; // r8d
  int v9; // r9d
  __int64 v10; // rax

  v2 = *(_QWORD *)(*(_QWORD *)a1 + 8LL) + ((unsigned __int64)a2 << 6);
  v3 = *(_QWORD *)(v2 + 24);
  v4 = *(_QWORD *)(v2 + 32);
  v5 = sub_385B960(v3, *(_QWORD *)(a1 + 16), *(_QWORD *)(*(_QWORD *)a1 + 264LL));
  if ( !v5 )
    return 0;
  v6 = v5;
  v7 = sub_385B960(v4, *(_QWORD *)(a1 + 8), *(_QWORD *)(*(_QWORD *)a1 + 264LL));
  if ( !v7 )
    return 0;
  if ( v3 == v6 )
  {
    *(_QWORD *)(a1 + 16) = v3;
    if ( v4 == v7 )
      goto LABEL_6;
    goto LABEL_5;
  }
  if ( v4 != v7 )
LABEL_5:
    *(_QWORD *)(a1 + 8) = v4;
LABEL_6:
  v10 = *(unsigned int *)(a1 + 32);
  if ( (unsigned int)v10 >= *(_DWORD *)(a1 + 36) )
  {
    sub_16CD150(a1 + 24, (const void *)(a1 + 40), 0, 4, v8, v9);
    v10 = *(unsigned int *)(a1 + 32);
  }
  *(_DWORD *)(*(_QWORD *)(a1 + 24) + 4 * v10) = a2;
  ++*(_DWORD *)(a1 + 32);
  return 1;
}
