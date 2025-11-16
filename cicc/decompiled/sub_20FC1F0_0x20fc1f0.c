// Function: sub_20FC1F0
// Address: 0x20fc1f0
//
_QWORD *__fastcall sub_20FC1F0(__int64 a1)
{
  _QWORD *v1; // r8
  __int64 v3; // rax
  __int64 v4; // rdx
  __int64 v5; // r14
  unsigned __int64 v6; // r13
  unsigned int v7; // ecx
  __int64 v8; // rax
  int v9; // r8d
  int v10; // r9d
  __int64 v11; // [rsp+8h] [rbp-28h]

  v1 = *(_QWORD **)a1;
  if ( *(_QWORD *)a1 )
  {
    *(_QWORD *)a1 = *v1;
LABEL_3:
    memset(v1, 0, 0xC0u);
    return v1;
  }
  v3 = *(_QWORD *)(a1 + 8);
  v4 = *(_QWORD *)(a1 + 16);
  *(_QWORD *)(a1 + 88) += 192LL;
  if ( ((v3 + 63) & 0xFFFFFFFFFFFFFFC0LL) - v3 + 192 <= v4 - v3 )
  {
    v1 = (_QWORD *)((v3 + 63) & 0xFFFFFFFFFFFFFFC0LL);
    *(_QWORD *)(a1 + 8) = v1 + 24;
  }
  else
  {
    v5 = *(unsigned int *)(a1 + 32);
    v6 = 0x40000000000LL;
    v7 = *(_DWORD *)(a1 + 32) >> 7;
    if ( v7 < 0x1E )
      v6 = 4096LL << v7;
    v8 = malloc(v6);
    if ( !v8 )
    {
      sub_16BD1C0("Allocation failed", 1u);
      v5 = *(unsigned int *)(a1 + 32);
      v8 = 0;
    }
    if ( *(_DWORD *)(a1 + 36) <= (unsigned int)v5 )
    {
      v11 = v8;
      sub_16CD150(a1 + 24, (const void *)(a1 + 40), 0, 8, v9, v10);
      v5 = *(unsigned int *)(a1 + 32);
      v8 = v11;
    }
    v1 = (_QWORD *)((v8 + 63) & 0xFFFFFFFFFFFFFFC0LL);
    *(_QWORD *)(*(_QWORD *)(a1 + 24) + 8 * v5) = v8;
    ++*(_DWORD *)(a1 + 32);
    *(_QWORD *)(a1 + 16) = v8 + v6;
    *(_QWORD *)(a1 + 8) = v1 + 24;
  }
  if ( v1 )
    goto LABEL_3;
  return 0;
}
