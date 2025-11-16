// Function: sub_1DA89E0
// Address: 0x1da89e0
//
_QWORD *__fastcall sub_1DA89E0(__int64 a1)
{
  _QWORD *v1; // r8
  _DWORD *v2; // rax
  __int64 v4; // rax
  __int64 v5; // rdx
  __int64 v6; // r14
  unsigned __int64 v7; // r13
  unsigned int v8; // ecx
  __int64 v9; // rax
  int v10; // r8d
  int v11; // r9d
  __int64 v12; // [rsp+8h] [rbp-28h]

  v1 = *(_QWORD **)a1;
  if ( *(_QWORD *)a1 )
  {
    *(_QWORD *)a1 = *v1;
  }
  else
  {
    v4 = *(_QWORD *)(a1 + 8);
    v5 = *(_QWORD *)(a1 + 16);
    *(_QWORD *)(a1 + 88) += 192LL;
    if ( ((v4 + 63) & 0xFFFFFFFFFFFFFFC0LL) - v4 + 192 <= v5 - v4 )
    {
      v1 = (_QWORD *)((v4 + 63) & 0xFFFFFFFFFFFFFFC0LL);
      *(_QWORD *)(a1 + 8) = v1 + 24;
    }
    else
    {
      v6 = *(unsigned int *)(a1 + 32);
      v7 = 0x40000000000LL;
      v8 = *(_DWORD *)(a1 + 32) >> 7;
      if ( v8 < 0x1E )
        v7 = 4096LL << v8;
      v9 = malloc(v7);
      if ( !v9 )
      {
        sub_16BD1C0("Allocation failed", 1u);
        v6 = *(unsigned int *)(a1 + 32);
        v9 = 0;
      }
      if ( *(_DWORD *)(a1 + 36) <= (unsigned int)v6 )
      {
        v12 = v9;
        sub_16CD150(a1 + 24, (const void *)(a1 + 40), 0, 8, v10, v11);
        v6 = *(unsigned int *)(a1 + 32);
        v9 = v12;
      }
      v1 = (_QWORD *)((v9 + 63) & 0xFFFFFFFFFFFFFFC0LL);
      *(_QWORD *)(*(_QWORD *)(a1 + 24) + 8 * v6) = v9;
      ++*(_DWORD *)(a1 + 32);
      *(_QWORD *)(a1 + 16) = v9 + v7;
      *(_QWORD *)(a1 + 8) = v1 + 24;
    }
    if ( !v1 )
      return 0;
  }
  memset(v1, 0, 0xB8u);
  v2 = v1 + 18;
  do
    *v2++ = 0;
  while ( (_DWORD *)((char *)v1 + 180) != v2 );
  return v1;
}
