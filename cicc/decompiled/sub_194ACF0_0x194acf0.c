// Function: sub_194ACF0
// Address: 0x194acf0
//
_QWORD *__fastcall sub_194ACF0(__int64 a1)
{
  __int64 v1; // rdx
  __int64 v2; // rax
  __int64 v3; // r14
  unsigned __int64 v4; // r13
  unsigned int v5; // ecx
  __int64 v6; // rax
  int v7; // r8d
  int v8; // r9d
  _QWORD *v9; // r8
  __int64 v11; // [rsp+8h] [rbp-28h]

  v1 = *(_QWORD *)(a1 + 56);
  v2 = *(_QWORD *)(a1 + 64);
  *(_QWORD *)(a1 + 136) += 168LL;
  if ( ((v1 + 7) & 0xFFFFFFFFFFFFFFF8LL) - v1 + 168 <= v2 - v1 )
  {
    v9 = (_QWORD *)((v1 + 7) & 0xFFFFFFFFFFFFFFF8LL);
    *(_QWORD *)(a1 + 56) = v9 + 21;
  }
  else
  {
    v3 = *(unsigned int *)(a1 + 80);
    v4 = 0x40000000000LL;
    v5 = *(_DWORD *)(a1 + 80) >> 7;
    if ( v5 < 0x1E )
      v4 = 4096LL << v5;
    v6 = malloc(v4);
    if ( !v6 )
    {
      sub_16BD1C0("Allocation failed", 1u);
      v3 = *(unsigned int *)(a1 + 80);
      v6 = 0;
    }
    if ( *(_DWORD *)(a1 + 84) <= (unsigned int)v3 )
    {
      v11 = v6;
      sub_16CD150(a1 + 72, (const void *)(a1 + 88), 0, 8, v7, v8);
      v3 = *(unsigned int *)(a1 + 80);
      v6 = v11;
    }
    v9 = (_QWORD *)((v6 + 7) & 0xFFFFFFFFFFFFFFF8LL);
    *(_QWORD *)(*(_QWORD *)(a1 + 72) + 8 * v3) = v6;
    ++*(_DWORD *)(a1 + 80);
    *(_QWORD *)(a1 + 64) = v6 + v4;
    *(_QWORD *)(a1 + 56) = v9 + 21;
  }
  memset(v9, 0, 0xA8u);
  v9[8] = v9 + 12;
  v9[9] = v9 + 12;
  v9[10] = 8;
  return v9;
}
