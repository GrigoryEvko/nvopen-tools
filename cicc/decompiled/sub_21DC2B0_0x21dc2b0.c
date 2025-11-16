// Function: sub_21DC2B0
// Address: 0x21dc2b0
//
_QWORD *__fastcall sub_21DC2B0(__int64 a1)
{
  _QWORD *result; // rax
  __int64 v2; // rdx
  __int64 v3; // rcx
  __int64 v4; // r14
  unsigned __int64 v5; // r13
  unsigned int v6; // ecx
  __int64 v7; // rax
  int v8; // r8d
  int v9; // r9d
  unsigned __int64 v10; // r13
  __int64 v11; // [rsp-30h] [rbp-30h]

  result = *(_QWORD **)(a1 + 48);
  if ( !result )
  {
    v2 = *(_QWORD *)(a1 + 120);
    v3 = *(_QWORD *)(a1 + 128);
    *(_QWORD *)(a1 + 200) += 280LL;
    if ( ((v2 + 7) & 0xFFFFFFFFFFFFFFF8LL) - v2 + 280 <= v3 - v2 )
    {
      result = (_QWORD *)((v2 + 7) & 0xFFFFFFFFFFFFFFF8LL);
      *(_QWORD *)(a1 + 120) = result + 35;
    }
    else
    {
      v4 = *(unsigned int *)(a1 + 144);
      v5 = 0x40000000000LL;
      v6 = *(_DWORD *)(a1 + 144) >> 7;
      if ( v6 < 0x1E )
        v5 = 4096LL << v6;
      v7 = malloc(v5);
      if ( !v7 )
      {
        sub_16BD1C0("Allocation failed", 1u);
        v4 = *(unsigned int *)(a1 + 144);
        v7 = 0;
      }
      if ( *(_DWORD *)(a1 + 148) <= (unsigned int)v4 )
      {
        v11 = v7;
        sub_16CD150(a1 + 136, (const void *)(a1 + 152), 0, 8, v8, v9);
        v4 = *(unsigned int *)(a1 + 144);
        v7 = v11;
      }
      v10 = v7 + v5;
      *(_QWORD *)(*(_QWORD *)(a1 + 136) + 8 * v4) = v7;
      result = (_QWORD *)((v7 + 7) & 0xFFFFFFFFFFFFFFF8LL);
      *(_QWORD *)(a1 + 128) = v10;
      ++*(_DWORD *)(a1 + 144);
      *(_QWORD *)(a1 + 120) = result + 35;
    }
    result[2] = 0x800000000LL;
    *result = &unk_4A016E8;
    result[1] = result + 3;
    *(_QWORD *)(a1 + 48) = result;
  }
  return result;
}
