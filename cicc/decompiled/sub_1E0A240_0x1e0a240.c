// Function: sub_1E0A240
// Address: 0x1e0a240
//
unsigned __int64 __fastcall sub_1E0A240(__int64 a1, __int64 a2)
{
  __int64 v2; // r13
  __int64 v3; // rax
  __int64 v4; // rcx
  __int64 v5; // r15
  unsigned __int64 v6; // r14
  unsigned int v7; // ecx
  __int64 v8; // rax
  int v9; // r8d
  int v10; // r9d
  unsigned __int64 v11; // r14
  unsigned __int64 result; // rax
  __int64 v13; // rbx
  int v14; // r8d
  int v15; // r9d
  __int64 v16; // rax
  __int64 *v17; // rax
  __int64 v18; // [rsp+8h] [rbp-38h]

  v2 = 8 * a2;
  v3 = *(_QWORD *)(a1 + 120);
  v4 = *(_QWORD *)(a1 + 128);
  *(_QWORD *)(a1 + 200) += 8 * a2;
  if ( 8 * a2 + ((v3 + 7) & 0xFFFFFFFFFFFFFFF8LL) - v3 <= v4 - v3 )
  {
    result = (v3 + 7) & 0xFFFFFFFFFFFFFFF8LL;
    *(_QWORD *)(a1 + 120) = result + v2;
  }
  else if ( (unsigned __int64)(v2 + 7) > 0x1000 )
  {
    v13 = malloc(v2 + 7);
    if ( !v13 )
      sub_16BD1C0("Allocation failed", 1u);
    v16 = *(unsigned int *)(a1 + 192);
    if ( (unsigned int)v16 >= *(_DWORD *)(a1 + 196) )
    {
      sub_16CD150(a1 + 184, (const void *)(a1 + 200), 0, 16, v14, v15);
      v16 = *(unsigned int *)(a1 + 192);
    }
    v17 = (__int64 *)(*(_QWORD *)(a1 + 184) + 16 * v16);
    *v17 = v13;
    v17[1] = v2 + 7;
    ++*(_DWORD *)(a1 + 192);
    return (v13 + 7) & 0xFFFFFFFFFFFFFFF8LL;
  }
  else
  {
    v5 = *(unsigned int *)(a1 + 144);
    v6 = 0x40000000000LL;
    v7 = *(_DWORD *)(a1 + 144) >> 7;
    if ( v7 < 0x1E )
      v6 = 4096LL << v7;
    v8 = malloc(v6);
    if ( !v8 )
    {
      sub_16BD1C0("Allocation failed", 1u);
      v5 = *(unsigned int *)(a1 + 144);
      v8 = 0;
    }
    if ( (unsigned int)v5 >= *(_DWORD *)(a1 + 148) )
    {
      v18 = v8;
      sub_16CD150(a1 + 136, (const void *)(a1 + 152), 0, 8, v9, v10);
      v5 = *(unsigned int *)(a1 + 144);
      v8 = v18;
    }
    v11 = v8 + v6;
    *(_QWORD *)(*(_QWORD *)(a1 + 136) + 8 * v5) = v8;
    result = (v8 + 7) & 0xFFFFFFFFFFFFFFF8LL;
    *(_QWORD *)(a1 + 128) = v11;
    ++*(_DWORD *)(a1 + 144);
    *(_QWORD *)(a1 + 120) = result + v2;
  }
  return result;
}
