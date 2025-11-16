// Function: sub_38E22B0
// Address: 0x38e22b0
//
unsigned __int64 __fastcall sub_38E22B0(unsigned int a1, __int64 a2, __int64 a3)
{
  __int64 v3; // r13
  __int64 v5; // rax
  __int64 v6; // rcx
  unsigned int v7; // ebx
  unsigned __int64 v8; // r15
  __int64 v9; // rax
  int v10; // r8d
  int v11; // r9d
  unsigned __int64 v12; // r15
  unsigned __int64 v13; // rax
  __int64 v15; // rbx
  int v16; // r8d
  int v17; // r9d
  __int64 v18; // rax
  __int64 *v19; // rax
  __int64 v20; // [rsp+8h] [rbp-38h]

  v3 = 0;
  if ( a2 )
  {
    a1 += 8;
    v3 = 8;
  }
  v5 = *(_QWORD *)(a3 + 48);
  v6 = *(_QWORD *)(a3 + 56);
  *(_QWORD *)(a3 + 128) += a1;
  if ( a1 + ((v5 + 7) & 0xFFFFFFFFFFFFFFF8LL) - v5 <= v6 - v5 )
  {
    v13 = (v5 + 7) & 0xFFFFFFFFFFFFFFF8LL;
    *(_QWORD *)(a3 + 48) = v13 + a1;
  }
  else if ( (unsigned __int64)a1 + 7 > 0x1000 )
  {
    v15 = malloc(a1 + 7LL);
    if ( !v15 )
      sub_16BD1C0("Allocation failed", 1u);
    v18 = *(unsigned int *)(a3 + 120);
    if ( (unsigned int)v18 >= *(_DWORD *)(a3 + 124) )
    {
      sub_16CD150(a3 + 112, (const void *)(a3 + 128), 0, 16, v16, v17);
      v18 = *(unsigned int *)(a3 + 120);
    }
    v19 = (__int64 *)(*(_QWORD *)(a3 + 112) + 16 * v18);
    *v19 = v15;
    v19[1] = a1 + 7LL;
    ++*(_DWORD *)(a3 + 120);
    v13 = (v15 + 7) & 0xFFFFFFFFFFFFFFF8LL;
  }
  else
  {
    v7 = *(_DWORD *)(a3 + 72);
    v8 = 0x40000000000LL;
    if ( v7 >> 7 < 0x1E )
      v8 = 4096LL << (v7 >> 7);
    v9 = malloc(v8);
    if ( !v9 )
    {
      sub_16BD1C0("Allocation failed", 1u);
      v7 = *(_DWORD *)(a3 + 72);
      v9 = 0;
    }
    if ( v7 >= *(_DWORD *)(a3 + 76) )
    {
      v20 = v9;
      sub_16CD150(a3 + 64, (const void *)(a3 + 80), 0, 8, v10, v11);
      v7 = *(_DWORD *)(a3 + 72);
      v9 = v20;
    }
    v12 = v9 + v8;
    *(_QWORD *)(*(_QWORD *)(a3 + 64) + 8LL * v7) = v9;
    v13 = (v9 + 7) & 0xFFFFFFFFFFFFFFF8LL;
    ++*(_DWORD *)(a3 + 72);
    *(_QWORD *)(a3 + 56) = v12;
    *(_QWORD *)(a3 + 48) = v13 + a1;
  }
  return v3 + v13;
}
