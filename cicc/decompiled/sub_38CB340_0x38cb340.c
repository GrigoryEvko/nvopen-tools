// Function: sub_38CB340
// Address: 0x38cb340
//
unsigned __int64 __fastcall sub_38CB340(int a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v6; // rdx
  unsigned __int64 v7; // rbx
  __int64 v8; // rax
  int v9; // r8d
  int v10; // r9d
  __int64 v11; // rdx
  unsigned __int64 v12; // r8
  unsigned __int64 result; // rax
  unsigned int v14; // [rsp+8h] [rbp-38h]
  __int64 v15; // [rsp+8h] [rbp-38h]

  *(_QWORD *)(a3 + 128) += 32LL;
  v6 = *(_QWORD *)(a3 + 48);
  if ( ((v6 + 7) & 0xFFFFFFFFFFFFFFF8LL) - v6 + 32 <= *(_QWORD *)(a3 + 56) - v6 )
  {
    result = (v6 + 7) & 0xFFFFFFFFFFFFFFF8LL;
    *(_QWORD *)(a3 + 48) = result + 32;
  }
  else
  {
    v7 = 0x40000000000LL;
    v14 = *(_DWORD *)(a3 + 72);
    if ( v14 >> 7 < 0x1E )
      v7 = 4096LL << (v14 >> 7);
    v8 = malloc(v7);
    v11 = v14;
    if ( !v8 )
    {
      sub_16BD1C0("Allocation failed", 1u);
      v11 = *(unsigned int *)(a3 + 72);
      v8 = 0;
    }
    if ( *(_DWORD *)(a3 + 76) <= (unsigned int)v11 )
    {
      v15 = v8;
      sub_16CD150(a3 + 64, (const void *)(a3 + 80), 0, 8, v9, v10);
      v11 = *(unsigned int *)(a3 + 72);
      v8 = v15;
    }
    v12 = v8 + v7;
    *(_QWORD *)(*(_QWORD *)(a3 + 64) + 8 * v11) = v8;
    result = (v8 + 7) & 0xFFFFFFFFFFFFFFF8LL;
    ++*(_DWORD *)(a3 + 72);
    *(_QWORD *)(a3 + 56) = v12;
    *(_QWORD *)(a3 + 48) = result + 32;
  }
  *(_DWORD *)result = 3;
  *(_QWORD *)(result + 8) = a4;
  *(_DWORD *)(result + 16) = a1;
  *(_QWORD *)(result + 24) = a2;
  return result;
}
