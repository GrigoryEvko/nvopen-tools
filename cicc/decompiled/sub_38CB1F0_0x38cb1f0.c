// Function: sub_38CB1F0
// Address: 0x38cb1f0
//
unsigned __int64 __fastcall sub_38CB1F0(int a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v7; // rdx
  unsigned __int64 v8; // rbx
  __int64 v9; // rax
  int v10; // r9d
  __int64 v11; // rdx
  unsigned __int64 v12; // r9
  unsigned __int64 result; // rax
  __int64 v14; // [rsp+0h] [rbp-40h]
  __int64 v15; // [rsp+0h] [rbp-40h]
  unsigned int v16; // [rsp+8h] [rbp-38h]
  __int64 v17; // [rsp+8h] [rbp-38h]

  v7 = *(_QWORD *)(a4 + 48);
  *(_QWORD *)(a4 + 128) += 40LL;
  if ( ((v7 + 7) & 0xFFFFFFFFFFFFFFF8LL) - v7 + 40 <= *(_QWORD *)(a4 + 56) - v7 )
  {
    result = (v7 + 7) & 0xFFFFFFFFFFFFFFF8LL;
    *(_QWORD *)(a4 + 48) = result + 40;
  }
  else
  {
    v14 = a5;
    v8 = 0x40000000000LL;
    v16 = *(_DWORD *)(a4 + 72);
    if ( v16 >> 7 < 0x1E )
      v8 = 4096LL << (v16 >> 7);
    v9 = malloc(v8);
    v11 = v16;
    a5 = v14;
    if ( !v9 )
    {
      sub_16BD1C0("Allocation failed", 1u);
      v11 = *(unsigned int *)(a4 + 72);
      v9 = 0;
      a5 = v14;
    }
    if ( *(_DWORD *)(a4 + 76) <= (unsigned int)v11 )
    {
      v15 = v9;
      v17 = a5;
      sub_16CD150(a4 + 64, (const void *)(a4 + 80), 0, 8, a5, v10);
      v11 = *(unsigned int *)(a4 + 72);
      v9 = v15;
      a5 = v17;
    }
    v12 = v9 + v8;
    *(_QWORD *)(*(_QWORD *)(a4 + 64) + 8 * v11) = v9;
    result = (v9 + 7) & 0xFFFFFFFFFFFFFFF8LL;
    ++*(_DWORD *)(a4 + 72);
    *(_QWORD *)(a4 + 56) = v12;
    *(_QWORD *)(a4 + 48) = result + 40;
  }
  *(_DWORD *)result = 0;
  *(_QWORD *)(result + 8) = a5;
  *(_DWORD *)(result + 16) = a1;
  *(_QWORD *)(result + 24) = a2;
  *(_QWORD *)(result + 32) = a3;
  return result;
}
