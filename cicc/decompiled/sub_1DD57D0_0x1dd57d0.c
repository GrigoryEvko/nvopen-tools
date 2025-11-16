// Function: sub_1DD57D0
// Address: 0x1dd57d0
//
unsigned __int64 __fastcall sub_1DD57D0(__int64 a1, __int64 a2, int a3)
{
  __int64 v4; // rdx
  __int64 v5; // rcx
  unsigned int v6; // ebx
  unsigned __int64 v7; // r15
  __int64 v8; // rax
  int v9; // r8d
  int v10; // r9d
  unsigned __int64 v11; // r15
  unsigned __int64 result; // rax
  __int64 v13; // [rsp+8h] [rbp-38h]

  v4 = *(_QWORD *)(a1 + 232);
  v5 = *(_QWORD *)(a1 + 240);
  *(_QWORD *)(a1 + 312) += 32LL;
  if ( ((v4 + 7) & 0xFFFFFFFFFFFFFFF8LL) - v4 + 32 <= v5 - v4 )
  {
    result = (v4 + 7) & 0xFFFFFFFFFFFFFFF8LL;
    *(_QWORD *)(a1 + 232) = result + 32;
  }
  else
  {
    v6 = *(_DWORD *)(a1 + 256);
    v7 = 0x40000000000LL;
    if ( v6 >> 7 < 0x1E )
      v7 = 4096LL << (v6 >> 7);
    v8 = malloc(v7);
    if ( !v8 )
    {
      sub_16BD1C0("Allocation failed", 1u);
      v6 = *(_DWORD *)(a1 + 256);
      v8 = 0;
    }
    if ( *(_DWORD *)(a1 + 260) <= v6 )
    {
      v13 = v8;
      sub_16CD150(a1 + 248, (const void *)(a1 + 264), 0, 8, v9, v10);
      v6 = *(_DWORD *)(a1 + 256);
      v8 = v13;
    }
    v11 = v8 + v7;
    *(_QWORD *)(*(_QWORD *)(a1 + 248) + 8LL * v6) = v8;
    result = (v8 + 7) & 0xFFFFFFFFFFFFFFF8LL;
    *(_QWORD *)(a1 + 240) = v11;
    ++*(_DWORD *)(a1 + 256);
    *(_QWORD *)(a1 + 232) = result + 32;
  }
  *(_QWORD *)result = 0;
  *(_QWORD *)(result + 8) = 0;
  *(_QWORD *)(result + 16) = a2;
  *(_DWORD *)(result + 24) = a3;
  return result;
}
