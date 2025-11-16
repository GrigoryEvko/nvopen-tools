// Function: sub_38CB470
// Address: 0x38cb470
//
unsigned __int64 __fastcall sub_38CB470(__int64 a1, __int64 a2)
{
  __int64 v2; // rdx
  __int64 v3; // r15
  unsigned __int64 v4; // r14
  unsigned int v5; // ecx
  __int64 v6; // rax
  int v7; // r8d
  int v8; // r9d
  unsigned __int64 v9; // r14
  unsigned __int64 result; // rax
  __int64 v11; // [rsp+8h] [rbp-28h]

  v2 = *(_QWORD *)(a2 + 48);
  *(_QWORD *)(a2 + 128) += 24LL;
  if ( ((v2 + 7) & 0xFFFFFFFFFFFFFFF8LL) - v2 + 24 <= *(_QWORD *)(a2 + 56) - v2 )
  {
    result = (v2 + 7) & 0xFFFFFFFFFFFFFFF8LL;
    *(_QWORD *)(a2 + 48) = result + 24;
  }
  else
  {
    v3 = *(unsigned int *)(a2 + 72);
    v4 = 0x40000000000LL;
    v5 = *(_DWORD *)(a2 + 72) >> 7;
    if ( v5 < 0x1E )
      v4 = 4096LL << v5;
    v6 = malloc(v4);
    if ( !v6 )
    {
      sub_16BD1C0("Allocation failed", 1u);
      v3 = *(unsigned int *)(a2 + 72);
      v6 = 0;
    }
    if ( *(_DWORD *)(a2 + 76) <= (unsigned int)v3 )
    {
      v11 = v6;
      sub_16CD150(a2 + 64, (const void *)(a2 + 80), 0, 8, v7, v8);
      v3 = *(unsigned int *)(a2 + 72);
      v6 = v11;
    }
    v9 = v6 + v4;
    *(_QWORD *)(*(_QWORD *)(a2 + 64) + 8 * v3) = v6;
    result = (v6 + 7) & 0xFFFFFFFFFFFFFFF8LL;
    ++*(_DWORD *)(a2 + 72);
    *(_QWORD *)(a2 + 56) = v9;
    *(_QWORD *)(a2 + 48) = result + 24;
  }
  *(_DWORD *)result = 1;
  *(_QWORD *)(result + 8) = 0;
  *(_QWORD *)(result + 16) = a1;
  return result;
}
