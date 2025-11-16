// Function: sub_19E3CC0
// Address: 0x19e3cc0
//
unsigned __int64 __fastcall sub_19E3CC0(__int64 a1, __int64 *a2)
{
  __int64 v2; // rax
  __int64 v3; // rdx
  __int64 v4; // r14
  __int64 v5; // rdx
  unsigned int v6; // ebx
  unsigned __int64 v7; // r15
  __int64 v8; // rax
  int v9; // r8d
  int v10; // r9d
  unsigned __int64 v11; // r15
  unsigned __int64 result; // rax
  __int64 v13; // rbx
  int v14; // r8d
  int v15; // r9d
  __int64 v16; // rax
  __int64 *v17; // rax
  __int64 v18; // [rsp+8h] [rbp-38h]

  v2 = *a2;
  v3 = *a2 + 3;
  v4 = 4LL * *(unsigned int *)(a1 + 48);
  a2[10] += v4;
  v5 = (v3 & 0xFFFFFFFFFFFFFFFCLL) - v2;
  if ( v4 + v5 <= (unsigned __int64)(a2[1] - v2) )
  {
    result = v5 + v2;
    *a2 = result + v4;
  }
  else if ( (unsigned __int64)(v4 + 3) > 0x1000 )
  {
    v13 = malloc(v4 + 3);
    if ( !v13 )
      sub_16BD1C0("Allocation failed", 1u);
    v16 = *((unsigned int *)a2 + 18);
    if ( (unsigned int)v16 >= *((_DWORD *)a2 + 19) )
    {
      sub_16CD150((__int64)(a2 + 8), a2 + 10, 0, 16, v14, v15);
      v16 = *((unsigned int *)a2 + 18);
    }
    v17 = (__int64 *)(a2[8] + 16 * v16);
    *v17 = v13;
    v17[1] = v4 + 3;
    ++*((_DWORD *)a2 + 18);
    result = (v13 + 3) & 0xFFFFFFFFFFFFFFFCLL;
  }
  else
  {
    v6 = *((_DWORD *)a2 + 6);
    v7 = 0x40000000000LL;
    if ( v6 >> 7 < 0x1E )
      v7 = 4096LL << (v6 >> 7);
    v8 = malloc(v7);
    if ( !v8 )
    {
      sub_16BD1C0("Allocation failed", 1u);
      v6 = *((_DWORD *)a2 + 6);
      v8 = 0;
    }
    if ( v6 >= *((_DWORD *)a2 + 7) )
    {
      v18 = v8;
      sub_16CD150((__int64)(a2 + 2), a2 + 4, 0, 8, v9, v10);
      v6 = *((_DWORD *)a2 + 6);
      v8 = v18;
    }
    v11 = v8 + v7;
    *(_QWORD *)(a2[2] + 8LL * v6) = v8;
    result = (v8 + 3) & 0xFFFFFFFFFFFFFFFCLL;
    ++*((_DWORD *)a2 + 6);
    a2[1] = v11;
    *a2 = result + v4;
  }
  *(_QWORD *)(a1 + 56) = result;
  return result;
}
