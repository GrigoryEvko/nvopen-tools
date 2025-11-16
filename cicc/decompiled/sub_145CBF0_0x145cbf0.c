// Function: sub_145CBF0
// Address: 0x145cbf0
//
__int64 __fastcall sub_145CBF0(__int64 *a1, __int64 a2, __int64 a3)
{
  __int64 v3; // r14
  __int64 v4; // r12
  __int64 v5; // rdx
  __int64 v6; // rcx
  unsigned __int64 v7; // r15
  __int64 v8; // rax
  __int64 v9; // rdx
  unsigned __int64 v10; // r8
  __int64 result; // rax
  __int64 v12; // rax
  __int64 v13; // rdx
  __int64 *v14; // rdx
  unsigned int v15; // [rsp+8h] [rbp-38h]
  __int64 v16; // [rsp+8h] [rbp-38h]
  __int64 v17; // [rsp+8h] [rbp-38h]

  v3 = a3 - 1;
  v4 = -a3;
  v5 = *a1;
  v6 = a1[1];
  a1[10] += a2;
  if ( (v4 & (unsigned __int64)(v5 + v3)) - v5 + a2 <= v6 - v5 )
  {
    result = v4 & (v5 + v3);
    *a1 = result + a2;
  }
  else if ( (unsigned __int64)(a2 + v3) > 0x1000 )
  {
    v12 = malloc(a2 + v3);
    if ( !v12 )
    {
      sub_16BD1C0("Allocation failed");
      v12 = 0;
    }
    v13 = *((unsigned int *)a1 + 18);
    if ( (unsigned int)v13 >= *((_DWORD *)a1 + 19) )
    {
      v17 = v12;
      sub_16CD150(a1 + 8, a1 + 10, 0, 16);
      v13 = *((unsigned int *)a1 + 18);
      v12 = v17;
    }
    v14 = (__int64 *)(a1[8] + 16 * v13);
    *v14 = v12;
    v14[1] = a2 + v3;
    ++*((_DWORD *)a1 + 18);
    return v4 & (v3 + v12);
  }
  else
  {
    v7 = 0x40000000000LL;
    v15 = *((_DWORD *)a1 + 6);
    if ( v15 >> 7 < 0x1E )
      v7 = 4096LL << (v15 >> 7);
    v8 = malloc(v7);
    v9 = v15;
    if ( !v8 )
    {
      sub_16BD1C0("Allocation failed");
      v9 = *((unsigned int *)a1 + 6);
      v8 = 0;
    }
    if ( (unsigned int)v9 >= *((_DWORD *)a1 + 7) )
    {
      v16 = v8;
      sub_16CD150(a1 + 2, a1 + 4, 0, 8);
      v9 = *((unsigned int *)a1 + 6);
      v8 = v16;
    }
    v10 = v8 + v7;
    *(_QWORD *)(a1[2] + 8 * v9) = v8;
    result = v4 & (v3 + v8);
    ++*((_DWORD *)a1 + 6);
    a1[1] = v10;
    *a1 = result + a2;
  }
  return result;
}
