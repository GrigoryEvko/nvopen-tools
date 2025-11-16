// Function: sub_9D1E70
// Address: 0x9d1e70
//
__int64 __fastcall sub_9D1E70(__int64 a1, __int64 a2, __int64 a3, char a4)
{
  __int64 v4; // r13
  __int64 v5; // r12
  __int64 v6; // r14
  __int64 v8; // r14
  unsigned int v9; // ecx
  __int64 v10; // rax
  __int64 v11; // rdx
  __int64 v12; // r14
  __int64 result; // rax
  __int64 v14; // rax
  __int64 v15; // rdx
  __int64 *v16; // rdx
  __int64 v17; // [rsp+8h] [rbp-38h]
  __int64 v18; // [rsp+8h] [rbp-38h]

  v4 = 1LL << a4;
  v5 = (1LL << a4) - 1;
  v6 = a3 + v5;
  if ( (unsigned __int64)(a3 + v5) > 0x1000 )
  {
    v14 = sub_C7D670(a3 + v5, 16);
    v15 = *(unsigned int *)(a1 + 72);
    if ( v15 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 76) )
    {
      v18 = v14;
      sub_C8D5F0(a1 + 64, a1 + 80, v15 + 1, 16);
      v15 = *(unsigned int *)(a1 + 72);
      v14 = v18;
    }
    v16 = (__int64 *)(*(_QWORD *)(a1 + 64) + 16 * v15);
    *v16 = v14;
    v16[1] = v6;
    ++*(_DWORD *)(a1 + 72);
    return -v4 & (v5 + v14);
  }
  else
  {
    v8 = 0x40000000000LL;
    v9 = *(_DWORD *)(a1 + 24) >> 7;
    if ( v9 < 0x1E )
      v8 = 4096LL << v9;
    v10 = sub_C7D670(v8, 16);
    v11 = *(unsigned int *)(a1 + 24);
    if ( v11 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 28) )
    {
      v17 = v10;
      sub_C8D5F0(a1 + 16, a1 + 32, v11 + 1, 8);
      v11 = *(unsigned int *)(a1 + 24);
      v10 = v17;
    }
    v12 = v10 + v8;
    *(_QWORD *)(*(_QWORD *)(a1 + 16) + 8 * v11) = v10;
    result = -v4 & (v5 + v10);
    *(_QWORD *)(a1 + 8) = v12;
    ++*(_DWORD *)(a1 + 24);
    *(_QWORD *)a1 = result + a3;
  }
  return result;
}
