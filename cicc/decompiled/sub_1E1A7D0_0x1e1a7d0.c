// Function: sub_1E1A7D0
// Address: 0x1e1a7d0
//
_QWORD *__fastcall sub_1E1A7D0(__int64 a1, unsigned __int8 a2, __int64 *a3)
{
  _QWORD *v4; // rdx
  _QWORD *result; // rax
  __int64 v6; // rdx
  __int64 v7; // r13
  __int64 v8; // rcx
  __int64 v9; // r15
  unsigned __int64 v10; // r14
  unsigned int v11; // ecx
  __int64 v12; // rax
  int v13; // r8d
  int v14; // r9d
  unsigned __int64 v15; // r14
  __int64 v16; // rbx
  int v17; // r8d
  int v18; // r9d
  __int64 v19; // rax
  __int64 *v20; // rax
  __int64 v21; // [rsp+8h] [rbp-38h]

  if ( *(_DWORD *)(a1 + 8) > (unsigned int)a2
    && (v4 = (_QWORD *)(*(_QWORD *)a1 + 8LL * a2), (result = (_QWORD *)*v4) != 0) )
  {
    *v4 = *result;
  }
  else
  {
    v6 = *a3;
    v7 = 40LL << a2;
    v8 = a3[1];
    a3[10] += 40LL << a2;
    if ( (40LL << a2) + ((v6 + 7) & 0xFFFFFFFFFFFFFFF8LL) - v6 <= v8 - v6 )
    {
      result = (_QWORD *)((v6 + 7) & 0xFFFFFFFFFFFFFFF8LL);
      *a3 = (__int64)result + v7;
    }
    else if ( (unsigned __int64)(v7 + 7) > 0x1000 )
    {
      v16 = malloc(v7 + 7);
      if ( !v16 )
        sub_16BD1C0("Allocation failed", 1u);
      v19 = *((unsigned int *)a3 + 18);
      if ( (unsigned int)v19 >= *((_DWORD *)a3 + 19) )
      {
        sub_16CD150((__int64)(a3 + 8), a3 + 10, 0, 16, v17, v18);
        v19 = *((unsigned int *)a3 + 18);
      }
      v20 = (__int64 *)(a3[8] + 16 * v19);
      *v20 = v16;
      v20[1] = v7 + 7;
      ++*((_DWORD *)a3 + 18);
      return (_QWORD *)((v16 + 7) & 0xFFFFFFFFFFFFFFF8LL);
    }
    else
    {
      v9 = *((unsigned int *)a3 + 6);
      v10 = 0x40000000000LL;
      v11 = *((_DWORD *)a3 + 6) >> 7;
      if ( v11 < 0x1E )
        v10 = 4096LL << v11;
      v12 = malloc(v10);
      if ( !v12 )
      {
        sub_16BD1C0("Allocation failed", 1u);
        v9 = *((unsigned int *)a3 + 6);
        v12 = 0;
      }
      if ( (unsigned int)v9 >= *((_DWORD *)a3 + 7) )
      {
        v21 = v12;
        sub_16CD150((__int64)(a3 + 2), a3 + 4, 0, 8, v13, v14);
        v9 = *((unsigned int *)a3 + 6);
        v12 = v21;
      }
      v15 = v12 + v10;
      *(_QWORD *)(a3[2] + 8 * v9) = v12;
      result = (_QWORD *)((v12 + 7) & 0xFFFFFFFFFFFFFFF8LL);
      a3[1] = v15;
      ++*((_DWORD *)a3 + 6);
      *a3 = (__int64)result + v7;
    }
  }
  return result;
}
