// Function: sub_161FD00
// Address: 0x161fd00
//
_QWORD *__fastcall sub_161FD00(void *src, size_t n, __int64 *a3)
{
  __int64 v4; // rdx
  unsigned __int64 v5; // r12
  __int64 v6; // rax
  __int64 v7; // rdx
  size_t v8; // r8
  unsigned __int64 v9; // r9
  _QWORD *v10; // r12
  _BYTE *v11; // rcx
  __int64 v13; // rax
  __int64 v14; // r9
  __int64 v15; // r12
  __int64 v16; // rax
  __int64 *v17; // rax
  __int64 v18; // [rsp+0h] [rbp-40h]
  unsigned int v19; // [rsp+8h] [rbp-38h]
  size_t v20; // [rsp+8h] [rbp-38h]
  __int64 v21; // [rsp+8h] [rbp-38h]

  a3[10] += n + 25;
  v4 = *a3;
  if ( n + 25 + ((v4 + 7) & 0xFFFFFFFFFFFFFFF8LL) - v4 <= a3[1] - v4 )
  {
    v10 = (_QWORD *)((v4 + 7) & 0xFFFFFFFFFFFFFFF8LL);
    *a3 = (__int64)v10 + n + 25;
  }
  else if ( n + 32 > 0x1000 )
  {
    v13 = malloc(n + 32);
    v14 = n + 32;
    v15 = v13;
    if ( !v13 )
    {
      sub_16BD1C0("Allocation failed");
      v14 = n + 32;
    }
    v16 = *((unsigned int *)a3 + 18);
    if ( (unsigned int)v16 >= *((_DWORD *)a3 + 19) )
    {
      v21 = v14;
      sub_16CD150(a3 + 8, a3 + 10, 0, 16);
      v16 = *((unsigned int *)a3 + 18);
      v14 = v21;
    }
    v17 = (__int64 *)(a3[8] + 16 * v16);
    *v17 = v15;
    v17[1] = v14;
    v10 = (_QWORD *)((v15 + 7) & 0xFFFFFFFFFFFFFFF8LL);
    ++*((_DWORD *)a3 + 18);
  }
  else
  {
    v5 = 0x40000000000LL;
    v19 = *((_DWORD *)a3 + 6);
    if ( v19 >> 7 < 0x1E )
      v5 = 4096LL << (v19 >> 7);
    v6 = malloc(v5);
    v7 = v19;
    v8 = n + 25;
    if ( !v6 )
    {
      sub_16BD1C0("Allocation failed");
      v7 = *((unsigned int *)a3 + 6);
      v6 = 0;
      v8 = n + 25;
    }
    if ( (unsigned int)v7 >= *((_DWORD *)a3 + 7) )
    {
      v18 = v6;
      v20 = v8;
      sub_16CD150(a3 + 2, a3 + 4, 0, 8);
      v7 = *((unsigned int *)a3 + 6);
      v6 = v18;
      v8 = v20;
    }
    v9 = v6 + v5;
    v10 = (_QWORD *)((v6 + 7) & 0xFFFFFFFFFFFFFFF8LL);
    *(_QWORD *)(a3[2] + 8 * v7) = v6;
    ++*((_DWORD *)a3 + 6);
    a3[1] = v9;
    *a3 = (__int64)v10 + v8;
  }
  v11 = v10 + 3;
  if ( n + 1 > 1 )
    v11 = memcpy(v10 + 3, src, n);
  v11[n] = 0;
  *v10 = n;
  v10[1] = 0;
  v10[2] = 0;
  return v10;
}
