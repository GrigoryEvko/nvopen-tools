// Function: sub_1E6ACC0
// Address: 0x1e6acc0
//
void *__fastcall sub_1E6ACC0(__int64 *a1, __int64 a2)
{
  __int64 (*v3)(void); // rdx
  char v4; // al
  __int64 (*v5)(); // rax
  int v6; // edx
  int v7; // r8d
  int v8; // r9d
  unsigned int v9; // r12d
  __int64 v10; // r15
  __int64 v11; // r14
  unsigned int v12; // eax
  void *result; // rax
  __int64 v14; // rcx
  __int64 v15; // rdi
  unsigned __int64 v16; // rax
  unsigned int v17; // r13d
  int v18; // r12d
  unsigned __int64 v19; // rdx
  unsigned int v20; // r13d
  unsigned __int64 v21; // rdi
  unsigned __int64 v22; // r13
  __int64 v23; // rdx
  int v24; // ecx
  unsigned int v25; // r8d
  int v26; // ecx
  __int64 v27; // rcx
  unsigned __int64 v28; // rdx
  unsigned __int64 v29; // rax
  unsigned __int64 v30; // r13
  unsigned int v31; // [rsp+8h] [rbp-38h]
  __int64 v32; // [rsp+8h] [rbp-38h]

  *a1 = a2;
  a1[1] = 0;
  v3 = *(__int64 (**)(void))(**(_QWORD **)(a2 + 16) + 296LL);
  v4 = 0;
  if ( v3 != sub_1E69390 )
  {
    v4 = v3();
    if ( v4 )
      v4 = byte_4FC72A0;
    a2 = *a1;
  }
  *((_BYTE *)a1 + 16) = v4;
  a1[8] = (__int64)(a1 + 10);
  a1[10] = (__int64)(a1 + 12);
  a1[17] = 0x1000000000LL;
  a1[21] = 0x1000000000LL;
  a1[26] = (__int64)(a1 + 28);
  a1[29] = (__int64)(a1 + 31);
  a1[30] = 0x400000000LL;
  a1[41] = (__int64)(a1 + 43);
  a1[3] = (__int64)(a1 + 5);
  a1[4] = 0;
  a1[5] = 0;
  a1[6] = 0;
  a1[9] = 0;
  a1[11] = 0;
  *((_BYTE *)a1 + 96) = 0;
  a1[15] = 0;
  a1[16] = 0;
  *((_BYTE *)a1 + 152) = 0;
  a1[20] = (__int64)(a1 + 22);
  a1[27] = 0;
  *((_DWORD *)a1 + 56) = 0;
  a1[34] = 0;
  a1[35] = 0;
  a1[36] = 0;
  *((_DWORD *)a1 + 74) = 0;
  a1[38] = 0;
  a1[39] = 0;
  *((_DWORD *)a1 + 80) = 0;
  a1[42] = 0;
  a1[43] = 0;
  a1[45] = 0;
  a1[46] = 0;
  a1[47] = 0;
  v5 = *(__int64 (**)())(**(_QWORD **)(a2 + 16) + 112LL);
  if ( v5 == sub_1D00B10 )
    BUG();
  v9 = *(_DWORD *)(v5() + 16);
  if ( *((_DWORD *)a1 + 9) <= 0xFFu )
    sub_16CD150((__int64)(a1 + 3), a1 + 5, 0x100u, 16, v7, v8);
  if ( *((_DWORD *)a1 + 55) <= 0xFFu )
    sub_1E6AA60((unsigned __int64 *)a1 + 26, 0x100u);
  v10 = a1[36];
  v11 = v9;
  if ( v9 > (unsigned __int64)(v10 << 6) )
  {
    v21 = a1[35];
    v22 = (v9 + 63) >> 6;
    if ( v22 < 2 * v10 )
      v22 = 2 * v10;
    v23 = (__int64)realloc(v21, 8 * v22, v6, 8 * (int)v22, v7, v8);
    if ( !v23 && (8 * v22 || (v23 = malloc(1u)) == 0) )
    {
      v32 = v23;
      sub_16BD1C0("Allocation failed", 1u);
      v23 = v32;
    }
    v24 = *((_DWORD *)a1 + 74);
    a1[35] = v23;
    a1[36] = v22;
    v25 = (unsigned int)(v24 + 63) >> 6;
    if ( v22 > v25 )
    {
      v30 = v22 - v25;
      if ( v30 )
      {
        v31 = (unsigned int)(v24 + 63) >> 6;
        memset((void *)(v23 + 8LL * v25), 0, 8 * v30);
        v24 = *((_DWORD *)a1 + 74);
        v23 = a1[35];
        v25 = v31;
      }
    }
    v26 = v24 & 0x3F;
    if ( v26 )
    {
      *(_QWORD *)(v23 + 8LL * (v25 - 1)) &= ~(-1LL << v26);
      v23 = a1[35];
    }
    v27 = a1[36] - (unsigned int)v10;
    if ( v27 )
      memset((void *)(v23 + 8LL * (unsigned int)v10), 0, 8 * v27);
  }
  v12 = *((_DWORD *)a1 + 74);
  if ( v9 > v12 )
  {
    v19 = a1[36];
    v20 = (v12 + 63) >> 6;
    if ( v19 > v20 )
    {
      v28 = v19 - v20;
      if ( v28 )
      {
        memset((void *)(a1[35] + 8LL * v20), 0, 8 * v28);
        v12 = *((_DWORD *)a1 + 74);
      }
    }
    if ( (v12 & 0x3F) != 0 )
    {
      *(_QWORD *)(a1[35] + 8LL * (v20 - 1)) &= ~(-1LL << (v12 & 0x3F));
      v12 = *((_DWORD *)a1 + 74);
    }
  }
  *((_DWORD *)a1 + 74) = v9;
  if ( v9 < v12 )
  {
    v16 = a1[36];
    v17 = (v9 + 63) >> 6;
    if ( v16 > v17 )
    {
      v29 = v16 - v17;
      if ( v29 )
      {
        memset((void *)(a1[35] + 8LL * v17), 0, 8 * v29);
        v9 = *((_DWORD *)a1 + 74);
      }
    }
    v18 = v9 & 0x3F;
    if ( v18 )
      *(_QWORD *)(a1[35] + 8LL * (v17 - 1)) &= ~(-1LL << v18);
  }
  result = (void *)sub_2207820(8 * v11);
  v14 = (__int64)result;
  if ( result && v11 )
  {
    result = memset(result, 0, 8 * v11);
    v14 = (__int64)result;
  }
  v15 = a1[34];
  a1[34] = v14;
  if ( v15 )
    return (void *)j_j___libc_free_0_0(v15);
  return result;
}
