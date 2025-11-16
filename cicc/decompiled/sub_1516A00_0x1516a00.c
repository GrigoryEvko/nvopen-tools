// Function: sub_1516A00
// Address: 0x1516a00
//
__int64 __fastcall sub_1516A00(__int64 *a1, int *a2)
{
  char *v4; // r14
  char *v5; // rsi
  __int64 v6; // r13
  __int64 v7; // rcx
  __int64 v8; // rdi
  unsigned __int64 v9; // rdx
  __int64 v10; // rax
  int v11; // edx
  __int64 *v12; // rdx
  __int64 result; // rax
  __int64 v14; // rdx
  __int64 v15; // r15
  __int64 v16; // rax
  unsigned __int64 v17; // r14
  _QWORD *v18; // r15
  const void *v19; // rsi
  __int64 v20; // rdx
  __int64 v21; // rax
  __int64 v22; // rax
  char *v23; // r14
  size_t v24; // rdx
  __int64 v25; // [rsp+8h] [rbp-38h]

  v4 = (char *)a1[9];
  v5 = (char *)a1[5];
  v6 = v4 - v5;
  v7 = (a1[6] - a1[7]) >> 4;
  v8 = (v4 - v5) >> 3;
  if ( v7 + 32 * (v8 - 1) + ((a1[4] - a1[2]) >> 4) == 0x7FFFFFFFFFFFFFFLL )
    sub_4262D8((__int64)"cannot create std::deque larger than max_size()");
  v9 = a1[1];
  if ( v9 - ((__int64)&v4[-*a1] >> 3) <= 1 )
  {
    v15 = v8 + 2;
    if ( v9 > 2 * (v8 + 2) )
    {
      v23 = v4 + 8;
      v18 = (_QWORD *)(*a1 + 8 * ((v9 - v15) >> 1));
      v24 = v23 - v5;
      if ( v5 <= (char *)v18 )
      {
        if ( v5 != v23 )
          memmove((char *)v18 + v6 + 8 - v24, v5, v24);
      }
      else if ( v5 != v23 )
      {
        memmove(v18, v5, v24);
      }
    }
    else
    {
      v16 = 1;
      if ( v9 )
        v16 = a1[1];
      v17 = v9 + v16 + 2;
      if ( v17 > 0xFFFFFFFFFFFFFFFLL )
        sub_4261EA(v8, v5, v9);
      v25 = sub_22077B0(8 * v17);
      v18 = (_QWORD *)(v25 + 8 * ((v17 - v15) >> 1));
      v19 = (const void *)a1[5];
      v20 = a1[9] + 8;
      if ( (const void *)v20 != v19 )
        memmove(v18, v19, v20 - (_QWORD)v19);
      j_j___libc_free_0(*a1, 8 * a1[1]);
      a1[1] = v17;
      *a1 = v25;
    }
    a1[5] = (__int64)v18;
    v21 = *v18;
    v4 = (char *)v18 + v6;
    a1[9] = (__int64)v18 + v6;
    a1[3] = v21;
    a1[4] = v21 + 512;
    v22 = *(_QWORD *)((char *)v18 + v6);
    a1[7] = v22;
    a1[8] = v22 + 512;
  }
  *((_QWORD *)v4 + 1) = sub_22077B0(512);
  v10 = a1[6];
  if ( v10 )
  {
    v11 = *a2;
    *(_DWORD *)v10 = 259;
    *(_QWORD *)(v10 + 8) = 0;
    *(_DWORD *)(v10 + 4) = v11;
  }
  v12 = (__int64 *)(a1[9] + 8);
  a1[9] = (__int64)v12;
  result = *v12;
  v14 = *v12 + 512;
  a1[7] = result;
  a1[8] = v14;
  a1[6] = result;
  return result;
}
