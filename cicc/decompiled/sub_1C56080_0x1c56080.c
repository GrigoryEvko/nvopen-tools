// Function: sub_1C56080
// Address: 0x1c56080
//
__int64 __fastcall sub_1C56080(__int64 *a1, _QWORD *a2)
{
  char *v4; // r14
  char *v5; // rsi
  __int64 v6; // r13
  __int64 v7; // rcx
  __int64 v8; // rdi
  unsigned __int64 v9; // rdx
  _QWORD *v10; // rax
  __int64 *v11; // rdx
  __int64 result; // rax
  __int64 v13; // rdx
  __int64 v14; // r15
  __int64 v15; // rax
  unsigned __int64 v16; // r14
  _QWORD *v17; // r15
  const void *v18; // rsi
  __int64 v19; // rdx
  __int64 v20; // rax
  __int64 v21; // rax
  char *v22; // r14
  size_t v23; // rdx
  __int64 v24; // [rsp+8h] [rbp-38h]

  v4 = (char *)a1[9];
  v5 = (char *)a1[5];
  v6 = v4 - v5;
  v7 = (a1[6] - a1[7]) >> 3;
  v8 = (v4 - v5) >> 3;
  if ( v7 + ((v8 - 1) << 6) + ((a1[4] - a1[2]) >> 3) == 0xFFFFFFFFFFFFFFFLL )
    sub_4262D8((__int64)"cannot create std::deque larger than max_size()");
  v9 = a1[1];
  if ( v9 - ((__int64)&v4[-*a1] >> 3) <= 1 )
  {
    v14 = v8 + 2;
    if ( v9 > 2 * (v8 + 2) )
    {
      v22 = v4 + 8;
      v17 = (_QWORD *)(*a1 + 8 * ((v9 - v14) >> 1));
      v23 = v22 - v5;
      if ( v5 <= (char *)v17 )
      {
        if ( v5 != v22 )
          memmove((char *)v17 + v6 + 8 - v23, v5, v23);
      }
      else if ( v5 != v22 )
      {
        memmove(v17, v5, v23);
      }
    }
    else
    {
      v15 = 1;
      if ( v9 )
        v15 = a1[1];
      v16 = v9 + v15 + 2;
      if ( v16 > 0xFFFFFFFFFFFFFFFLL )
        sub_4261EA(v8, v5, v9);
      v24 = sub_22077B0(8 * v16);
      v17 = (_QWORD *)(v24 + 8 * ((v16 - v14) >> 1));
      v18 = (const void *)a1[5];
      v19 = a1[9] + 8;
      if ( (const void *)v19 != v18 )
        memmove(v17, v18, v19 - (_QWORD)v18);
      j_j___libc_free_0(*a1, 8 * a1[1]);
      a1[1] = v16;
      *a1 = v24;
    }
    a1[5] = (__int64)v17;
    v20 = *v17;
    v4 = (char *)v17 + v6;
    a1[9] = (__int64)v17 + v6;
    a1[3] = v20;
    a1[4] = v20 + 512;
    v21 = *(_QWORD *)((char *)v17 + v6);
    a1[7] = v21;
    a1[8] = v21 + 512;
  }
  *((_QWORD *)v4 + 1) = sub_22077B0(512);
  v10 = (_QWORD *)a1[6];
  if ( v10 )
    *v10 = *a2;
  v11 = (__int64 *)(a1[9] + 8);
  a1[9] = (__int64)v11;
  result = *v11;
  v13 = *v11 + 512;
  a1[7] = result;
  a1[8] = v13;
  a1[6] = result;
  return result;
}
