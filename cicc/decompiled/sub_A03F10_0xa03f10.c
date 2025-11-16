// Function: sub_A03F10
// Address: 0xa03f10
//
__int64 __fastcall sub_A03F10(__int64 *a1, int *a2)
{
  char *v4; // r14
  char *v5; // rsi
  __int64 v6; // r13
  __int64 v7; // rcx
  __int64 v8; // rdi
  __int64 v9; // rcx
  unsigned __int64 v10; // rdx
  __int64 v11; // rax
  int v12; // edx
  __int64 *v13; // rdx
  __int64 result; // rax
  __int64 v15; // rdx
  __int64 v16; // r15
  __int64 v17; // rax
  unsigned __int64 v18; // r14
  _QWORD *v19; // r15
  const void *v20; // rsi
  __int64 v21; // rdx
  __int64 v22; // rax
  __int64 v23; // rax
  char *v24; // r14
  size_t v25; // rdx
  __int64 v26; // [rsp+8h] [rbp-38h]

  v4 = (char *)a1[9];
  v5 = (char *)a1[5];
  v6 = v4 - v5;
  v7 = (a1[6] - a1[7]) >> 4;
  v8 = (v4 - v5) >> 3;
  if ( v7 + 32 * (v8 - 1) + ((a1[4] - a1[2]) >> 4) == 0x7FFFFFFFFFFFFFFLL )
    sub_4262D8((__int64)"cannot create std::deque larger than max_size()");
  v9 = *a1;
  v10 = a1[1];
  if ( v10 - ((__int64)&v4[-*a1] >> 3) <= 1 )
  {
    v16 = v8 + 2;
    if ( v10 > 2 * (v8 + 2) )
    {
      v24 = v4 + 8;
      v19 = (_QWORD *)(v9 + 8 * ((v10 - v16) >> 1));
      v25 = v24 - v5;
      if ( v5 <= (char *)v19 )
      {
        if ( v5 != v24 )
          memmove((char *)v19 + v6 + 8 - v25, v5, v25);
      }
      else if ( v5 != v24 )
      {
        memmove(v19, v5, v25);
      }
    }
    else
    {
      v17 = 1;
      if ( v10 )
        v17 = a1[1];
      v18 = v10 + v17 + 2;
      if ( v18 > 0xFFFFFFFFFFFFFFFLL )
        sub_4261EA(v8, v5, v10, v9);
      v26 = sub_22077B0(8 * v18);
      v19 = (_QWORD *)(v26 + 8 * ((v18 - v16) >> 1));
      v20 = (const void *)a1[5];
      v21 = a1[9] + 8;
      if ( (const void *)v21 != v20 )
        memmove(v19, v20, v21 - (_QWORD)v20);
      j_j___libc_free_0(*a1, 8 * a1[1]);
      a1[1] = v18;
      *a1 = v26;
    }
    a1[5] = (__int64)v19;
    v22 = *v19;
    v4 = (char *)v19 + v6;
    a1[9] = (__int64)v19 + v6;
    a1[3] = v22;
    a1[4] = v22 + 512;
    v23 = *(_QWORD *)((char *)v19 + v6);
    a1[7] = v23;
    a1[8] = v23 + 512;
  }
  *((_QWORD *)v4 + 1) = sub_22077B0(512);
  v11 = a1[6];
  if ( v11 )
  {
    v12 = *a2;
    *(_DWORD *)v11 = 259;
    *(_QWORD *)(v11 + 8) = 0;
    *(_DWORD *)(v11 + 4) = v12;
  }
  v13 = (__int64 *)(a1[9] + 8);
  a1[9] = (__int64)v13;
  result = *v13;
  v15 = *v13 + 512;
  a1[7] = result;
  a1[8] = v15;
  a1[6] = result;
  return result;
}
