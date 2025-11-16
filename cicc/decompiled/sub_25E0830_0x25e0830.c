// Function: sub_25E0830
// Address: 0x25e0830
//
unsigned __int64 __fastcall sub_25E0830(unsigned __int64 *a1)
{
  char *v2; // r13
  char *v3; // rsi
  __int64 v4; // r12
  __int64 v5; // rcx
  __int64 v6; // rdi
  unsigned __int64 v7; // rdx
  unsigned __int64 v8; // rax
  unsigned __int64 *v9; // rdx
  unsigned __int64 result; // rax
  __int64 v11; // rdx
  __int64 v12; // r14
  __int64 v13; // rax
  unsigned __int64 v14; // r13
  __int64 v15; // rax
  const void *v16; // rsi
  unsigned __int64 v17; // r15
  _QWORD *v18; // r14
  __int64 v19; // rdx
  unsigned __int64 v20; // rax
  unsigned __int64 v21; // rax
  char *v22; // r13
  size_t v23; // rdx

  v2 = (char *)a1[9];
  v3 = (char *)a1[5];
  v4 = v2 - v3;
  v5 = (__int64)(a1[6] - a1[7]) >> 5;
  v6 = (v2 - v3) >> 3;
  if ( v5 + 16 * (v6 - 1) + ((__int64)(a1[4] - a1[2]) >> 5) == 0x3FFFFFFFFFFFFFFLL )
    sub_4262D8((__int64)"cannot create std::deque larger than max_size()");
  v7 = a1[1];
  if ( v7 - ((__int64)&v2[-*a1] >> 3) <= 1 )
  {
    v12 = v6 + 2;
    if ( v7 > 2 * (v6 + 2) )
    {
      v22 = v2 + 8;
      v18 = (_QWORD *)(*a1 + 8 * ((v7 - v12) >> 1));
      v23 = v22 - v3;
      if ( v3 <= (char *)v18 )
      {
        if ( v3 != v22 )
          memmove((char *)v18 + v4 + 8 - v23, v3, v23);
      }
      else if ( v3 != v22 )
      {
        memmove(v18, v3, v23);
      }
    }
    else
    {
      v13 = 1;
      if ( v7 )
        v13 = a1[1];
      v14 = v7 + v13 + 2;
      if ( v14 > 0xFFFFFFFFFFFFFFFLL )
        sub_4261EA(v6, v3, v7);
      v15 = sub_22077B0(8 * v14);
      v16 = (const void *)a1[5];
      v17 = v15;
      v18 = (_QWORD *)(v15 + 8 * ((v14 - v12) >> 1));
      v19 = a1[9] + 8;
      if ( (const void *)v19 != v16 )
        memmove(v18, v16, v19 - (_QWORD)v16);
      j_j___libc_free_0(*a1);
      *a1 = v17;
      a1[1] = v14;
    }
    a1[5] = (unsigned __int64)v18;
    v20 = *v18;
    v2 = (char *)v18 + v4;
    a1[9] = (unsigned __int64)v18 + v4;
    a1[3] = v20;
    a1[4] = v20 + 512;
    v21 = *(_QWORD *)((char *)v18 + v4);
    a1[7] = v21;
    a1[8] = v21 + 512;
  }
  *((_QWORD *)v2 + 1) = sub_22077B0(0x200u);
  v8 = a1[6];
  if ( v8 )
  {
    *(_QWORD *)v8 = 0;
    *(_QWORD *)(v8 + 8) = 0;
    *(_QWORD *)(v8 + 16) = 0;
    *(_DWORD *)(v8 + 24) = 0;
  }
  v9 = (unsigned __int64 *)(a1[9] + 8);
  a1[9] = (unsigned __int64)v9;
  result = *v9;
  v11 = *v9 + 512;
  a1[7] = result;
  a1[8] = v11;
  a1[6] = result;
  return result;
}
