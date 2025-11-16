// Function: sub_22E2DD0
// Address: 0x22e2dd0
//
__int64 __fastcall sub_22E2DD0(__int64 a1, unsigned __int64 *a2)
{
  _QWORD *v3; // rax
  __int64 result; // rax
  __int64 *v5; // r13
  __int64 *i; // rbx
  __int64 v7; // rdi
  char *v8; // r14
  char *v9; // rsi
  __int64 v10; // r13
  __int64 v11; // rcx
  unsigned __int64 v12; // rdx
  _QWORD *v13; // rax
  __int64 *v14; // rdx
  __int64 v15; // rdx
  __int64 v16; // r15
  __int64 v17; // rax
  unsigned __int64 v18; // r14
  __int64 v19; // rax
  const void *v20; // rsi
  __int64 v21; // rcx
  _QWORD *v22; // r15
  __int64 v23; // rdx
  unsigned __int64 v24; // rax
  unsigned __int64 v25; // rax
  char *v26; // r14
  size_t v27; // rdx
  __int64 v28; // [rsp+8h] [rbp-38h]
  unsigned __int64 v29; // [rsp+8h] [rbp-38h]

  v3 = (_QWORD *)a2[6];
  if ( v3 == (_QWORD *)(a2[8] - 8) )
  {
    v8 = (char *)a2[9];
    v9 = (char *)a2[5];
    v10 = v8 - v9;
    v11 = (v8 - v9) >> 3;
    if ( ((__int64)((__int64)v3 - a2[7]) >> 3) + ((v11 - 1) << 6) + ((__int64)(a2[4] - a2[2]) >> 3) == 0xFFFFFFFFFFFFFFFLL )
      sub_4262D8((__int64)"cannot create std::deque larger than max_size()");
    v12 = a2[1];
    if ( v12 - ((__int64)&v8[-*a2] >> 3) <= 1 )
    {
      v16 = v11 + 2;
      if ( v12 > 2 * (v11 + 2) )
      {
        v26 = v8 + 8;
        v22 = (_QWORD *)(*a2 + 8 * ((v12 - v16) >> 1));
        v27 = v26 - v9;
        if ( v9 <= (char *)v22 )
        {
          if ( v9 != v26 )
            memmove((char *)v22 + v10 + 8 - v27, v9, v27);
        }
        else if ( v9 != v26 )
        {
          memmove(v22, v9, v27);
        }
      }
      else
      {
        v17 = 1;
        if ( v12 )
          v17 = a2[1];
        v18 = v12 + v17 + 2;
        if ( v18 > 0xFFFFFFFFFFFFFFFLL )
          sub_4261EA(0xFFFFFFFFFFFFFFFLL, v9, v12);
        v19 = sub_22077B0(8 * v18);
        v20 = (const void *)a2[5];
        v21 = v19;
        v22 = (_QWORD *)(v19 + 8 * ((v18 - v16) >> 1));
        v23 = a2[9] + 8;
        if ( (const void *)v23 != v20 )
        {
          v28 = v19;
          memmove(v22, v20, v23 - (_QWORD)v20);
          v21 = v28;
        }
        v29 = v21;
        j_j___libc_free_0(*a2);
        a2[1] = v18;
        *a2 = v29;
      }
      a2[5] = (unsigned __int64)v22;
      v24 = *v22;
      v8 = (char *)v22 + v10;
      a2[9] = (unsigned __int64)v22 + v10;
      a2[3] = v24;
      a2[4] = v24 + 512;
      v25 = *(_QWORD *)((char *)v22 + v10);
      a2[7] = v25;
      a2[8] = v25 + 512;
    }
    *((_QWORD *)v8 + 1) = sub_22077B0(0x200u);
    v13 = (_QWORD *)a2[6];
    if ( v13 )
      *v13 = a1;
    v14 = (__int64 *)(a2[9] + 8);
    a2[9] = (unsigned __int64)v14;
    result = *v14;
    v15 = *v14 + 512;
    a2[7] = result;
    a2[8] = v15;
    a2[6] = result;
  }
  else
  {
    if ( v3 )
    {
      *v3 = a1;
      v3 = (_QWORD *)a2[6];
    }
    result = (__int64)(v3 + 1);
    a2[6] = result;
  }
  v5 = *(__int64 **)(a1 + 48);
  for ( i = *(__int64 **)(a1 + 40); v5 != i; result = sub_22E2DD0(v7, a2) )
    v7 = *i++;
  return result;
}
