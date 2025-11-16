// Function: sub_395CDA0
// Address: 0x395cda0
//
__int64 __fastcall sub_395CDA0(__int64 a1, _QWORD *a2)
{
  __int64 result; // rax
  __int64 v5; // rdi
  __int64 *v6; // r13
  __int64 v7; // r15
  __int64 v8; // rsi
  __int64 *v9; // rdx
  __int64 v10; // rdx
  __int64 v11; // rsi
  unsigned __int64 v12; // rax
  __int64 v13; // rcx
  unsigned __int64 v14; // r14
  const void *v15; // rsi
  __int64 v16; // rdx
  __int64 v17; // rax
  __int64 v18; // rax
  char *v19; // rdi
  unsigned __int64 v20; // rax
  unsigned __int64 v21; // r14
  size_t v22; // rdx
  __int64 v23; // [rsp+0h] [rbp-40h]
  __int64 v24; // [rsp+8h] [rbp-38h]

  result = *(_QWORD *)(a1 + 16);
  if ( result != *(_QWORD *)(a1 + 24) )
  {
    *(_QWORD *)(result - 8) = *a2;
    *(_QWORD *)(a1 + 16) -= 8LL;
    return result;
  }
  v5 = *(_QWORD *)(a1 + 72);
  v6 = *(__int64 **)(a1 + 40);
  v7 = v5 - (_QWORD)v6;
  v8 = (v5 - (__int64)v6) >> 3;
  if ( ((__int64)(*(_QWORD *)(a1 + 48) - *(_QWORD *)(a1 + 56)) >> 3)
     + ((v8 - 1) << 6)
     + ((*(_QWORD *)(a1 + 32) - result) >> 3) == 0xFFFFFFFFFFFFFFFLL )
    sub_4262D8((__int64)"cannot create std::deque larger than max_size()");
  if ( v6 == *(__int64 **)a1 )
  {
    v11 = v8 + 2;
    v12 = *(_QWORD *)(a1 + 8);
    if ( v12 <= 2 * v11 )
    {
      v13 = 1;
      if ( v12 )
        v13 = *(_QWORD *)(a1 + 8);
      v14 = v12 + v13 + 2;
      if ( v14 > 0xFFFFFFFFFFFFFFFLL )
        sub_4261EA(v5, v11, 0xFFFFFFFFFFFFFFFLL);
      v23 = v11;
      v24 = sub_22077B0(8 * v14);
      v15 = *(const void **)(a1 + 40);
      v6 = (__int64 *)(v24 + 8 * ((v14 - v23) >> 1) + 8);
      v16 = *(_QWORD *)(a1 + 72) + 8LL;
      if ( (const void *)v16 != v15 )
        memmove(v6, v15, v16 - (_QWORD)v15);
      j_j___libc_free_0(*(_QWORD *)a1);
      *(_QWORD *)(a1 + 8) = v14;
      *(_QWORD *)a1 = v24;
      goto LABEL_13;
    }
    v19 = (char *)(v5 + 8);
    v20 = (v12 - v11) >> 1;
    v21 = (unsigned __int64)&v6[v20 + 1];
    v22 = v19 - (char *)v6;
    if ( (unsigned __int64)v6 <= v21 )
    {
      if ( v6 != (__int64 *)v19 )
      {
        v6 += v20 + 1;
        memmove((void *)(v21 + v7 + 8 - v22), *(const void **)(a1 + 40), v22);
        goto LABEL_13;
      }
    }
    else if ( v6 != (__int64 *)v19 )
    {
      v6 += v20 + 1;
      memmove(v6, *(const void **)(a1 + 40), v22);
LABEL_13:
      *(_QWORD *)(a1 + 40) = v6;
      v17 = *v6;
      *(_QWORD *)(a1 + 72) = (char *)v6 + v7;
      *(_QWORD *)(a1 + 24) = v17;
      *(_QWORD *)(a1 + 32) = v17 + 512;
      v18 = *(__int64 *)((char *)v6 + v7);
      *(_QWORD *)(a1 + 56) = v18;
      *(_QWORD *)(a1 + 64) = v18 + 512;
      goto LABEL_5;
    }
    v6 += v20 + 1;
    goto LABEL_13;
  }
LABEL_5:
  *(v6 - 1) = sub_22077B0(0x200u);
  v9 = (__int64 *)(*(_QWORD *)(a1 + 40) - 8LL);
  *(_QWORD *)(a1 + 40) = v9;
  result = *v9;
  v10 = *v9 + 512;
  *(_QWORD *)(a1 + 24) = result;
  *(_QWORD *)(a1 + 32) = v10;
  *(_QWORD *)(a1 + 16) = result + 504;
  *(_QWORD *)(result + 504) = *a2;
  return result;
}
