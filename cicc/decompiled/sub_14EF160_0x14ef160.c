// Function: sub_14EF160
// Address: 0x14ef160
//
_QWORD *__fastcall sub_14EF160(__int64 *a1, _QWORD *a2)
{
  _QWORD *v3; // rax
  _QWORD *result; // rax
  char *v5; // r14
  char *v6; // rsi
  __int64 v7; // r13
  __int64 v8; // rcx
  unsigned __int64 v9; // rdx
  _QWORD *v10; // rax
  _QWORD *v11; // rdx
  __int64 v12; // rdx
  __int64 v13; // r15
  __int64 v14; // rax
  unsigned __int64 v15; // r14
  __int64 v16; // rax
  const void *v17; // rsi
  _QWORD *v18; // r15
  __int64 v19; // rdx
  __int64 v20; // rax
  __int64 v21; // rax
  char *v22; // r14
  size_t v23; // rdx
  __int64 v24; // [rsp+8h] [rbp-38h]

  v3 = (_QWORD *)a1[6];
  if ( v3 == (_QWORD *)(a1[8] - 8) )
  {
    v5 = (char *)a1[9];
    v6 = (char *)a1[5];
    v7 = v5 - v6;
    v8 = (v5 - v6) >> 3;
    if ( (((__int64)v3 - a1[7]) >> 3) + ((v8 - 1) << 6) + ((a1[4] - a1[2]) >> 3) == 0xFFFFFFFFFFFFFFFLL )
      sub_4262D8((__int64)"cannot create std::deque larger than max_size()");
    v9 = a1[1];
    if ( v9 - ((__int64)&v5[-*a1] >> 3) <= 1 )
    {
      v13 = v8 + 2;
      if ( v9 > 2 * (v8 + 2) )
      {
        v22 = v5 + 8;
        v18 = (_QWORD *)(*a1 + 8 * ((v9 - v13) >> 1));
        v23 = v22 - v6;
        if ( v6 <= (char *)v18 )
        {
          if ( v6 != v22 )
            memmove((char *)v18 + v7 + 8 - v23, v6, v23);
        }
        else if ( v6 != v22 )
        {
          memmove(v18, v6, v23);
        }
      }
      else
      {
        v14 = 1;
        if ( v9 )
          v14 = a1[1];
        v15 = v9 + v14 + 2;
        if ( v15 > 0xFFFFFFFFFFFFFFFLL )
          sub_4261EA(0xFFFFFFFFFFFFFFFLL, v6, v9);
        v16 = sub_22077B0(8 * v15);
        v17 = (const void *)a1[5];
        v24 = v16;
        v18 = (_QWORD *)(v16 + 8 * ((v15 - v13) >> 1));
        v19 = a1[9] + 8;
        if ( (const void *)v19 != v17 )
          memmove(v18, v17, v19 - (_QWORD)v17);
        j_j___libc_free_0(*a1, 8 * a1[1]);
        a1[1] = v15;
        *a1 = v24;
      }
      a1[5] = (__int64)v18;
      v20 = *v18;
      v5 = (char *)v18 + v7;
      a1[9] = (__int64)v18 + v7;
      a1[3] = v20;
      a1[4] = v20 + 512;
      v21 = *(_QWORD *)((char *)v18 + v7);
      a1[7] = v21;
      a1[8] = v21 + 512;
    }
    *((_QWORD *)v5 + 1) = sub_22077B0(512);
    v10 = (_QWORD *)a1[6];
    if ( v10 )
      *v10 = *a2;
    v11 = (_QWORD *)(a1[9] + 8);
    a1[9] = (__int64)v11;
    result = (_QWORD *)*v11;
    v12 = *v11 + 512LL;
    a1[7] = (__int64)result;
    a1[8] = v12;
    a1[6] = (__int64)result;
  }
  else
  {
    if ( v3 )
    {
      *v3 = *a2;
      v3 = (_QWORD *)a1[6];
    }
    result = v3 + 1;
    a1[6] = (__int64)result;
  }
  return result;
}
