// Function: sub_2A5B6F0
// Address: 0x2a5b6f0
//
_QWORD *__fastcall sub_2A5B6F0(unsigned __int64 *a1, _QWORD *a2, int *a3)
{
  _QWORD *v6; // rax
  _QWORD *result; // rax
  char *v8; // r14
  char *v9; // rsi
  __int64 v10; // r15
  __int64 v11; // rdi
  unsigned __int64 v12; // rdx
  _QWORD *v13; // rax
  _QWORD *v14; // rdx
  __int64 v15; // rdx
  __int64 v16; // rax
  unsigned __int64 v17; // r14
  __int64 v18; // rax
  const void *v19; // rsi
  void *v20; // r8
  __int64 v21; // rdx
  char *v22; // r8
  unsigned __int64 v23; // rax
  unsigned __int64 v24; // rax
  char *v25; // r14
  size_t v26; // rdx
  char *v27; // [rsp+0h] [rbp-40h]
  unsigned __int64 v28; // [rsp+8h] [rbp-38h]
  char *v29; // [rsp+8h] [rbp-38h]

  v6 = (_QWORD *)a1[6];
  if ( v6 == (_QWORD *)(a1[8] - 16) )
  {
    v8 = (char *)a1[9];
    v9 = (char *)a1[5];
    v10 = v8 - v9;
    v11 = (v8 - v9) >> 3;
    if ( ((__int64)((__int64)v6 - a1[7]) >> 4) + 32 * (v11 - 1) + ((__int64)(a1[4] - a1[2]) >> 4) == 0x7FFFFFFFFFFFFFFLL )
      sub_4262D8((__int64)"cannot create std::deque larger than max_size()");
    v12 = a1[1];
    if ( v12 - ((__int64)&v8[-*a1] >> 3) <= 1 )
    {
      if ( v12 > 2 * (v11 + 2) )
      {
        v25 = v8 + 8;
        v22 = (char *)(*a1 + 8 * ((v12 - (v11 + 2)) >> 1));
        v26 = v25 - v9;
        if ( v9 <= v22 )
        {
          if ( v9 != v25 )
          {
            v29 = v22;
            memmove(&v22[v10 + 8 - v26], v9, v26);
            v22 = v29;
          }
        }
        else if ( v9 != v25 )
        {
          v22 = (char *)memmove(v22, v9, v26);
        }
      }
      else
      {
        v16 = 1;
        if ( v12 )
          v16 = a1[1];
        v17 = v12 + v16 + 2;
        if ( v17 > 0xFFFFFFFFFFFFFFFLL )
          sub_4261EA(v11, v9, v12);
        v18 = sub_22077B0(8 * v17);
        v19 = (const void *)a1[5];
        v28 = v18;
        v20 = (void *)(v18 + 8 * ((v17 - (v11 + 2)) >> 1));
        v21 = a1[9] + 8;
        if ( (const void *)v21 != v19 )
          v20 = memmove(v20, v19, v21 - (_QWORD)v19);
        v27 = (char *)v20;
        j_j___libc_free_0(*a1);
        a1[1] = v17;
        v22 = v27;
        *a1 = v28;
      }
      a1[5] = (unsigned __int64)v22;
      v23 = *(_QWORD *)v22;
      v8 = &v22[v10];
      a1[9] = (unsigned __int64)&v22[v10];
      a1[3] = v23;
      a1[4] = v23 + 512;
      v24 = *(_QWORD *)&v22[v10];
      a1[7] = v24;
      a1[8] = v24 + 512;
    }
    *((_QWORD *)v8 + 1) = sub_22077B0(0x200u);
    v13 = (_QWORD *)a1[6];
    if ( v13 )
    {
      *v13 = *a2;
      v13[1] = *a3;
    }
    v14 = (_QWORD *)(a1[9] + 8);
    a1[9] = (unsigned __int64)v14;
    result = (_QWORD *)*v14;
    v15 = *v14 + 512LL;
    a1[7] = (unsigned __int64)result;
    a1[8] = v15;
    a1[6] = (unsigned __int64)result;
  }
  else
  {
    if ( v6 )
    {
      *v6 = *a2;
      v6[1] = *a3;
      v6 = (_QWORD *)a1[6];
    }
    result = v6 + 2;
    a1[6] = (unsigned __int64)result;
  }
  return result;
}
