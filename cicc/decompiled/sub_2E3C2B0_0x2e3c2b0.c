// Function: sub_2E3C2B0
// Address: 0x2e3c2b0
//
unsigned __int64 __fastcall sub_2E3C2B0(unsigned __int64 *a1, unsigned __int64 a2, __int64 a3)
{
  _QWORD *v3; // r8
  unsigned __int64 v4; // rbx
  __int64 v5; // r12
  __int64 v6; // rcx
  __int64 v7; // rbx
  __int64 v8; // rax
  _QWORD *v9; // rbx
  size_t v10; // rdx
  unsigned __int64 v11; // rax
  unsigned __int64 v12; // r14
  __int64 v13; // rax
  unsigned __int64 v14; // r15
  __int64 v15; // rbx
  const void *v16; // rsi
  __int64 v17; // rdx
  unsigned __int64 v18; // rax
  unsigned __int64 *v19; // rbx
  unsigned __int64 v20; // rax
  unsigned __int64 result; // rax
  char v22; // [rsp+Ch] [rbp-44h]
  __int64 v24; // [rsp+18h] [rbp-38h]

  v3 = (_QWORD *)a1[5];
  v4 = a1[1];
  v5 = a1[9] - (_QWORD)v3;
  v6 = a2 + (v5 >> 3) + 1;
  if ( v4 <= 2 * v6 )
  {
    v11 = a2;
    if ( v4 >= a2 )
      v11 = a1[1];
    v12 = v4 + v11 + 2;
    if ( v12 > 0xFFFFFFFFFFFFFFFLL )
      sub_4261EA(2 * v6, a2, a3);
    v22 = a3;
    v24 = a2 + ((__int64)(a1[9] - (_QWORD)v3) >> 3) + 1;
    v13 = sub_22077B0(8 * v12);
    v14 = v13;
    v15 = 8 * ((v12 - v24) >> 1);
    v16 = (const void *)a1[5];
    if ( v22 )
      v15 += 8 * a2;
    v17 = a1[9] + 8;
    v9 = (_QWORD *)(v13 + v15);
    if ( (const void *)v17 != v16 )
      memmove(v9, v16, v17 - (_QWORD)v16);
    j_j___libc_free_0(*a1);
    *a1 = v14;
    a1[1] = v12;
  }
  else
  {
    v7 = 8 * ((v4 - v6) >> 1);
    if ( (_BYTE)a3 )
      v7 += 8 * a2;
    v8 = a1[9] + 8;
    v9 = (_QWORD *)(*a1 + v7);
    v10 = v8 - (_QWORD)v3;
    if ( v3 <= v9 )
    {
      if ( v3 != (_QWORD *)v8 )
        memmove((char *)v9 + v5 + 8 - v10, v3, v10);
    }
    else if ( v3 != (_QWORD *)v8 )
    {
      memmove(v9, v3, v10);
    }
  }
  a1[5] = (unsigned __int64)v9;
  v18 = *v9;
  v19 = (_QWORD *)((char *)v9 + v5);
  a1[9] = (unsigned __int64)v19;
  a1[3] = v18;
  a1[4] = v18 + 512;
  v20 = *v19;
  a1[7] = *v19;
  result = v20 + 512;
  a1[8] = result;
  return result;
}
