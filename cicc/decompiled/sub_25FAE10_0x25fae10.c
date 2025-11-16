// Function: sub_25FAE10
// Address: 0x25fae10
//
unsigned __int64 *__fastcall sub_25FAE10(
        unsigned __int64 *a1,
        unsigned __int64 *a2,
        unsigned __int64 *a3,
        unsigned __int64 *a4,
        unsigned __int64 *a5)
{
  unsigned __int64 *v7; // r12
  unsigned __int64 v8; // r13
  unsigned __int64 v9; // r15
  __int64 v10; // rsi
  __int64 v11; // rdi
  unsigned __int64 *v12; // r13
  unsigned __int64 v13; // r12
  unsigned __int64 v14; // r15
  __int64 v15; // rsi
  __int64 v16; // rdi
  unsigned __int64 v18; // r15
  __int64 v19; // rsi
  __int64 v20; // rdi
  unsigned __int64 v21; // r13
  unsigned __int64 v22; // r15
  __int64 v23; // rsi
  __int64 v24; // rdi
  __int64 v25; // rax
  __int64 v26; // [rsp+0h] [rbp-60h]
  unsigned __int64 v27; // [rsp+8h] [rbp-58h]
  __int64 v29; // [rsp+10h] [rbp-50h]
  unsigned __int64 v31; // [rsp+18h] [rbp-48h]
  unsigned __int64 *v32; // [rsp+20h] [rbp-40h]
  unsigned __int64 v33; // [rsp+28h] [rbp-38h]
  unsigned __int64 v34; // [rsp+28h] [rbp-38h]
  unsigned __int64 v35; // [rsp+28h] [rbp-38h]

  if ( a2 != a1 )
  {
    v7 = a1;
    while ( a4 != a3 )
    {
      v8 = a5[1];
      v33 = *a5;
      if ( *(unsigned int *)(*a3 + 4) * 0x86BCA1AF286BCA1BLL * ((__int64)(a3[1] - *a3) >> 3) <= *(unsigned int *)(*v7 + 4)
                                                                                              * 0x86BCA1AF286BCA1BLL
                                                                                              * ((__int64)(v7[1] - *v7) >> 3) )
      {
        *a5 = *v7;
        a5[1] = v7[1];
        a5[2] = v7[2];
        *v7 = 0;
        v7[1] = 0;
        v18 = v33;
        v7[2] = 0;
        while ( v8 != v18 )
        {
          v19 = *(unsigned int *)(v18 + 144);
          v20 = *(_QWORD *)(v18 + 128);
          v18 += 152LL;
          sub_C7D6A0(v20, 8 * v19, 4);
          sub_C7D6A0(*(_QWORD *)(v18 - 56), 8LL * *(unsigned int *)(v18 - 40), 4);
          sub_C7D6A0(*(_QWORD *)(v18 - 88), 16LL * *(unsigned int *)(v18 - 72), 8);
          sub_C7D6A0(*(_QWORD *)(v18 - 120), 16LL * *(unsigned int *)(v18 - 104), 8);
        }
        if ( v33 )
          j_j___libc_free_0(v33);
        v7 += 3;
      }
      else
      {
        *a5 = *a3;
        a5[1] = a3[1];
        a5[2] = a3[2];
        *a3 = 0;
        a3[1] = 0;
        v9 = v33;
        a3[2] = 0;
        while ( v8 != v9 )
        {
          v10 = *(unsigned int *)(v9 + 144);
          v11 = *(_QWORD *)(v9 + 128);
          v9 += 152LL;
          sub_C7D6A0(v11, 8 * v10, 4);
          sub_C7D6A0(*(_QWORD *)(v9 - 56), 8LL * *(unsigned int *)(v9 - 40), 4);
          sub_C7D6A0(*(_QWORD *)(v9 - 88), 16LL * *(unsigned int *)(v9 - 72), 8);
          sub_C7D6A0(*(_QWORD *)(v9 - 120), 16LL * *(unsigned int *)(v9 - 104), 8);
        }
        if ( v33 )
          j_j___libc_free_0(v33);
        a3 += 3;
      }
      a5 += 3;
      if ( a2 == v7 )
        goto LABEL_11;
    }
    v26 = (char *)a2 - (char *)v7;
    v27 = 0xAAAAAAAAAAAAAAABLL * (a2 - v7);
    if ( (char *)a2 - (char *)v7 <= 0 )
      return a5;
    v32 = a5;
    do
    {
      v21 = v32[1];
      v22 = *v32;
      v35 = v22;
      *v32 = *v7;
      v32[1] = v7[1];
      v32[2] = v7[2];
      *v7 = 0;
      v7[1] = 0;
      v7[2] = 0;
      while ( v21 != v22 )
      {
        v23 = *(unsigned int *)(v22 + 144);
        v24 = *(_QWORD *)(v22 + 128);
        v22 += 152LL;
        sub_C7D6A0(v24, 8 * v23, 4);
        sub_C7D6A0(*(_QWORD *)(v22 - 56), 8LL * *(unsigned int *)(v22 - 40), 4);
        sub_C7D6A0(*(_QWORD *)(v22 - 88), 16LL * *(unsigned int *)(v22 - 72), 8);
        sub_C7D6A0(*(_QWORD *)(v22 - 120), 16LL * *(unsigned int *)(v22 - 104), 8);
      }
      if ( v35 )
        j_j___libc_free_0(v35);
      v32 += 3;
      v7 += 3;
      --v27;
    }
    while ( v27 );
    v25 = 24;
    if ( v26 > 0 )
      v25 = v26;
    a5 = (unsigned __int64 *)((char *)a5 + v25);
  }
LABEL_11:
  v29 = (char *)a4 - (char *)a3;
  v31 = 0xAAAAAAAAAAAAAAABLL * (v29 >> 3);
  if ( v29 <= 0 )
    return a5;
  v12 = a5;
  do
  {
    v13 = v12[1];
    v14 = *v12;
    v34 = v14;
    *v12 = *a3;
    v12[1] = a3[1];
    v12[2] = a3[2];
    *a3 = 0;
    a3[1] = 0;
    a3[2] = 0;
    while ( v13 != v14 )
    {
      v15 = *(unsigned int *)(v14 + 144);
      v16 = *(_QWORD *)(v14 + 128);
      v14 += 152LL;
      sub_C7D6A0(v16, 8 * v15, 4);
      sub_C7D6A0(*(_QWORD *)(v14 - 56), 8LL * *(unsigned int *)(v14 - 40), 4);
      sub_C7D6A0(*(_QWORD *)(v14 - 88), 16LL * *(unsigned int *)(v14 - 72), 8);
      sub_C7D6A0(*(_QWORD *)(v14 - 120), 16LL * *(unsigned int *)(v14 - 104), 8);
    }
    if ( v34 )
      j_j___libc_free_0(v34);
    a3 += 3;
    v12 += 3;
    --v31;
  }
  while ( v31 );
  return (unsigned __int64 *)((char *)a5 + v29);
}
