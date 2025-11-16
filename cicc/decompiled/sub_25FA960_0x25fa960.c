// Function: sub_25FA960
// Address: 0x25fa960
//
unsigned __int64 *__fastcall sub_25FA960(
        unsigned __int64 *a1,
        char *a2,
        unsigned __int64 *a3,
        char *a4,
        unsigned __int64 *a5)
{
  unsigned __int64 *v6; // r12
  unsigned __int64 *i; // rbx
  unsigned __int64 v8; // r9
  unsigned __int64 v9; // r13
  unsigned __int64 v10; // r15
  __int64 v11; // rsi
  __int64 v12; // rdi
  unsigned __int64 v13; // r13
  unsigned __int64 v14; // r15
  __int64 v15; // rsi
  __int64 v16; // rdi
  __int64 v17; // rax
  unsigned __int64 *v18; // r13
  unsigned __int64 v19; // rbx
  unsigned __int64 v20; // r15
  __int64 v21; // rsi
  __int64 v22; // rdi
  unsigned __int64 v24; // r15
  __int64 v25; // rsi
  __int64 v26; // rdi
  __int64 v27; // [rsp+0h] [rbp-60h]
  unsigned __int64 v28; // [rsp+8h] [rbp-58h]
  __int64 v30; // [rsp+10h] [rbp-50h]
  unsigned __int64 v32; // [rsp+18h] [rbp-48h]
  unsigned __int64 *v33; // [rsp+20h] [rbp-40h]
  unsigned __int64 v34; // [rsp+28h] [rbp-38h]
  unsigned __int64 v35; // [rsp+28h] [rbp-38h]
  unsigned __int64 v36; // [rsp+28h] [rbp-38h]

  v6 = a3;
  for ( i = a1; a2 != (char *)i && a4 != (char *)v6; a5 += 3 )
  {
    v8 = *a5;
    v9 = a5[1];
    v34 = *a5;
    if ( *(unsigned int *)(*v6 + 4) * 0x86BCA1AF286BCA1BLL * ((__int64)(v6[1] - *v6) >> 3) <= *(unsigned int *)(*i + 4)
                                                                                            * 0x86BCA1AF286BCA1BLL
                                                                                            * ((__int64)(i[1] - *i) >> 3) )
    {
      *a5 = *i;
      a5[1] = i[1];
      a5[2] = i[2];
      *i = 0;
      i[1] = 0;
      v24 = v34;
      i[2] = 0;
      while ( v24 != v9 )
      {
        v25 = *(unsigned int *)(v24 + 144);
        v26 = *(_QWORD *)(v24 + 128);
        v24 += 152LL;
        sub_C7D6A0(v26, 8 * v25, 4);
        sub_C7D6A0(*(_QWORD *)(v24 - 56), 8LL * *(unsigned int *)(v24 - 40), 4);
        sub_C7D6A0(*(_QWORD *)(v24 - 88), 16LL * *(unsigned int *)(v24 - 72), 8);
        sub_C7D6A0(*(_QWORD *)(v24 - 120), 16LL * *(unsigned int *)(v24 - 104), 8);
      }
      if ( v34 )
        j_j___libc_free_0(v34);
      i += 3;
    }
    else
    {
      *a5 = *v6;
      v10 = v8;
      a5[1] = v6[1];
      a5[2] = v6[2];
      *v6 = 0;
      v6[1] = 0;
      v6[2] = 0;
      while ( v10 != v9 )
      {
        v11 = *(unsigned int *)(v10 + 144);
        v12 = *(_QWORD *)(v10 + 128);
        v10 += 152LL;
        sub_C7D6A0(v12, 8 * v11, 4);
        sub_C7D6A0(*(_QWORD *)(v10 - 56), 8LL * *(unsigned int *)(v10 - 40), 4);
        sub_C7D6A0(*(_QWORD *)(v10 - 88), 16LL * *(unsigned int *)(v10 - 72), 8);
        sub_C7D6A0(*(_QWORD *)(v10 - 120), 16LL * *(unsigned int *)(v10 - 104), 8);
      }
      if ( v34 )
        j_j___libc_free_0(v34);
      v6 += 3;
    }
  }
  v27 = a2 - (char *)i;
  v28 = 0xAAAAAAAAAAAAAAABLL * ((a2 - (char *)i) >> 3);
  if ( a2 - (char *)i > 0 )
  {
    v33 = a5;
    do
    {
      v13 = v33[1];
      v14 = *v33;
      v35 = v14;
      *v33 = *i;
      v33[1] = i[1];
      v33[2] = i[2];
      *i = 0;
      i[1] = 0;
      i[2] = 0;
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
      if ( v35 )
        j_j___libc_free_0(v35);
      v33 += 3;
      i += 3;
      --v28;
    }
    while ( v28 );
    v17 = 24;
    if ( v27 > 0 )
      v17 = v27;
    a5 = (unsigned __int64 *)((char *)a5 + v17);
  }
  v30 = a4 - (char *)v6;
  v32 = 0xAAAAAAAAAAAAAAABLL * (v30 >> 3);
  if ( v30 > 0 )
  {
    v18 = a5;
    do
    {
      v19 = v18[1];
      v20 = *v18;
      v36 = v20;
      *v18 = *v6;
      v18[1] = v6[1];
      v18[2] = v6[2];
      *v6 = 0;
      v6[1] = 0;
      v6[2] = 0;
      while ( v19 != v20 )
      {
        v21 = *(unsigned int *)(v20 + 144);
        v22 = *(_QWORD *)(v20 + 128);
        v20 += 152LL;
        sub_C7D6A0(v22, 8 * v21, 4);
        sub_C7D6A0(*(_QWORD *)(v20 - 56), 8LL * *(unsigned int *)(v20 - 40), 4);
        sub_C7D6A0(*(_QWORD *)(v20 - 88), 16LL * *(unsigned int *)(v20 - 72), 8);
        sub_C7D6A0(*(_QWORD *)(v20 - 120), 16LL * *(unsigned int *)(v20 - 104), 8);
      }
      if ( v36 )
        j_j___libc_free_0(v36);
      v6 += 3;
      v18 += 3;
      --v32;
    }
    while ( v32 );
    return (unsigned __int64 *)((char *)a5 + v30);
  }
  return a5;
}
