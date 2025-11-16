// Function: sub_34C3A40
// Address: 0x34c3a40
//
unsigned __int64 __fastcall sub_34C3A40(unsigned __int64 *a1, __int64 a2, __int64 a3)
{
  unsigned __int64 v4; // rbx
  unsigned __int64 v5; // r12
  unsigned __int64 v6; // rax
  unsigned __int64 v7; // rdx
  __int64 v8; // r10
  __int64 v9; // r14
  bool v10; // cf
  unsigned __int64 v11; // rax
  __int64 v12; // rdx
  __int64 v13; // r15
  __int64 v14; // rdx
  unsigned __int8 *v15; // rsi
  __int64 v16; // rax
  unsigned __int64 v17; // r15
  unsigned __int64 v18; // r9
  __int64 v19; // rsi
  __int64 v20; // rsi
  __int64 v21; // rax
  unsigned __int64 i; // r14
  __int64 v23; // rsi
  unsigned __int64 v25; // r15
  __int64 v26; // rax
  __int64 v27; // [rsp+8h] [rbp-58h]
  unsigned __int64 v28; // [rsp+10h] [rbp-50h]
  unsigned __int64 v29; // [rsp+18h] [rbp-48h]
  __int64 v30; // [rsp+20h] [rbp-40h]
  __int64 v31; // [rsp+20h] [rbp-40h]
  __int64 v32; // [rsp+28h] [rbp-38h]
  unsigned __int64 v33; // [rsp+28h] [rbp-38h]

  v4 = a1[1];
  v5 = *a1;
  v6 = 0xAAAAAAAAAAAAAAABLL * ((__int64)(v4 - *a1) >> 3);
  if ( v6 == 0x555555555555555LL )
    sub_4262D8((__int64)"vector::_M_realloc_insert");
  v7 = 1;
  v8 = a2;
  v9 = a2;
  if ( v6 )
    v7 = 0xAAAAAAAAAAAAAAABLL * ((__int64)(a1[1] - *a1) >> 3);
  v10 = __CFADD__(v7, v6);
  v11 = v7 - 0x5555555555555555LL * ((__int64)(a1[1] - *a1) >> 3);
  v12 = a2 - v5;
  if ( v10 )
  {
    v25 = 0x7FFFFFFFFFFFFFF8LL;
LABEL_30:
    v27 = a3;
    v26 = sub_22077B0(v25);
    v12 = a2 - v5;
    v8 = a2;
    v29 = v26;
    a3 = v27;
    v28 = v26 + v25;
    v13 = v26 + 24;
    goto LABEL_7;
  }
  if ( v11 )
  {
    if ( v11 > 0x555555555555555LL )
      v11 = 0x555555555555555LL;
    v25 = 24 * v11;
    goto LABEL_30;
  }
  v28 = 0;
  v13 = 24;
  v29 = 0;
LABEL_7:
  v14 = v29 + v12;
  if ( v14 )
  {
    v15 = *(unsigned __int8 **)(a3 + 16);
    *(_DWORD *)v14 = *(_DWORD *)a3;
    v16 = *(_QWORD *)(a3 + 8);
    *(_QWORD *)(v14 + 16) = v15;
    *(_QWORD *)(v14 + 8) = v16;
    if ( v15 )
    {
      v30 = v8;
      v32 = a3;
      sub_B976B0(a3 + 16, v15, v14 + 16);
      v8 = v30;
      *(_QWORD *)(v32 + 16) = 0;
    }
  }
  if ( v8 != v5 )
  {
    v17 = v29;
    v18 = v5;
    while ( 1 )
    {
      if ( v17 )
      {
        *(_DWORD *)v17 = *(_DWORD *)v18;
        *(_QWORD *)(v17 + 8) = *(_QWORD *)(v18 + 8);
        v19 = *(_QWORD *)(v18 + 16);
        *(_QWORD *)(v17 + 16) = v19;
        if ( v19 )
        {
          v31 = v8;
          v33 = v18;
          sub_B96E90(v17 + 16, v19, 1);
          v8 = v31;
          v18 = v33;
        }
      }
      v18 += 24LL;
      if ( v8 == v18 )
        break;
      v17 += 24LL;
    }
    v13 = v17 + 48;
  }
  if ( v8 != v4 )
  {
    do
    {
      v20 = *(_QWORD *)(v9 + 16);
      *(_DWORD *)v13 = *(_DWORD *)v9;
      v21 = *(_QWORD *)(v9 + 8);
      *(_QWORD *)(v13 + 16) = v20;
      *(_QWORD *)(v13 + 8) = v21;
      if ( v20 )
        sub_B96E90(v13 + 16, v20, 1);
      v9 += 24;
      v13 += 24;
    }
    while ( v4 != v9 );
  }
  for ( i = v5; i != v4; i += 24LL )
  {
    v23 = *(_QWORD *)(i + 16);
    if ( v23 )
      sub_B91220(i + 16, v23);
  }
  if ( v5 )
    j_j___libc_free_0(v5);
  a1[1] = v13;
  *a1 = v29;
  a1[2] = v28;
  return v28;
}
