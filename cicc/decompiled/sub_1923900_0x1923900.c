// Function: sub_1923900
// Address: 0x1923900
//
unsigned __int64 __fastcall sub_1923900(unsigned __int64 *a1, __int64 a2, __int64 a3)
{
  unsigned __int64 v4; // r12
  unsigned __int64 v5; // r15
  unsigned __int64 v6; // rax
  unsigned __int64 v7; // rdx
  bool v8; // cf
  unsigned __int64 result; // rax
  __int64 v10; // rdx
  __int64 v11; // r9
  unsigned __int64 v12; // r8
  unsigned __int64 v13; // r14
  __int64 v14; // rdx
  unsigned __int64 v15; // rdx
  unsigned __int64 v16; // rax
  char v17; // cl
  __int64 v18; // rdx
  __int64 v19; // rax
  char v20; // cl
  __int64 v21; // r8
  __int64 v22; // [rsp+8h] [rbp-48h]
  unsigned __int64 v23; // [rsp+10h] [rbp-40h]
  __int64 v24; // [rsp+18h] [rbp-38h]
  __int64 v25; // [rsp+18h] [rbp-38h]

  v4 = a1[1];
  v5 = *a1;
  v6 = 0xAAAAAAAAAAAAAAABLL * ((__int64)(v4 - *a1) >> 3);
  if ( v6 == 0x555555555555555LL )
    sub_4262D8((__int64)"vector::_M_realloc_insert");
  v7 = 1;
  if ( v6 )
    v7 = 0xAAAAAAAAAAAAAAABLL * ((__int64)(a1[1] - *a1) >> 3);
  v8 = __CFADD__(v7, v6);
  result = v7 - 0x5555555555555555LL * ((__int64)(a1[1] - *a1) >> 3);
  v10 = a2 - v5;
  if ( v8 )
  {
    v21 = 0x7FFFFFFFFFFFFFF8LL;
LABEL_27:
    v22 = a3;
    v25 = v21;
    result = sub_22077B0(v21);
    v10 = a2 - v5;
    a3 = v22;
    v13 = result;
    v11 = result + 24;
    v12 = result + v25;
    goto LABEL_7;
  }
  if ( result )
  {
    if ( result > 0x555555555555555LL )
      result = 0x555555555555555LL;
    v21 = 24 * result;
    goto LABEL_27;
  }
  v11 = 24;
  v12 = 0;
  v13 = 0;
LABEL_7:
  v14 = v13 + v10;
  if ( v14 )
  {
    *(_QWORD *)v14 = *(_QWORD *)a3;
    result = *(unsigned __int8 *)(a3 + 16);
    *(_BYTE *)(v14 + 16) = result;
    if ( (_BYTE)result )
    {
      result = *(_QWORD *)(a3 + 8);
      *(_QWORD *)(v14 + 8) = result;
    }
  }
  if ( a2 != v5 )
  {
    v15 = v13;
    v16 = v5;
    do
    {
      if ( v15 )
      {
        *(_QWORD *)v15 = *(_QWORD *)v16;
        v17 = *(_BYTE *)(v16 + 16);
        *(_BYTE *)(v15 + 16) = v17;
        if ( v17 )
          *(_QWORD *)(v15 + 8) = *(_QWORD *)(v16 + 8);
      }
      v16 += 24LL;
      v15 += 24LL;
    }
    while ( a2 != v16 );
    result = (a2 - 24 - v5) >> 3;
    v11 = v13 + 8 * result + 48;
  }
  if ( a2 != v4 )
  {
    v18 = v11;
    v19 = a2;
    do
    {
      *(_QWORD *)v18 = *(_QWORD *)v19;
      v20 = *(_BYTE *)(v19 + 16);
      *(_BYTE *)(v18 + 16) = v20;
      if ( v20 )
        *(_QWORD *)(v18 + 8) = *(_QWORD *)(v19 + 8);
      v19 += 24;
      v18 += 24;
    }
    while ( v4 != v19 );
    result = (v4 - a2 - 24) >> 3;
    v11 += 8 * result + 24;
  }
  if ( v5 )
  {
    v23 = v12;
    v24 = v11;
    result = j_j___libc_free_0(v5, a1[2] - v5);
    v12 = v23;
    v11 = v24;
  }
  *a1 = v13;
  a1[1] = v11;
  a1[2] = v12;
  return result;
}
