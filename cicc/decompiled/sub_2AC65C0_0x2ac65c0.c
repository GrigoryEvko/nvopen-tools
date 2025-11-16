// Function: sub_2AC65C0
// Address: 0x2ac65c0
//
void __fastcall sub_2AC65C0(unsigned __int64 *a1, __int64 *a2, __int64 *a3)
{
  unsigned __int64 v3; // r13
  unsigned __int64 v4; // r15
  __int64 v5; // rax
  __int64 v6; // rcx
  __int64 *v8; // rbx
  bool v9; // cf
  unsigned __int64 v10; // rax
  __int64 v11; // r9
  __int64 v12; // rcx
  unsigned __int64 v13; // r12
  unsigned __int64 v14; // r8
  __int64 v15; // r9
  __int64 v16; // rax
  bool v17; // zf
  unsigned __int64 v18; // rax
  unsigned __int64 v19; // rdx
  unsigned __int64 v20; // rcx
  __int64 v21; // rdi
  __int64 v22; // rdi
  __int64 v23; // rax
  __int64 v24; // rdx
  __int64 v25; // rdx
  __int64 v26; // rax
  unsigned __int64 v27; // r12
  __int64 v28; // rax
  __int64 *v29; // [rsp+8h] [rbp-48h]
  __int64 v30; // [rsp+10h] [rbp-40h]
  unsigned __int64 v31; // [rsp+18h] [rbp-38h]

  v3 = a1[1];
  v4 = *a1;
  v5 = (__int64)(v3 - *a1) >> 5;
  if ( v5 == 0x3FFFFFFFFFFFFFFLL )
    sub_4262D8((__int64)"vector::_M_realloc_insert");
  v6 = 1;
  if ( v5 )
    v6 = (__int64)(a1[1] - *a1) >> 5;
  v8 = a2;
  v9 = __CFADD__(v6, v5);
  v10 = v6 + v5;
  v11 = (__int64)a2 - v4;
  if ( v9 )
  {
    v27 = 0x7FFFFFFFFFFFFFE0LL;
LABEL_27:
    v29 = a3;
    v28 = sub_22077B0(v27);
    v11 = (__int64)a2 - v4;
    a3 = v29;
    v14 = v28;
    v13 = v28 + v27;
    v12 = v28 + 32;
    goto LABEL_7;
  }
  if ( v10 )
  {
    if ( v10 > 0x3FFFFFFFFFFFFFFLL )
      v10 = 0x3FFFFFFFFFFFFFFLL;
    v27 = 32 * v10;
    goto LABEL_27;
  }
  v12 = 32;
  v13 = 0;
  v14 = 0;
LABEL_7:
  v15 = v14 + v11;
  if ( v15 )
  {
    v16 = *a3;
    v17 = *((_BYTE *)a3 + 24) == 0;
    *(_BYTE *)(v15 + 24) = 0;
    *(_QWORD *)v15 = v16;
    if ( !v17 )
    {
      v26 = a3[1];
      *(_BYTE *)(v15 + 24) = 1;
      *(_QWORD *)(v15 + 8) = v26;
      *(_QWORD *)(v15 + 16) = a3[2];
    }
  }
  if ( a2 != (__int64 *)v4 )
  {
    v18 = v14;
    v19 = v4;
    v20 = (unsigned __int64)a2 + v14 - v4;
    do
    {
      if ( v18 )
      {
        v21 = *(_QWORD *)v19;
        *(_BYTE *)(v18 + 24) = 0;
        *(_QWORD *)v18 = v21;
        if ( *(_BYTE *)(v19 + 24) )
        {
          *(_QWORD *)(v18 + 8) = *(_QWORD *)(v19 + 8);
          v22 = *(_QWORD *)(v19 + 16);
          *(_BYTE *)(v18 + 24) = 1;
          *(_QWORD *)(v18 + 16) = v22;
        }
      }
      v18 += 32LL;
      v19 += 32LL;
    }
    while ( v20 != v18 );
    v12 = v20 + 32;
  }
  if ( a2 != (__int64 *)v3 )
  {
    v23 = v12;
    do
    {
      v24 = *v8;
      v17 = *((_BYTE *)v8 + 24) == 0;
      *(_BYTE *)(v23 + 24) = 0;
      *(_QWORD *)v23 = v24;
      if ( !v17 )
      {
        v25 = v8[1];
        *(_BYTE *)(v23 + 24) = 1;
        *(_QWORD *)(v23 + 8) = v25;
        *(_QWORD *)(v23 + 16) = v8[2];
      }
      v8 += 4;
      v23 += 32;
    }
    while ( (__int64 *)v3 != v8 );
    v12 += v3 - (_QWORD)a2;
  }
  if ( v4 )
  {
    v30 = v12;
    v31 = v14;
    j_j___libc_free_0(v4);
    v12 = v30;
    v14 = v31;
  }
  *a1 = v14;
  a1[1] = v12;
  a1[2] = v13;
}
