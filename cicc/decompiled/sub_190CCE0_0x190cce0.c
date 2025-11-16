// Function: sub_190CCE0
// Address: 0x190cce0
//
__int64 __fastcall sub_190CCE0(__int64 *a1, int *a2, __int64 a3, __int64 a4)
{
  __int64 v5; // r15
  __int64 v6; // r12
  __int64 v7; // rax
  __int64 v8; // rdi
  int *v9; // r8
  int *v10; // r14
  bool v11; // cf
  unsigned __int64 v12; // rax
  __int64 v13; // r9
  __int64 v14; // rbx
  __int64 v15; // rdi
  __int64 v16; // rbx
  __int64 v17; // rax
  __int64 v18; // rsi
  char v19; // si
  int v20; // eax
  __int64 v21; // rsi
  __int64 v22; // rdi
  __int64 i; // r14
  unsigned __int64 v24; // rdi
  __int64 v26; // rbx
  __int64 v27; // rax
  __int64 v28; // [rsp+8h] [rbp-58h]
  __int64 v29; // [rsp+10h] [rbp-50h]
  int *v30; // [rsp+18h] [rbp-48h]
  __int64 v31; // [rsp+20h] [rbp-40h]
  int *v32; // [rsp+20h] [rbp-40h]
  __int64 v33; // [rsp+28h] [rbp-38h]

  v5 = a1[1];
  v6 = *a1;
  v7 = 0x6DB6DB6DB6DB6DB7LL * ((v5 - *a1) >> 3);
  if ( v7 == 0x249249249249249LL )
    sub_4262D8((__int64)"vector::_M_realloc_insert");
  v8 = 1;
  v9 = a2;
  if ( v7 )
    v8 = 0x6DB6DB6DB6DB6DB7LL * ((v5 - v6) >> 3);
  v10 = a2;
  v11 = __CFADD__(v8, v7);
  v12 = v8 + v7;
  v13 = (__int64)a2 - v6;
  if ( v11 )
  {
    v26 = 0x7FFFFFFFFFFFFFF8LL;
LABEL_31:
    v28 = a3;
    v27 = sub_22077B0(v26);
    v13 = (__int64)a2 - v6;
    v9 = a2;
    v33 = v27;
    a3 = v28;
    v29 = v27 + v26;
    v14 = v27 + 56;
    goto LABEL_7;
  }
  if ( v12 )
  {
    if ( v12 > 0x249249249249249LL )
      v12 = 0x249249249249249LL;
    v26 = 56 * v12;
    goto LABEL_31;
  }
  v29 = 0;
  v14 = 56;
  v33 = 0;
LABEL_7:
  v15 = v33 + v13;
  if ( v33 + v13 )
  {
    a4 = *(unsigned int *)(a3 + 32);
    *(_DWORD *)v15 = *(_DWORD *)a3;
    *(_QWORD *)(v15 + 8) = *(_QWORD *)(a3 + 8);
    *(_BYTE *)(v15 + 16) = *(_BYTE *)(a3 + 16);
    *(_QWORD *)(v15 + 24) = v15 + 40;
    *(_QWORD *)(v15 + 32) = 0x400000000LL;
    if ( (_DWORD)a4 )
    {
      v32 = v9;
      sub_1909410(v15 + 24, a3 + 24, a3, a4, (int)v9, v13);
      v9 = v32;
    }
  }
  if ( v9 != (int *)v6 )
  {
    v16 = v33;
    v17 = v6;
    while ( 1 )
    {
      if ( !v16 )
        goto LABEL_12;
      *(_DWORD *)v16 = *(_DWORD *)v17;
      *(_QWORD *)(v16 + 8) = *(_QWORD *)(v17 + 8);
      v19 = *(_BYTE *)(v17 + 16);
      *(_DWORD *)(v16 + 32) = 0;
      *(_BYTE *)(v16 + 16) = v19;
      *(_QWORD *)(v16 + 24) = v16 + 40;
      *(_DWORD *)(v16 + 36) = 4;
      a3 = *(unsigned int *)(v17 + 32);
      if ( (_DWORD)a3 )
      {
        v30 = v9;
        v31 = v17;
        sub_1909410(v16 + 24, v17 + 24, a3, a4, (int)v9, v13);
        v9 = v30;
        v18 = v16 + 56;
        v17 = v31 + 56;
        if ( v30 == (int *)(v31 + 56) )
        {
LABEL_17:
          v14 = v16 + 112;
          break;
        }
      }
      else
      {
LABEL_12:
        v17 += 56;
        v18 = v16 + 56;
        if ( v9 == (int *)v17 )
          goto LABEL_17;
      }
      v16 = v18;
    }
  }
  if ( v9 != (int *)v5 )
  {
    do
    {
      while ( 1 )
      {
        v20 = *v10;
        *(_DWORD *)(v14 + 32) = 0;
        *(_DWORD *)(v14 + 36) = 4;
        *(_DWORD *)v14 = v20;
        *(_QWORD *)(v14 + 8) = *((_QWORD *)v10 + 1);
        *(_BYTE *)(v14 + 16) = *((_BYTE *)v10 + 16);
        *(_QWORD *)(v14 + 24) = v14 + 40;
        if ( v10[8] )
          break;
        v10 += 14;
        v14 += 56;
        if ( (int *)v5 == v10 )
          goto LABEL_23;
      }
      v21 = (__int64)(v10 + 6);
      v22 = v14 + 24;
      v10 += 14;
      v14 += 56;
      sub_1909410(v22, v21, a3, a4, (int)v9, v13);
    }
    while ( (int *)v5 != v10 );
  }
LABEL_23:
  for ( i = v6; v5 != i; i += 56 )
  {
    v24 = *(_QWORD *)(i + 24);
    if ( v24 != i + 40 )
      _libc_free(v24);
  }
  if ( v6 )
    j_j___libc_free_0(v6, a1[2] - v6);
  a1[1] = v14;
  *a1 = v33;
  a1[2] = v29;
  return v29;
}
