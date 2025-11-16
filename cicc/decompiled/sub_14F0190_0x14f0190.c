// Function: sub_14F0190
// Address: 0x14f0190
//
void __fastcall sub_14F0190(__int64 a1, unsigned __int64 a2)
{
  const void *v3; // r8
  __int64 v4; // r15
  unsigned __int64 v5; // r13
  __int64 v6; // rax
  bool v7; // cf
  unsigned __int64 v8; // rax
  signed __int64 v9; // r9
  char *v10; // r14
  const void *v11; // r8
  __int64 v12; // rsi
  __int64 v13; // rdx
  __int64 v14; // rax
  signed __int64 v15; // [rsp-50h] [rbp-50h]
  const void *v16; // [rsp-48h] [rbp-48h]
  __int64 v17; // [rsp-40h] [rbp-40h]
  __int64 v18; // [rsp-40h] [rbp-40h]

  if ( a2 )
  {
    v3 = *(const void **)a1;
    v4 = *(_QWORD *)(a1 + 8) - *(_QWORD *)a1;
    v5 = v4 >> 3;
    if ( (__int64)(*(_QWORD *)(a1 + 16) - *(_QWORD *)(a1 + 8)) >> 3 < a2 )
    {
      if ( 0xFFFFFFFFFFFFFFFLL - v5 < a2 )
        sub_4262D8((__int64)"vector::_M_default_append");
      v6 = a2;
      if ( v5 >= a2 )
        v6 = (__int64)(*(_QWORD *)(a1 + 8) - *(_QWORD *)a1) >> 3;
      v7 = __CFADD__(v5, v6);
      v8 = v5 + v6;
      if ( v7 )
      {
        v13 = 0x7FFFFFFFFFFFFFF8LL;
      }
      else
      {
        if ( !v8 )
        {
          v17 = 0;
          v9 = *(_QWORD *)(a1 + 8) - *(_QWORD *)a1;
          v10 = 0;
          goto LABEL_9;
        }
        if ( v8 > 0xFFFFFFFFFFFFFFFLL )
          v8 = 0xFFFFFFFFFFFFFFFLL;
        v13 = 8 * v8;
      }
      v18 = v13;
      v14 = sub_22077B0(v13);
      v3 = *(const void **)a1;
      v10 = (char *)v14;
      v17 = v18 + v14;
      v9 = *(_QWORD *)(a1 + 8) - *(_QWORD *)a1;
LABEL_9:
      v15 = v9;
      v16 = v3;
      memset(&v10[v4], 0, 8 * a2);
      v11 = v16;
      if ( v15 > 0 )
      {
        memmove(v10, v16, v15);
        v11 = v16;
        v12 = *(_QWORD *)(a1 + 16) - (_QWORD)v16;
      }
      else
      {
        if ( !v16 )
        {
LABEL_11:
          *(_QWORD *)a1 = v10;
          *(_QWORD *)(a1 + 8) = &v10[8 * v5 + 8 * a2];
          *(_QWORD *)(a1 + 16) = v17;
          return;
        }
        v12 = *(_QWORD *)(a1 + 16) - (_QWORD)v16;
      }
      j_j___libc_free_0(v11, v12);
      goto LABEL_11;
    }
    *(_QWORD *)(a1 + 8) = (char *)memset(*(void **)(a1 + 8), 0, 8 * a2) + 8 * a2;
  }
}
