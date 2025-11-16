// Function: sub_33E4560
// Address: 0x33e4560
//
void __fastcall sub_33E4560(__int64 a1, unsigned __int64 a2)
{
  const void *v2; // r8
  __int64 v3; // r15
  unsigned __int64 v4; // r13
  __int64 v5; // rax
  bool v6; // cf
  unsigned __int64 v7; // rax
  signed __int64 v8; // r9
  char *v9; // r14
  unsigned __int64 v10; // r8
  unsigned __int64 v11; // rdx
  __int64 v12; // rax
  signed __int64 v13; // [rsp-50h] [rbp-50h]
  const void *v14; // [rsp-48h] [rbp-48h]
  unsigned __int64 v15; // [rsp-40h] [rbp-40h]
  unsigned __int64 v16; // [rsp-40h] [rbp-40h]

  if ( a2 )
  {
    v2 = *(const void **)a1;
    v3 = *(_QWORD *)(a1 + 8) - *(_QWORD *)a1;
    v4 = v3 >> 3;
    if ( a2 > (__int64)(*(_QWORD *)(a1 + 16) - *(_QWORD *)(a1 + 8)) >> 3 )
    {
      if ( 0xFFFFFFFFFFFFFFFLL - v4 < a2 )
        sub_4262D8((__int64)"vector::_M_default_append");
      v5 = (__int64)(*(_QWORD *)(a1 + 8) - *(_QWORD *)a1) >> 3;
      if ( a2 >= v4 )
        v5 = a2;
      v6 = __CFADD__(v4, v5);
      v7 = v4 + v5;
      if ( v6 )
      {
        v11 = 0x7FFFFFFFFFFFFFF8LL;
      }
      else
      {
        if ( !v7 )
        {
          v15 = 0;
          v8 = *(_QWORD *)(a1 + 8) - *(_QWORD *)a1;
          v9 = 0;
          goto LABEL_9;
        }
        if ( v7 > 0xFFFFFFFFFFFFFFFLL )
          v7 = 0xFFFFFFFFFFFFFFFLL;
        v11 = 8 * v7;
      }
      v16 = v11;
      v12 = sub_22077B0(v11);
      v2 = *(const void **)a1;
      v9 = (char *)v12;
      v15 = v16 + v12;
      v8 = *(_QWORD *)(a1 + 8) - *(_QWORD *)a1;
LABEL_9:
      v13 = v8;
      v14 = v2;
      memset(&v9[v3], 0, 8 * a2);
      v10 = (unsigned __int64)v14;
      if ( v13 > 0 )
      {
        memmove(v9, v14, v13);
        v10 = (unsigned __int64)v14;
      }
      else if ( !v14 )
      {
LABEL_11:
        *(_QWORD *)a1 = v9;
        *(_QWORD *)(a1 + 8) = &v9[8 * v4 + 8 * a2];
        *(_QWORD *)(a1 + 16) = v15;
        return;
      }
      j_j___libc_free_0(v10);
      goto LABEL_11;
    }
    *(_QWORD *)(a1 + 8) = (char *)memset(*(void **)(a1 + 8), 0, 8 * a2) + 8 * a2;
  }
}
