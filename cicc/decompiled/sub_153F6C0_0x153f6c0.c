// Function: sub_153F6C0
// Address: 0x153f6c0
//
void __fastcall sub_153F6C0(__int64 a1, _QWORD *a2)
{
  _QWORD *v2; // rdx
  const void *v3; // r14
  signed __int64 v4; // r12
  __int64 v5; // rcx
  __int64 v6; // rax
  bool v7; // cf
  unsigned __int64 v8; // rax
  char *v9; // r13
  char *v10; // r15
  __int64 v11; // rcx
  __int64 v12; // rsi
  __int64 v13; // r13
  __int64 v14; // [rsp+8h] [rbp-38h]

  v2 = *(_QWORD **)(a1 + 8);
  if ( v2 == *(_QWORD **)(a1 + 16) )
  {
    v3 = *(const void **)a1;
    v4 = (signed __int64)v2 - *(_QWORD *)a1;
    v5 = v4 >> 3;
    if ( v4 >> 3 == 0xFFFFFFFFFFFFFFFLL )
      sub_4262D8((__int64)"vector::_M_realloc_insert");
    v6 = 1;
    if ( v5 )
      v6 = v4 >> 3;
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
        v9 = 0;
        v10 = 0;
        goto LABEL_11;
      }
      if ( v8 > 0xFFFFFFFFFFFFFFFLL )
        v8 = 0xFFFFFFFFFFFFFFFLL;
      v13 = 8 * v8;
    }
    v10 = (char *)sub_22077B0(v13);
    v9 = &v10[v13];
LABEL_11:
    if ( &v10[v4] )
      *(_QWORD *)&v10[v4] = *a2;
    v11 = (__int64)&v10[v4 + 8];
    if ( v4 > 0 )
    {
      memmove(v10, v3, v4);
      v11 = (__int64)&v10[v4 + 8];
      v12 = *(_QWORD *)(a1 + 16) - (_QWORD)v3;
    }
    else
    {
      if ( !v3 )
      {
LABEL_15:
        *(_QWORD *)a1 = v10;
        *(_QWORD *)(a1 + 16) = v9;
        *(_QWORD *)(a1 + 8) = v11;
        return;
      }
      v12 = *(_QWORD *)(a1 + 16) - (_QWORD)v3;
    }
    v14 = v11;
    j_j___libc_free_0(v3, v12);
    v11 = v14;
    goto LABEL_15;
  }
  if ( v2 )
  {
    *v2 = *a2;
    v2 = *(_QWORD **)(a1 + 8);
  }
  *(_QWORD *)(a1 + 8) = v2 + 1;
}
