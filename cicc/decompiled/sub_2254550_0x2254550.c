// Function: sub_2254550
// Address: 0x2254550
//
void __fastcall sub_2254550(__int64 a1, _BYTE *a2, _QWORD *a3)
{
  _BYTE *v3; // r14
  _BYTE *v4; // r13
  __int64 v5; // rax
  __int64 v7; // rdx
  bool v8; // cf
  unsigned __int64 v9; // rax
  char *v10; // rcx
  size_t v11; // rdx
  unsigned __int64 v12; // rbx
  char *v13; // r8
  size_t v14; // r9
  char *v15; // r15
  char *v16; // rax
  unsigned __int64 v17; // rbx
  __int64 v18; // rax
  char *dest; // [rsp+0h] [rbp-48h]
  char *desta; // [rsp+0h] [rbp-48h]
  char *destb; // [rsp+0h] [rbp-48h]

  v3 = *(_BYTE **)(a1 + 8);
  v4 = *(_BYTE **)a1;
  v5 = (__int64)&v3[-*(_QWORD *)a1] >> 3;
  if ( v5 == 0xFFFFFFFFFFFFFFFLL )
    sub_4262D8((__int64)"vector::_M_realloc_insert");
  v7 = 1;
  if ( v5 )
    v7 = (__int64)(*(_QWORD *)(a1 + 8) - *(_QWORD *)a1) >> 3;
  v8 = __CFADD__(v7, v5);
  v9 = v7 + v5;
  v10 = (char *)v8;
  v11 = a2 - v4;
  if ( v8 )
  {
    v17 = 0x7FFFFFFFFFFFFFF8LL;
  }
  else
  {
    if ( !v9 )
    {
      v12 = 0;
      goto LABEL_7;
    }
    if ( v9 > 0xFFFFFFFFFFFFFFFLL )
      v9 = 0xFFFFFFFFFFFFFFFLL;
    v17 = 8 * v9;
  }
  v18 = sub_22077B0(v17);
  v11 = a2 - v4;
  v10 = (char *)v18;
  v12 = v18 + v17;
LABEL_7:
  v13 = &v10[v11 + 8];
  v14 = v3 - a2;
  *(_QWORD *)&v10[v11] = *a3;
  v15 = &v13[v3 - a2];
  if ( a2 != v4 )
  {
    dest = &v10[v11 + 8];
    v16 = (char *)memmove(v10, v4, v11);
    v13 = dest;
    v14 = v3 - a2;
    v10 = v16;
    if ( a2 == v3 )
    {
LABEL_12:
      destb = v10;
      j___libc_free_0((unsigned __int64)v4);
      v10 = destb;
      goto LABEL_11;
    }
    goto LABEL_9;
  }
  if ( a2 != v3 )
  {
LABEL_9:
    desta = v10;
    memcpy(v13, a2, v14);
    v10 = desta;
  }
  if ( v4 )
    goto LABEL_12;
LABEL_11:
  *(_QWORD *)a1 = v10;
  *(_QWORD *)(a1 + 8) = v15;
  *(_QWORD *)(a1 + 16) = v12;
}
