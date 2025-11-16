// Function: sub_22B4880
// Address: 0x22b4880
//
__int64 __fastcall sub_22B4880(unsigned int *a1, __int64 *a2, __int64 a3, __int64 a4, __int64 a5, unsigned __int64 a6)
{
  bool v6; // zf
  __int64 v9; // rsi
  __int64 v10; // rax
  _BYTE *v11; // rsi
  unsigned int v12; // r14d
  unsigned int *v13; // rdx
  __int64 v15; // rax
  const void *v16; // r15
  signed __int64 v17; // r13
  __int64 v18; // rcx
  __int64 v19; // rax
  bool v20; // cf
  unsigned __int64 v21; // rax
  char *v22; // rbx
  char *v23; // rcx
  __int64 v24; // r8
  char *v25; // rax
  unsigned __int64 v26; // rbx
  __int64 v27; // [rsp+0h] [rbp-50h]
  __int64 v28; // [rsp+8h] [rbp-48h]
  char *v29; // [rsp+8h] [rbp-48h]
  __int64 v30[7]; // [rsp+18h] [rbp-38h] BYREF

  v6 = *((_BYTE *)a1 + 72) == 0;
  *((_BYTE *)a1 + 73) = 0;
  if ( !v6 )
    return *a1;
  v30[0] = 0;
  if ( !(_BYTE)a5 )
  {
    v9 = *a2;
    if ( v9 )
      v9 -= 24;
    v10 = sub_22B4800((__int64)a1, v9, 0, *((_QWORD *)a1 + 12), a5, a6);
    v11 = *(_BYTE **)(a4 + 8);
    v30[0] = v10;
    if ( v11 != *(_BYTE **)(a4 + 16) )
      goto LABEL_6;
LABEL_15:
    sub_22B18C0(a4, v11, v30);
    goto LABEL_9;
  }
  v15 = sub_22B4790((__int64)a1, *((_QWORD *)a1 + 12));
  v11 = *(_BYTE **)(a4 + 8);
  v30[0] = v15;
  if ( v11 == *(_BYTE **)(a4 + 16) )
    goto LABEL_15;
LABEL_6:
  if ( v11 )
  {
    *(_QWORD *)v11 = v30[0];
    v11 = *(_BYTE **)(a4 + 8);
  }
  *(_QWORD *)(a4 + 8) = v11 + 8;
LABEL_9:
  v12 = *a1;
  *((_BYTE *)a1 + 72) = 1;
  *a1 = v12 - 1;
  v13 = *(unsigned int **)(a3 + 8);
  if ( v13 == *(unsigned int **)(a3 + 16) )
  {
    v16 = *(const void **)a3;
    v17 = (signed __int64)v13 - *(_QWORD *)a3;
    v18 = v17 >> 2;
    if ( v17 >> 2 == 0x1FFFFFFFFFFFFFFFLL )
      sub_4262D8((__int64)"vector::_M_realloc_insert");
    v19 = 1;
    if ( v18 )
      v19 = v17 >> 2;
    v20 = __CFADD__(v18, v19);
    v21 = v18 + v19;
    if ( v20 )
    {
      v26 = 0x7FFFFFFFFFFFFFFCLL;
    }
    else
    {
      if ( !v21 )
      {
        v22 = 0;
        v23 = 0;
        goto LABEL_23;
      }
      if ( v21 > 0x1FFFFFFFFFFFFFFFLL )
        v21 = 0x1FFFFFFFFFFFFFFFLL;
      v26 = 4 * v21;
    }
    v23 = (char *)sub_22077B0(v26);
    v22 = &v23[v26];
LABEL_23:
    if ( &v23[v17] )
      *(_DWORD *)&v23[v17] = v12;
    v24 = (__int64)&v23[v17 + 4];
    if ( v17 > 0 )
    {
      v28 = (__int64)&v23[v17 + 4];
      v25 = (char *)memmove(v23, v16, v17);
      v24 = v28;
      v23 = v25;
    }
    else if ( !v16 )
    {
LABEL_27:
      *(_QWORD *)a3 = v23;
      *(_QWORD *)(a3 + 8) = v24;
      *(_QWORD *)(a3 + 16) = v22;
      return v12;
    }
    v27 = v24;
    v29 = v23;
    j_j___libc_free_0((unsigned __int64)v16);
    v24 = v27;
    v23 = v29;
    goto LABEL_27;
  }
  if ( v13 )
  {
    *v13 = v12;
    v13 = *(unsigned int **)(a3 + 8);
  }
  *(_QWORD *)(a3 + 8) = v13 + 1;
  return v12;
}
