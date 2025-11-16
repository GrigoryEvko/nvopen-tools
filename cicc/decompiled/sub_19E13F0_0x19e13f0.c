// Function: sub_19E13F0
// Address: 0x19e13f0
//
_DWORD *__fastcall sub_19E13F0(__int64 a1, __int64 a2, __int64 a3)
{
  int v4; // r15d
  _DWORD *v5; // rax
  _DWORD *v6; // r12
  _DWORD *v7; // rax
  char *v8; // rdx
  _BYTE *v10; // r15
  signed __int64 v11; // r13
  __int64 v12; // rcx
  __int64 v13; // rax
  bool v14; // cf
  unsigned __int64 v15; // rax
  char *v16; // r14
  char *v17; // rcx
  __int64 v18; // r8
  char *v19; // rax
  __int64 v20; // rsi
  __int64 v21; // r14
  __int64 v22; // [rsp+0h] [rbp-40h]
  __int64 v23; // [rsp+8h] [rbp-38h]
  char *v24; // [rsp+8h] [rbp-38h]

  v4 = *(_DWORD *)(a1 + 1464);
  *(_DWORD *)(a1 + 1464) = v4 + 1;
  v5 = (_DWORD *)sub_22077B0(192);
  v6 = v5;
  if ( v5 )
  {
    *v5 = v4;
    v7 = v5 + 24;
    *((_QWORD *)v7 - 11) = a2;
    *((_QWORD *)v7 - 10) = 0;
    *(v7 - 18) = -1;
    *((_QWORD *)v7 - 8) = 0;
    *((_QWORD *)v7 - 7) = 0;
    *((_QWORD *)v7 - 6) = a3;
    *((_QWORD *)v6 + 8) = v7;
    *((_QWORD *)v6 + 9) = v7;
    *((_QWORD *)v6 + 7) = 0;
    *((_QWORD *)v6 + 10) = 4;
    v6[22] = 0;
    *((_QWORD *)v6 + 16) = 0;
    *((_QWORD *)v6 + 17) = v6 + 42;
    *((_QWORD *)v6 + 18) = v6 + 42;
    *((_QWORD *)v6 + 19) = 2;
    v6[40] = 0;
    v6[46] = 0;
  }
  v8 = *(char **)(a1 + 1448);
  if ( v8 == *(char **)(a1 + 1456) )
  {
    v10 = *(_BYTE **)(a1 + 1440);
    v11 = v8 - v10;
    v12 = (v8 - v10) >> 3;
    if ( v12 == 0xFFFFFFFFFFFFFFFLL )
      sub_4262D8((__int64)"vector::_M_realloc_insert");
    v13 = 1;
    if ( v12 )
      v13 = (v8 - v10) >> 3;
    v14 = __CFADD__(v12, v13);
    v15 = v12 + v13;
    if ( v14 )
    {
      v21 = 0x7FFFFFFFFFFFFFF8LL;
    }
    else
    {
      if ( !v15 )
      {
        v16 = 0;
        v17 = 0;
        goto LABEL_14;
      }
      if ( v15 > 0xFFFFFFFFFFFFFFFLL )
        v15 = 0xFFFFFFFFFFFFFFFLL;
      v21 = 8 * v15;
    }
    v17 = (char *)sub_22077B0(v21);
    v16 = &v17[v21];
LABEL_14:
    if ( &v17[v11] )
      *(_QWORD *)&v17[v11] = v6;
    v18 = (__int64)&v17[v11 + 8];
    if ( v11 > 0 )
    {
      v23 = (__int64)&v17[v11 + 8];
      v19 = (char *)memmove(v17, v10, v11);
      v18 = v23;
      v17 = v19;
      v20 = *(_QWORD *)(a1 + 1456) - (_QWORD)v10;
    }
    else
    {
      if ( !v10 )
      {
LABEL_18:
        *(_QWORD *)(a1 + 1440) = v17;
        *(_QWORD *)(a1 + 1448) = v18;
        *(_QWORD *)(a1 + 1456) = v16;
        return v6;
      }
      v20 = *(_QWORD *)(a1 + 1456) - (_QWORD)v10;
    }
    v22 = v18;
    v24 = v17;
    j_j___libc_free_0(v10, v20);
    v18 = v22;
    v17 = v24;
    goto LABEL_18;
  }
  if ( v8 )
  {
    *(_QWORD *)v8 = v6;
    v8 = *(char **)(a1 + 1448);
  }
  *(_QWORD *)(a1 + 1448) = v8 + 8;
  return v6;
}
