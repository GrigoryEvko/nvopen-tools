// Function: sub_31F7160
// Address: 0x31f7160
//
__int64 __fastcall sub_31F7160(__int64 a1, __int64 a2, unsigned int a3)
{
  _BYTE *v4; // rbx
  _BYTE *v5; // rax
  unsigned __int64 v6; // rdx
  const void *v7; // r14
  size_t v8; // r13
  _QWORD *v9; // rax
  _BYTE *v10; // rdi
  size_t v11; // rcx
  __int64 v12; // rdx
  __int64 v13; // rsi
  unsigned __int64 v14; // rdi
  unsigned __int64 v16; // rdx
  _BYTE *v17; // r14
  size_t v18; // r13
  _QWORD *v19; // rax
  _QWORD *v20; // rdi
  size_t v21; // rdx
  _QWORD *v22; // rdi
  unsigned __int64 v23[2]; // [rsp+10h] [rbp-70h] BYREF
  _QWORD v24[2]; // [rsp+20h] [rbp-60h] BYREF
  _QWORD *v25; // [rsp+30h] [rbp-50h] BYREF
  size_t n; // [rsp+38h] [rbp-48h]
  _QWORD src[8]; // [rsp+40h] [rbp-40h] BYREF

  v4 = (_BYTE *)(a1 + 16);
  *(_QWORD *)a1 = a1 + 16;
  *(_QWORD *)(a1 + 8) = 0;
  *(_BYTE *)(a1 + 16) = 0;
  LODWORD(v25) = 0;
  if ( !a3 )
    return a1;
  if ( a3 <= 0xFFF )
  {
    v23[0] = (unsigned __int64)v24;
    v17 = (_BYTE *)sub_370C330(a3, a3);
    v18 = v16;
    if ( &v17[v16] && !v17 )
LABEL_38:
      sub_426248((__int64)"basic_string::_M_construct null not valid");
    v25 = (_QWORD *)v16;
    if ( v16 > 0xF )
    {
      v23[0] = sub_22409D0((__int64)v23, (unsigned __int64 *)&v25, 0);
      v22 = (_QWORD *)v23[0];
      v24[0] = v25;
    }
    else
    {
      if ( v16 == 1 )
      {
        LOBYTE(v24[0]) = *v17;
        v19 = v24;
        goto LABEL_20;
      }
      if ( !v16 )
      {
        v19 = v24;
        goto LABEL_20;
      }
      v22 = v24;
    }
    memcpy(v22, v17, v18);
    v18 = (size_t)v25;
    v19 = (_QWORD *)v23[0];
LABEL_20:
    v23[1] = v18;
    *((_BYTE *)v19 + v18) = 0;
    sub_2240D70(a1, v23);
    v14 = v23[0];
    if ( (_QWORD *)v23[0] == v24 )
      return a1;
    goto LABEL_13;
  }
  v5 = (_BYTE *)(*(__int64 (__fastcall **)(_QWORD, _QWORD))(**(_QWORD **)(a2 + 16) + 40LL))(*(_QWORD *)(a2 + 16), a3);
  v25 = src;
  v7 = v5;
  v8 = v6;
  if ( &v5[v6] && !v5 )
    goto LABEL_38;
  v23[0] = v6;
  if ( v6 > 0xF )
  {
    v25 = (_QWORD *)sub_22409D0((__int64)&v25, v23, 0);
    v20 = v25;
    src[0] = v23[0];
  }
  else
  {
    if ( v6 == 1 )
    {
      LOBYTE(src[0]) = *v5;
      v9 = src;
      goto LABEL_8;
    }
    if ( !v6 )
    {
      v9 = src;
      goto LABEL_8;
    }
    v20 = src;
  }
  memcpy(v20, v7, v8);
  v8 = v23[0];
  v9 = v25;
LABEL_8:
  n = v8;
  *((_BYTE *)v9 + v8) = 0;
  v10 = *(_BYTE **)a1;
  if ( v25 == src )
  {
    v21 = n;
    if ( n )
    {
      if ( n == 1 )
        *v10 = src[0];
      else
        memcpy(v10, src, n);
      v21 = n;
      v10 = *(_BYTE **)a1;
    }
    *(_QWORD *)(a1 + 8) = v21;
    v10[v21] = 0;
    v10 = v25;
  }
  else
  {
    v11 = n;
    v12 = src[0];
    if ( v4 == v10 )
    {
      *(_QWORD *)a1 = v25;
      *(_QWORD *)(a1 + 8) = v11;
      *(_QWORD *)(a1 + 16) = v12;
    }
    else
    {
      v13 = *(_QWORD *)(a1 + 16);
      *(_QWORD *)a1 = v25;
      *(_QWORD *)(a1 + 8) = v11;
      *(_QWORD *)(a1 + 16) = v12;
      if ( v10 )
      {
        v25 = v10;
        src[0] = v13;
        goto LABEL_12;
      }
    }
    v25 = src;
    v10 = src;
  }
LABEL_12:
  n = 0;
  *v10 = 0;
  v14 = (unsigned __int64)v25;
  if ( v25 != src )
LABEL_13:
    j_j___libc_free_0(v14);
  return a1;
}
