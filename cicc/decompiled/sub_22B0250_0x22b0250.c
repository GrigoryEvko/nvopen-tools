// Function: sub_22B0250
// Address: 0x22b0250
//
_BYTE *__fastcall sub_22B0250(char **a1, __int64 a2)
{
  _BYTE *result; // rax
  _BYTE *v3; // r13
  char *v4; // r8
  signed __int64 v5; // r12
  char *v6; // r9
  unsigned __int64 v7; // rax
  signed __int64 v8; // rdx
  unsigned __int64 v9; // rsi
  bool v10; // cf
  unsigned __int64 v11; // rax
  char *v12; // r14
  char *v13; // r11
  char *v14; // r15
  char *v15; // r8
  unsigned __int64 v16; // r9
  size_t v17; // rax
  size_t v18; // r12
  char *v19; // r15
  size_t v20; // rdx
  unsigned __int64 v21; // r15
  __int64 v22; // rax
  char *v23; // rdx
  char *dest; // [rsp+8h] [rbp-58h]
  char *src; // [rsp+18h] [rbp-48h]
  void *srca; // [rsp+18h] [rbp-48h]
  unsigned __int64 v27; // [rsp+20h] [rbp-40h]
  unsigned __int64 v28; // [rsp+20h] [rbp-40h]
  char *v29; // [rsp+20h] [rbp-40h]
  unsigned __int64 v30; // [rsp+28h] [rbp-38h]
  char *v31; // [rsp+28h] [rbp-38h]

  result = *(_BYTE **)(a2 + 8);
  v3 = *(_BYTE **)a2;
  v4 = a1[1];
  if ( *(_BYTE **)a2 != result )
  {
    v5 = result - v3;
    if ( a1[2] - v4 >= (unsigned __int64)(result - v3) )
    {
      result = memmove(a1[1], *(const void **)a2, v5);
      a1[1] += v5;
      return result;
    }
    v6 = *a1;
    v7 = v5 >> 3;
    v8 = v4 - *a1;
    v9 = v8 >> 3;
    if ( v5 >> 3 > (unsigned __int64)(0xFFFFFFFFFFFFFFFLL - (v8 >> 3)) )
      sub_4262D8((__int64)"vector::_M_range_insert");
    if ( v7 < v9 )
      v7 = (v4 - *a1) >> 3;
    v10 = __CFADD__(v9, v7);
    v11 = v9 + v7;
    if ( v10 )
    {
      v21 = 0x7FFFFFFFFFFFFFF8LL;
    }
    else
    {
      if ( !v11 )
      {
        v12 = 0;
        v30 = 0;
        v13 = (char *)(v4 - *a1);
        v14 = (char *)(v8 + v5);
        if ( v4 != v6 )
          goto LABEL_11;
        goto LABEL_15;
      }
      if ( v11 > 0xFFFFFFFFFFFFFFFLL )
        v11 = 0xFFFFFFFFFFFFFFFLL;
      v21 = 8 * v11;
    }
    v31 = a1[1];
    v22 = sub_22077B0(v21);
    v4 = v31;
    v6 = *a1;
    v12 = (char *)v22;
    v23 = v31;
    v30 = v21 + v22;
    v8 = v23 - *a1;
    v13 = (char *)(v22 + v8);
    v14 = (char *)(v22 + v8 + v5);
    if ( v4 != *a1 )
    {
LABEL_11:
      src = v4;
      v27 = (unsigned __int64)v6;
      dest = v13;
      memmove(v12, v6, v8);
      memcpy(dest, v3, v5);
      v15 = src;
      v16 = v27;
      v17 = a1[1] - src;
      if ( src == a1[1] )
      {
        v19 = &v14[v17];
        goto LABEL_18;
      }
      goto LABEL_12;
    }
LABEL_15:
    v20 = v5;
    srca = v6;
    v18 = 0;
    v29 = v4;
    memcpy(v13, v3, v20);
    v15 = v29;
    v16 = (unsigned __int64)srca;
    v17 = a1[1] - v29;
    if ( a1[1] == v29 )
    {
LABEL_13:
      v19 = &v14[v18];
      if ( !v16 )
      {
LABEL_14:
        *a1 = v12;
        a1[1] = v19;
        a1[2] = (char *)v30;
        return (_BYTE *)v30;
      }
LABEL_18:
      j_j___libc_free_0(v16);
      goto LABEL_14;
    }
LABEL_12:
    v28 = v16;
    v18 = v17;
    memcpy(v14, v15, v17);
    v16 = v28;
    goto LABEL_13;
  }
  return result;
}
