// Function: sub_16E2FC0
// Address: 0x16e2fc0
//
__int64 *__fastcall sub_16E2FC0(__int64 *a1, __int64 a2)
{
  __int16 v3; // ax
  bool v4; // zf
  size_t v5; // r13
  _BYTE *v6; // r14
  _BYTE *v7; // rdi
  char v8; // al
  const char *v9; // r15
  const char *v10; // rsi
  const char *v12; // rax
  __int64 v13; // rdi
  _BYTE *v14; // r14
  _BYTE *v15; // r13
  __int64 v16; // rax
  __int64 v17; // rax
  size_t v18; // [rsp+8h] [rbp-148h] BYREF
  void *src; // [rsp+10h] [rbp-140h] BYREF
  size_t n; // [rsp+18h] [rbp-138h]
  __int64 v21; // [rsp+20h] [rbp-130h] BYREF
  __int64 v22; // [rsp+28h] [rbp-128h]
  int v23; // [rsp+30h] [rbp-120h]
  __int64 *v24; // [rsp+38h] [rbp-118h]

  v3 = *(_WORD *)(a2 + 16);
  if ( v3 == 260 )
  {
    v12 = *(const char **)a2;
    v13 = (__int64)(a1 + 2);
    *a1 = (__int64)(a1 + 2);
    v14 = *(_BYTE **)v12;
    v15 = (_BYTE *)*((_QWORD *)v12 + 1);
    if ( &v15[*(_QWORD *)v12] && !v14 )
      sub_426248((__int64)"basic_string::_M_construct null not valid");
    src = (void *)*((_QWORD *)v12 + 1);
    if ( (unsigned __int64)v15 > 0xF )
    {
      v17 = sub_22409D0(a1, &src, 0);
      *a1 = v17;
      v13 = v17;
      a1[2] = (__int64)src;
    }
    else
    {
      if ( v15 == (_BYTE *)1 )
      {
        *((_BYTE *)a1 + 16) = *v14;
LABEL_25:
        a1[1] = (__int64)v15;
        v15[v13] = 0;
        return a1;
      }
      if ( !v15 )
        goto LABEL_25;
    }
    memcpy((void *)v13, v14, (size_t)v15);
    v15 = src;
    v13 = *a1;
    goto LABEL_25;
  }
  if ( v3 != 263 )
  {
    v4 = *(_BYTE *)(a2 + 17) == 1;
    src = &v21;
    n = 0x10000000000LL;
    if ( v4 )
    {
      v8 = *(_BYTE *)(a2 + 16);
      if ( v8 == 1 )
      {
LABEL_16:
        v7 = a1 + 2;
        goto LABEL_17;
      }
      v9 = *(const char **)a2;
      switch ( v8 )
      {
        case 3:
          if ( !v9 )
            goto LABEL_16;
          v6 = *(_BYTE **)a2;
          v5 = strlen(*(const char **)a2);
          break;
        case 4:
        case 5:
          v6 = *(_BYTE **)v9;
          v5 = *((_QWORD *)v9 + 1);
          break;
        case 6:
          v5 = *((unsigned int *)v9 + 2);
          v6 = *(_BYTE **)v9;
          break;
        default:
          goto LABEL_4;
      }
    }
    else
    {
LABEL_4:
      sub_16E2F40(a2, (__int64)&src);
      v5 = (unsigned int)n;
      v6 = src;
    }
    v7 = a1 + 2;
    if ( v6 )
    {
      *a1 = (__int64)v7;
      v18 = v5;
      if ( v5 > 0xF )
      {
        v16 = sub_22409D0(a1, &v18, 0);
        *a1 = v16;
        v7 = (_BYTE *)v16;
        a1[2] = v18;
      }
      else
      {
        if ( v5 == 1 )
        {
          *((_BYTE *)a1 + 16) = *v6;
LABEL_9:
          a1[1] = v5;
          v7[v5] = 0;
          goto LABEL_18;
        }
        if ( !v5 )
          goto LABEL_9;
      }
      memcpy(v7, v6, v5);
      v5 = v18;
      v7 = (_BYTE *)*a1;
      goto LABEL_9;
    }
LABEL_17:
    *a1 = (__int64)v7;
    a1[1] = 0;
    *((_BYTE *)a1 + 16) = 0;
LABEL_18:
    if ( src != &v21 )
      _libc_free((unsigned __int64)src);
    return a1;
  }
  v10 = *(const char **)a2;
  a1[1] = 0;
  *a1 = (__int64)(a1 + 2);
  *((_BYTE *)a1 + 16) = 0;
  v23 = 1;
  v22 = 0;
  v21 = 0;
  n = 0;
  src = &unk_49EFBE0;
  v24 = a1;
  sub_16E8650(&src, v10);
  if ( v22 != n )
    sub_16E7BA0(&src);
  sub_16E7BC0(&src);
  return a1;
}
