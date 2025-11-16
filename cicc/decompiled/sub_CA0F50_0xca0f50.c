// Function: sub_CA0F50
// Address: 0xca0f50
//
__int64 *__fastcall sub_CA0F50(__int64 *a1, void **p_src)
{
  __int64 v3; // rdi
  __int16 v4; // ax
  bool v5; // zf
  size_t v6; // r13
  void **v7; // r14
  _BYTE *v8; // rdi
  unsigned __int8 v9; // al
  const char *v10; // r14
  _QWORD **v12; // rax
  __int64 v13; // rdi
  _BYTE *v14; // r14
  _BYTE *v15; // r13
  __int64 v16; // rax
  __int64 v17; // rax
  size_t v18; // [rsp+8h] [rbp-148h] BYREF
  void *src; // [rsp+10h] [rbp-140h] BYREF
  size_t n; // [rsp+18h] [rbp-138h]
  __int64 v21; // [rsp+20h] [rbp-130h]
  __int64 v22; // [rsp+28h] [rbp-128h] BYREF
  __int64 v23; // [rsp+30h] [rbp-120h]
  __int64 v24; // [rsp+38h] [rbp-118h]
  __int64 *v25; // [rsp+40h] [rbp-110h]

  v3 = (__int64)p_src;
  v4 = *((_WORD *)p_src + 16);
  if ( v4 == 260 )
  {
    v12 = (_QWORD **)*p_src;
    v13 = (__int64)(a1 + 2);
    *a1 = (__int64)(a1 + 2);
    v14 = *v12;
    v15 = v12[1];
    if ( (_QWORD *)((char *)*v12 + (_QWORD)v15) && !v14 )
      sub_426248((__int64)"basic_string::_M_construct null not valid");
    src = v12[1];
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
LABEL_31:
        a1[1] = (__int64)v15;
        v15[v13] = 0;
        return a1;
      }
      if ( !v15 )
        goto LABEL_31;
    }
    memcpy((void *)v13, v14, (size_t)v15);
    v15 = src;
    v13 = *a1;
    goto LABEL_31;
  }
  if ( v4 != 263 )
  {
    v5 = *((_BYTE *)p_src + 33) == 1;
    n = 0;
    src = &v22;
    v21 = 256;
    if ( !v5 )
    {
LABEL_4:
      p_src = &src;
      sub_CA0EC0(v3, (__int64)&src);
      v6 = n;
      v7 = (void **)src;
      goto LABEL_5;
    }
    v9 = *((_BYTE *)p_src + 32);
    if ( v9 == 1 )
      goto LABEL_22;
    if ( (unsigned __int8)(v9 - 3) > 3u )
      goto LABEL_4;
    if ( v9 == 4 )
    {
      v7 = *(void ***)*p_src;
      v6 = *((_QWORD *)*p_src + 1);
    }
    else
    {
      if ( v9 > 4u )
      {
        if ( (unsigned __int8)(v9 - 5) <= 1u )
        {
          v6 = (size_t)p_src[1];
          v7 = (void **)*p_src;
          goto LABEL_5;
        }
LABEL_44:
        BUG();
      }
      if ( v9 != 3 )
        goto LABEL_44;
      v7 = (void **)*p_src;
      if ( !*p_src )
      {
LABEL_22:
        v8 = a1 + 2;
        goto LABEL_23;
      }
      v6 = strlen((const char *)*p_src);
    }
LABEL_5:
    v8 = a1 + 2;
    if ( v7 )
    {
      *a1 = (__int64)v8;
      v18 = v6;
      if ( v6 > 0xF )
      {
        v16 = sub_22409D0(a1, &v18, 0);
        *a1 = v16;
        v8 = (_BYTE *)v16;
        a1[2] = v18;
      }
      else
      {
        if ( v6 == 1 )
        {
          *((_BYTE *)a1 + 16) = *(_BYTE *)v7;
LABEL_9:
          a1[1] = v6;
          v8[v6] = 0;
          goto LABEL_24;
        }
        if ( !v6 )
          goto LABEL_9;
      }
      p_src = v7;
      memcpy(v8, v7, v6);
      v6 = v18;
      v8 = (_BYTE *)*a1;
      goto LABEL_9;
    }
LABEL_23:
    *a1 = (__int64)v8;
    a1[1] = 0;
    *((_BYTE *)a1 + 16) = 0;
LABEL_24:
    if ( src != &v22 )
      _libc_free(src, p_src);
    return a1;
  }
  v10 = (const char *)*p_src;
  *((_BYTE *)a1 + 16) = 0;
  *a1 = (__int64)(a1 + 2);
  a1[1] = 0;
  v24 = 0x100000000LL;
  src = &unk_49DD210;
  n = 0;
  v21 = 0;
  v22 = 0;
  v23 = 0;
  v25 = a1;
  sub_CB5980(&src, 0, 0, 0);
  sub_CB6840(&src, v10);
  if ( v23 != v21 )
    sub_CB5AE0(&src);
  src = &unk_49DD210;
  sub_CB5840(&src);
  return a1;
}
