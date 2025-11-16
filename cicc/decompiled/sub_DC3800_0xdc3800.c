// Function: sub_DC3800
// Address: 0xdc3800
//
__int64 __fastcall sub_DC3800(__int64 a1, __int64 a2, _BYTE *a3, _BYTE *a4)
{
  _BYTE *v5; // rsi
  __int64 v6; // rax
  __int64 *v7; // rbx
  __int64 *v8; // r15
  __int64 v9; // r13
  __int64 v10; // r14
  __int64 v11; // rdx
  __int64 v12; // r15
  __int64 v13; // rbx
  __int64 v14; // rdx
  unsigned int v15; // r12d
  __int64 *i; // rcx
  __int64 *v18; // r14
  __int64 j; // rax
  char v20; // al
  __int64 v24; // [rsp+18h] [rbp-98h]
  __int64 v25; // [rsp+20h] [rbp-90h] BYREF
  __int64 *v26; // [rsp+28h] [rbp-88h]
  __int64 v27; // [rsp+30h] [rbp-80h]
  int v28; // [rsp+38h] [rbp-78h]
  char v29; // [rsp+3Ch] [rbp-74h]
  char v30; // [rsp+40h] [rbp-70h] BYREF

  v26 = (__int64 *)&v30;
  v25 = 0;
  v27 = 8;
  v28 = 0;
  v29 = 1;
  sub_D9A5D0(a1, a3, (__int64)&v25);
  v5 = a4;
  sub_D9A5D0(a1, a4, (__int64)&v25);
  v6 = HIDWORD(v27);
  if ( v28 == HIDWORD(v27) )
    goto LABEL_13;
  v7 = v26;
  if ( !v29 )
    v6 = (unsigned int)v27;
  v8 = &v26[v6];
  v9 = *v26;
  if ( v26 != v8 )
  {
    while ( 1 )
    {
      v9 = *v7;
      if ( (unsigned __int64)*v7 < 0xFFFFFFFFFFFFFFFELL )
        break;
      if ( v8 == v7 + 1 )
      {
        v9 = v7[1];
        goto LABEL_10;
      }
      ++v7;
    }
    if ( v8 != v7 )
    {
LABEL_18:
      for ( i = v7; ; i = v18 )
      {
        v18 = i + 1;
        if ( i + 1 == v8 )
          break;
        for ( j = *v18; (unsigned __int64)*v18 >= 0xFFFFFFFFFFFFFFFELL; j = *v18 )
        {
          if ( v8 == ++v18 )
            goto LABEL_10;
        }
        if ( v8 == v18 )
          break;
        sub_B196A0(*(_QWORD *)(a1 + 40), **(_QWORD **)(v9 + 32), **(_QWORD **)(j + 32));
        if ( v20 )
        {
          v9 = *v18;
          v7 = v18;
          goto LABEL_18;
        }
        v9 = *v7;
      }
    }
  }
LABEL_10:
  v5 = (_BYTE *)v9;
  v10 = sub_DC3660(a1, v9, (__int64)a3);
  v12 = v11;
  if ( sub_D970F0(a1) == v10
    || (v5 = (_BYTE *)v9, v13 = sub_DC3660(a1, v9, (__int64)a4), v24 = v14, sub_D970F0(a1) == v13)
    || (v5 = (_BYTE *)v10, !sub_DAEB70(a1, v10, v9))
    || (v5 = (_BYTE *)v13, !sub_DAEB70(a1, v13, v9))
    || (v5 = (_BYTE *)v9, !(unsigned __int8)sub_DDDA00(a1, v9, a2, v12, v24)) )
  {
LABEL_13:
    v15 = 0;
  }
  else
  {
    v5 = (_BYTE *)v9;
    v15 = sub_DDD5B0(a1, v9, a2, v10, v13);
  }
  if ( !v29 )
    _libc_free(v26, v5);
  return v15;
}
