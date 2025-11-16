// Function: sub_1633130
// Address: 0x1633130
//
_QWORD *__fastcall sub_1633130(_QWORD *a1, __int64 a2)
{
  char *v2; // r13
  void *v4; // r15
  void *v5; // r14
  size_t v6; // rbx
  void *v7; // rcx
  __int64 v8; // rax
  char *v9; // rbx
  void *v10; // [rsp+8h] [rbp-B8h]
  __int64 v11; // [rsp+10h] [rbp-B0h] BYREF
  __int64 v12; // [rsp+18h] [rbp-A8h]
  __int64 v13; // [rsp+20h] [rbp-A0h]
  __int64 v14; // [rsp+28h] [rbp-98h]
  __int64 v15; // [rsp+30h] [rbp-90h]
  __int64 v16; // [rsp+38h] [rbp-88h]
  __int64 v17; // [rsp+40h] [rbp-80h]
  __int64 v18; // [rsp+48h] [rbp-78h]
  __int64 v19; // [rsp+50h] [rbp-70h]
  __int64 v20; // [rsp+58h] [rbp-68h]
  __int64 v21; // [rsp+60h] [rbp-60h]
  __int64 v22; // [rsp+68h] [rbp-58h]
  void *src; // [rsp+70h] [rbp-50h]
  _BYTE *v24; // [rsp+78h] [rbp-48h]
  __int64 v25; // [rsp+80h] [rbp-40h]
  char v26; // [rsp+88h] [rbp-38h]

  v2 = *(char **)(a2 + 168);
  if ( v2 )
  {
    (*(void (__fastcall **)(_QWORD *, _QWORD))(*(_QWORD *)v2 + 48LL))(a1, *(_QWORD *)(a2 + 168));
  }
  else
  {
    v11 = 0;
    v12 = 0;
    v13 = 0;
    v14 = 0;
    v15 = 0;
    v16 = 0;
    v17 = 0;
    v18 = 0;
    v19 = 0;
    v20 = 0;
    v21 = 0;
    v22 = 0;
    src = 0;
    v24 = 0;
    v25 = 0;
    v26 = 0;
    sub_1648080(&v11, a2, 1);
    v4 = v24;
    v5 = src;
    v6 = v24 - (_BYTE *)src;
    if ( v24 == src )
    {
      v9 = 0;
    }
    else
    {
      if ( v6 > 0x7FFFFFFFFFFFFFF8LL )
        sub_4262D8((__int64)"cannot create std::vector larger than max_size()");
      v7 = src;
      if ( v6 )
      {
        v8 = sub_22077B0(v24 - (_BYTE *)src);
        v7 = src;
        v2 = (char *)v8;
      }
      if ( v5 != v4 )
      {
        v10 = v7;
        memcpy(v2, v5, v6);
        v7 = v10;
      }
      v9 = &v2[v6];
      v5 = v7;
    }
    *a1 = v2;
    a1[1] = v9;
    a1[2] = v9;
    if ( v5 )
      j_j___libc_free_0(v5, v25 - (_QWORD)v5);
    j___libc_free_0(v20);
    j___libc_free_0(v16);
    j___libc_free_0(v12);
  }
  return a1;
}
