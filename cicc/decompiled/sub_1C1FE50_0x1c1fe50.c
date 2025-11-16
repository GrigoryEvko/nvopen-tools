// Function: sub_1C1FE50
// Address: 0x1c1fe50
//
void __fastcall sub_1C1FE50(__int64 a1, __int64 a2, __int64 a3)
{
  signed __int64 v4; // r14
  const void *v5; // r15
  char *v6; // rbx
  char *v7; // rax
  __int64 v8; // rax
  char v9; // al
  _BOOL8 v10; // rcx
  size_t v11; // rdx
  __int64 v12; // rax
  char v13; // al
  _BOOL8 v14; // rcx
  size_t v15; // rdx
  _BYTE *v16; // r13
  __int64 v17; // rax
  signed __int64 v18; // r12
  void *v19; // rax
  char *v20; // rdx
  size_t v21; // rdx
  char v22; // [rsp+7h] [rbp-79h] BYREF
  __int64 v23; // [rsp+8h] [rbp-78h] BYREF
  void *src; // [rsp+10h] [rbp-70h] BYREF
  char *v25; // [rsp+18h] [rbp-68h]
  char *v26; // [rsp+20h] [rbp-60h]
  void *s2; // [rsp+30h] [rbp-50h] BYREF
  __int64 v28; // [rsp+38h] [rbp-48h]
  __int64 v29; // [rsp+40h] [rbp-40h]

  if ( (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1) )
  {
    v4 = *(_QWORD *)(a3 + 8);
    src = 0;
    v25 = 0;
    v5 = *(const void **)a3;
    v26 = 0;
    if ( v4 < 0 )
      sub_4262D8((__int64)"cannot create std::vector larger than max_size()");
    v6 = 0;
    if ( v4 )
    {
      v7 = (char *)sub_22077B0(v4);
      v6 = &v7[v4];
      src = v7;
      v26 = &v7[v4];
      memcpy(v7, v5, v4);
    }
    v8 = *(_QWORD *)a1;
    v25 = v6;
    s2 = 0;
    v28 = 0;
    v29 = 0;
    v9 = (*(__int64 (__fastcall **)(__int64))(v8 + 16))(a1);
    v10 = 0;
    if ( v9 )
    {
      v11 = v25 - (_BYTE *)src;
      if ( v25 - (_BYTE *)src == v28 - (_QWORD)s2 )
      {
        v10 = 1;
        if ( v11 )
          v10 = memcmp(src, s2, v11) == 0;
      }
    }
    if ( (*(unsigned __int8 (__fastcall **)(__int64, __int64, _QWORD, _BOOL8, char *, __int64 *))(*(_QWORD *)a1 + 120LL))(
           a1,
           a2,
           0,
           v10,
           &v22,
           &v23) )
    {
      sub_1C1FB60(a1, (__int64 *)&src);
      (*(void (__fastcall **)(__int64, __int64))(*(_QWORD *)a1 + 128LL))(a1, v23);
    }
    else if ( v22 )
    {
      sub_1C13CF0((char **)&src, (const void **)&s2);
    }
    if ( s2 )
      j_j___libc_free_0(s2, v29 - (_QWORD)s2);
    if ( src )
      j_j___libc_free_0(src, v26 - (_BYTE *)src);
  }
  else
  {
    v12 = *(_QWORD *)a1;
    src = 0;
    v25 = 0;
    v26 = 0;
    s2 = 0;
    v28 = 0;
    v29 = 0;
    v13 = (*(__int64 (__fastcall **)(__int64))(v12 + 16))(a1);
    v14 = 0;
    if ( v13 )
    {
      v15 = v25 - (_BYTE *)src;
      if ( v25 - (_BYTE *)src == v28 - (_QWORD)s2 )
      {
        v14 = 1;
        if ( v15 )
          v14 = memcmp(src, s2, v15) == 0;
      }
    }
    if ( (*(unsigned __int8 (__fastcall **)(__int64, __int64, _QWORD, _BOOL8, char *, __int64 *))(*(_QWORD *)a1 + 120LL))(
           a1,
           a2,
           0,
           v14,
           &v22,
           &v23) )
    {
      sub_1C1FB60(a1, (__int64 *)&src);
      (*(void (__fastcall **)(__int64, __int64))(*(_QWORD *)a1 + 128LL))(a1, v23);
    }
    else if ( v22 )
    {
      sub_1C13CF0((char **)&src, (const void **)&s2);
    }
    if ( s2 )
      j_j___libc_free_0(s2, v29 - (_QWORD)s2);
    v16 = src;
    if ( src == v25 )
    {
      *(_QWORD *)a3 = 0;
      *(_QWORD *)(a3 + 8) = 0;
    }
    else
    {
      v17 = sub_16E4080(a1);
      v18 = v25 - (_BYTE *)src;
      v19 = (void *)sub_145CBF0(*(__int64 **)(v17 + 8), v25 - (_BYTE *)src, 1);
      v16 = src;
      v20 = v25;
      *(_QWORD *)(a3 + 8) = v18;
      *(_QWORD *)a3 = v19;
      v21 = v20 - v16;
      if ( v21 )
        memmove(v19, v16, v21);
    }
    if ( v16 )
      j_j___libc_free_0(v16, v26 - v16);
  }
}
