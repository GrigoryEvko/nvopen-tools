// Function: sub_1C1DFB0
// Address: 0x1c1dfb0
//
void __fastcall sub_1C1DFB0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v4; // r14
  const void *v5; // r15
  size_t v6; // r14
  char *v7; // rbx
  char *v8; // rax
  __int64 v9; // rax
  char v10; // al
  _BOOL8 v11; // rcx
  size_t v12; // rdx
  __int64 v13; // rax
  char v14; // al
  _BOOL8 v15; // rcx
  size_t v16; // rdx
  char *v17; // r13
  __int64 v18; // rax
  __int64 v19; // r12
  void *v20; // rax
  char *v21; // rdx
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
    v6 = 4 * v4;
    v26 = 0;
    if ( v6 > 0x7FFFFFFFFFFFFFFCLL )
      sub_4262D8((__int64)"cannot create std::vector larger than max_size()");
    v7 = 0;
    if ( v6 )
    {
      v8 = (char *)sub_22077B0(v6);
      v7 = &v8[v6];
      src = v8;
      v26 = &v8[v6];
      memcpy(v8, v5, v6);
    }
    v9 = *(_QWORD *)a1;
    v25 = v7;
    s2 = 0;
    v28 = 0;
    v29 = 0;
    v10 = (*(__int64 (__fastcall **)(__int64))(v9 + 16))(a1);
    v11 = 0;
    if ( v10 )
    {
      v12 = v25 - (_BYTE *)src;
      if ( v25 - (_BYTE *)src == v28 - (_QWORD)s2 )
      {
        v11 = 1;
        if ( v12 )
          v11 = memcmp(src, s2, v12) == 0;
      }
    }
    if ( (*(unsigned __int8 (__fastcall **)(__int64, __int64, _QWORD, _BOOL8, char *, __int64 *))(*(_QWORD *)a1 + 120LL))(
           a1,
           a2,
           0,
           v11,
           &v22,
           &v23) )
    {
      sub_1C1D3E0(a1, (__int64 *)&src);
      (*(void (__fastcall **)(__int64, __int64))(*(_QWORD *)a1 + 128LL))(a1, v23);
    }
    else if ( v22 )
    {
      sub_1C13A10((__int64)&src, (char **)&s2);
    }
    if ( s2 )
      j_j___libc_free_0(s2, v29 - (_QWORD)s2);
    if ( src )
      j_j___libc_free_0(src, v26 - (_BYTE *)src);
  }
  else
  {
    v13 = *(_QWORD *)a1;
    src = 0;
    v25 = 0;
    v26 = 0;
    s2 = 0;
    v28 = 0;
    v29 = 0;
    v14 = (*(__int64 (__fastcall **)(__int64))(v13 + 16))(a1);
    v15 = 0;
    if ( v14 )
    {
      v16 = v25 - (_BYTE *)src;
      if ( v25 - (_BYTE *)src == v28 - (_QWORD)s2 )
      {
        v15 = 1;
        if ( v16 )
          v15 = memcmp(src, s2, v16) == 0;
      }
    }
    if ( (*(unsigned __int8 (__fastcall **)(__int64, __int64, _QWORD, _BOOL8, char *, __int64 *))(*(_QWORD *)a1 + 120LL))(
           a1,
           a2,
           0,
           v15,
           &v22,
           &v23) )
    {
      sub_1C1D3E0(a1, (__int64 *)&src);
      (*(void (__fastcall **)(__int64, __int64))(*(_QWORD *)a1 + 128LL))(a1, v23);
    }
    else if ( v22 )
    {
      sub_1C13A10((__int64)&src, (char **)&s2);
    }
    if ( s2 )
      j_j___libc_free_0(s2, v29 - (_QWORD)s2);
    v17 = (char *)src;
    if ( src == v25 )
    {
      *(_QWORD *)a3 = 0;
      *(_QWORD *)(a3 + 8) = 0;
    }
    else
    {
      v18 = sub_16E4080(a1);
      v19 = (v25 - (_BYTE *)src) >> 2;
      v20 = (void *)sub_145CBF0(*(__int64 **)(v18 + 8), v25 - (_BYTE *)src, 4);
      v21 = v25;
      v17 = (char *)src;
      *(_QWORD *)(a3 + 8) = v19;
      *(_QWORD *)a3 = v20;
      if ( v17 != v21 )
        memmove(v20, v17, v21 - v17);
    }
    if ( v17 )
      j_j___libc_free_0(v17, v26 - v17);
  }
}
