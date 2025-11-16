// Function: sub_CD7780
// Address: 0xcd7780
//
void __fastcall sub_CD7780(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v5; // r14
  const void *v6; // r15
  size_t v7; // r14
  char *v8; // rbx
  char *v9; // rax
  __int64 v10; // rax
  char v11; // al
  _BOOL8 v12; // rcx
  size_t v13; // rdx
  _BYTE *v14; // rdi
  __int64 v15; // rax
  char v16; // al
  _BOOL8 v17; // rcx
  size_t v18; // rdx
  __int64 v19; // rax
  __int64 v20; // rsi
  unsigned __int64 *v21; // r8
  __int64 v22; // r12
  unsigned __int64 v23; // rax
  unsigned __int64 v24; // rdx
  void *v25; // r8
  _BYTE *v26; // rsi
  char v27; // [rsp+7h] [rbp-79h] BYREF
  __int64 v28; // [rsp+8h] [rbp-78h] BYREF
  void *src; // [rsp+10h] [rbp-70h] BYREF
  char *v30; // [rsp+18h] [rbp-68h]
  char *v31; // [rsp+20h] [rbp-60h]
  void *s2; // [rsp+30h] [rbp-50h] BYREF
  __int64 v33; // [rsp+38h] [rbp-48h]
  __int64 v34; // [rsp+40h] [rbp-40h]

  if ( (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1) )
  {
    v5 = *(_QWORD *)(a3 + 8);
    src = 0;
    v30 = 0;
    v6 = *(const void **)a3;
    v7 = 4 * v5;
    v31 = 0;
    if ( v7 > 0x7FFFFFFFFFFFFFFCLL )
      sub_4262D8((__int64)"cannot create std::vector larger than max_size()");
    v8 = 0;
    if ( v7 )
    {
      v9 = (char *)sub_22077B0(v7);
      v8 = &v9[v7];
      src = v9;
      v31 = &v9[v7];
      memcpy(v9, v6, v7);
    }
    v10 = *(_QWORD *)a1;
    v30 = v8;
    s2 = 0;
    v33 = 0;
    v34 = 0;
    v11 = (*(__int64 (__fastcall **)(__int64))(v10 + 16))(a1);
    v12 = 0;
    if ( v11 )
    {
      v13 = v30 - (_BYTE *)src;
      if ( v30 - (_BYTE *)src == v33 - (_QWORD)s2 )
      {
        v12 = 1;
        if ( v13 )
          v12 = memcmp(src, s2, v13) == 0;
      }
    }
    if ( (*(unsigned __int8 (__fastcall **)(__int64, __int64, _QWORD, _BOOL8, char *, __int64 *))(*(_QWORD *)a1 + 120LL))(
           a1,
           a2,
           0,
           v12,
           &v27,
           &v28) )
    {
      sub_CD6B10(a1, (__int64 *)&src);
      (*(void (__fastcall **)(__int64, __int64))(*(_QWORD *)a1 + 128LL))(a1, v28);
    }
    else if ( v27 )
    {
      sub_CCBC90((__int64)&src, (char **)&s2);
    }
    if ( s2 )
      j_j___libc_free_0(s2, v34 - (_QWORD)s2);
    v14 = src;
    if ( src )
      goto LABEL_13;
  }
  else
  {
    v15 = *(_QWORD *)a1;
    src = 0;
    v30 = 0;
    v31 = 0;
    s2 = 0;
    v33 = 0;
    v34 = 0;
    v16 = (*(__int64 (__fastcall **)(__int64))(v15 + 16))(a1);
    v17 = 0;
    if ( v16 )
    {
      v18 = v30 - (_BYTE *)src;
      if ( v30 - (_BYTE *)src == v33 - (_QWORD)s2 )
      {
        v17 = 1;
        if ( v18 )
          v17 = memcmp(src, s2, v18) == 0;
      }
    }
    if ( (*(unsigned __int8 (__fastcall **)(__int64, __int64, _QWORD, _BOOL8, char *, __int64 *))(*(_QWORD *)a1 + 120LL))(
           a1,
           a2,
           0,
           v17,
           &v27,
           &v28) )
    {
      sub_CD6B10(a1, (__int64 *)&src);
      (*(void (__fastcall **)(__int64, __int64))(*(_QWORD *)a1 + 128LL))(a1, v28);
    }
    else if ( v27 )
    {
      sub_CCBC90((__int64)&src, (char **)&s2);
    }
    if ( s2 )
      j_j___libc_free_0(s2, v34 - (_QWORD)s2);
    v14 = src;
    if ( src == v30 )
    {
      *(_QWORD *)a3 = 0;
      *(_QWORD *)(a3 + 8) = 0;
      if ( v14 )
LABEL_13:
        j_j___libc_free_0(v14, v31 - v14);
    }
    else
    {
      v19 = sub_CB0A70(a1);
      v20 = v30 - (_BYTE *)src;
      v21 = *(unsigned __int64 **)(v19 + 8);
      v22 = (v30 - (_BYTE *)src) >> 2;
      v23 = *v21;
      v21[10] += v30 - (_BYTE *)src;
      v24 = v20 + ((v23 + 3) & 0xFFFFFFFFFFFFFFFCLL);
      if ( v21[1] >= v24 && v23 )
      {
        *v21 = v24;
        v25 = (void *)((v23 + 3) & 0xFFFFFFFFFFFFFFFCLL);
      }
      else
      {
        v25 = (void *)sub_9D1E70((__int64)v21, v20, v20, 2);
      }
      v14 = v30;
      v26 = src;
      *(_QWORD *)a3 = v25;
      *(_QWORD *)(a3 + 8) = v22;
      if ( v26 != v14 )
      {
        memmove(v25, v26, v14 - v26);
        v14 = src;
      }
      if ( v14 )
        goto LABEL_13;
    }
  }
}
