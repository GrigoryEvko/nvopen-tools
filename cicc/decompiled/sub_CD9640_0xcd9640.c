// Function: sub_CD9640
// Address: 0xcd9640
//
void __fastcall sub_CD9640(__int64 a1, __int64 a2, __int64 a3)
{
  signed __int64 v5; // r14
  const void *v6; // r15
  char *v7; // rbx
  char *v8; // rax
  __int64 v9; // rax
  char v10; // al
  _BOOL8 v11; // rcx
  size_t v12; // rdx
  _BYTE *v13; // rdi
  __int64 v14; // rax
  char v15; // al
  _BOOL8 v16; // rcx
  size_t v17; // rdx
  __int64 v18; // rax
  __int64 v19; // r12
  char **v20; // rdi
  char *v21; // r8
  char *v22; // rdx
  size_t v23; // rdx
  char v24; // [rsp+7h] [rbp-79h] BYREF
  __int64 v25; // [rsp+8h] [rbp-78h] BYREF
  void *src; // [rsp+10h] [rbp-70h] BYREF
  char *v27; // [rsp+18h] [rbp-68h]
  char *v28; // [rsp+20h] [rbp-60h]
  void *s2; // [rsp+30h] [rbp-50h] BYREF
  __int64 v30; // [rsp+38h] [rbp-48h]
  __int64 v31; // [rsp+40h] [rbp-40h]

  if ( (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1) )
  {
    v5 = *(_QWORD *)(a3 + 8);
    src = 0;
    v27 = 0;
    v6 = *(const void **)a3;
    v28 = 0;
    if ( v5 < 0 )
      sub_4262D8((__int64)"cannot create std::vector larger than max_size()");
    v7 = 0;
    if ( v5 )
    {
      v8 = (char *)sub_22077B0(v5);
      v7 = &v8[v5];
      src = v8;
      v28 = &v8[v5];
      memcpy(v8, v6, v5);
    }
    v9 = *(_QWORD *)a1;
    v27 = v7;
    s2 = 0;
    v30 = 0;
    v31 = 0;
    v10 = (*(__int64 (__fastcall **)(__int64))(v9 + 16))(a1);
    v11 = 0;
    if ( v10 )
    {
      v12 = v27 - (_BYTE *)src;
      if ( v27 - (_BYTE *)src == v30 - (_QWORD)s2 )
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
           &v24,
           &v25) )
    {
      sub_CD9530(a1, (__int64 *)&src);
      (*(void (__fastcall **)(__int64, __int64))(*(_QWORD *)a1 + 128LL))(a1, v25);
    }
    else if ( v24 )
    {
      sub_CCBF70((char **)&src, (const void **)&s2);
    }
    if ( s2 )
      j_j___libc_free_0(s2, v31 - (_QWORD)s2);
    v13 = src;
    if ( !src )
      return;
LABEL_28:
    j_j___libc_free_0(v13, v28 - v13);
    return;
  }
  v14 = *(_QWORD *)a1;
  src = 0;
  v27 = 0;
  v28 = 0;
  s2 = 0;
  v30 = 0;
  v31 = 0;
  v15 = (*(__int64 (__fastcall **)(__int64))(v14 + 16))(a1);
  v16 = 0;
  if ( v15 )
  {
    v17 = v27 - (_BYTE *)src;
    if ( v27 - (_BYTE *)src == v30 - (_QWORD)s2 )
    {
      v16 = 1;
      if ( v17 )
        v16 = memcmp(src, s2, v17) == 0;
    }
  }
  if ( (*(unsigned __int8 (__fastcall **)(__int64, __int64, _QWORD, _BOOL8, char *, __int64 *))(*(_QWORD *)a1 + 120LL))(
         a1,
         a2,
         0,
         v16,
         &v24,
         &v25) )
  {
    sub_CD9530(a1, (__int64 *)&src);
    (*(void (__fastcall **)(__int64, __int64))(*(_QWORD *)a1 + 128LL))(a1, v25);
  }
  else if ( v24 )
  {
    sub_CCBF70((char **)&src, (const void **)&s2);
  }
  if ( s2 )
    j_j___libc_free_0(s2, v31 - (_QWORD)s2);
  v13 = src;
  if ( src == v27 )
  {
    *(_QWORD *)a3 = 0;
    *(_QWORD *)(a3 + 8) = 0;
  }
  else
  {
    v18 = sub_CB0A70(a1);
    v19 = v27 - (_BYTE *)src;
    v20 = *(char ***)(v18 + 8);
    v21 = *v20;
    v20[10] += v27 - (_BYTE *)src;
    if ( v20[1] >= &v21[v19] && v21 )
      *v20 = &v21[v19];
    else
      v21 = (char *)sub_9D1E70((__int64)v20, v19, v19, 0);
    v13 = src;
    v22 = v27;
    *(_QWORD *)a3 = v21;
    *(_QWORD *)(a3 + 8) = v19;
    v23 = v22 - v13;
    if ( v23 )
    {
      memmove(v21, v13, v23);
      v13 = src;
    }
  }
  if ( v13 )
    goto LABEL_28;
}
