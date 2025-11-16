// Function: sub_CD9000
// Address: 0xcd9000
//
void __fastcall sub_CD9000(__int64 a1, __int64 a2, _QWORD *a3)
{
  __int64 v5; // rdx
  _QWORD *v6; // rbx
  __int64 v7; // r14
  _QWORD *v8; // rax
  _QWORD *v9; // rdx
  __int64 v10; // rax
  char v11; // al
  __int64 v12; // rcx
  char *v13; // rax
  char *v14; // rdx
  __int64 v15; // rdx
  _BYTE *v16; // rdi
  __int64 v17; // rax
  char v18; // al
  _BOOL8 v19; // rcx
  __int64 v20; // rdx
  __int64 v21; // rax
  __int64 v22; // rsi
  unsigned __int64 *v23; // r8
  __int64 v24; // rbx
  unsigned __int64 v25; // rax
  unsigned __int64 v26; // rdx
  void *v27; // r8
  _BYTE *v28; // rsi
  char v29; // [rsp+7h] [rbp-69h] BYREF
  __int64 v30; // [rsp+8h] [rbp-68h] BYREF
  void *src; // [rsp+10h] [rbp-60h] BYREF
  char *v32; // [rsp+18h] [rbp-58h]
  char *v33; // [rsp+20h] [rbp-50h]
  char *v34; // [rsp+30h] [rbp-40h] BYREF
  __int64 v35; // [rsp+38h] [rbp-38h]
  __int64 v36; // [rsp+40h] [rbp-30h]

  if ( !(*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1) )
  {
    v17 = *(_QWORD *)a1;
    src = 0;
    v32 = 0;
    v33 = 0;
    v34 = 0;
    v35 = 0;
    v36 = 0;
    v18 = (*(__int64 (__fastcall **)(__int64))(v17 + 16))(a1);
    v19 = 0;
    if ( v18 )
      v19 = v32 - (_BYTE *)src == v35 - (_QWORD)v34;
    if ( (*(unsigned __int8 (__fastcall **)(__int64, __int64, _QWORD, _BOOL8, char *, __int64 *))(*(_QWORD *)a1 + 120LL))(
           a1,
           a2,
           0,
           v19,
           &v29,
           &v30) )
    {
      sub_CD8D20(a1, (__int64)&src);
      (*(void (__fastcall **)(__int64, __int64))(*(_QWORD *)a1 + 128LL))(a1, v30);
    }
    else if ( v29 )
    {
      sub_CCBDF0((__int64)&src, &v34, v20);
    }
    if ( v34 )
      j_j___libc_free_0(v34, v36 - (_QWORD)v34);
    v16 = src;
    if ( src == v32 )
    {
      *a3 = 0;
      a3[1] = 0;
      if ( !v16 )
        return;
    }
    else
    {
      v21 = sub_CB0A70(a1);
      v22 = v32 - (_BYTE *)src;
      v23 = *(unsigned __int64 **)(v21 + 8);
      v24 = (v32 - (_BYTE *)src) >> 3;
      v25 = *v23;
      v23[10] += v32 - (_BYTE *)src;
      v26 = v22 + ((v25 + 3) & 0xFFFFFFFFFFFFFFFCLL);
      if ( v23[1] >= v26 && v25 )
      {
        *v23 = v26;
        v27 = (void *)((v25 + 3) & 0xFFFFFFFFFFFFFFFCLL);
      }
      else
      {
        v27 = (void *)sub_9D1E70((__int64)v23, v22, v22, 2);
      }
      v16 = v32;
      v28 = src;
      *a3 = v27;
      a3[1] = v24;
      if ( v28 != v16 )
      {
        memmove(v27, v28, v16 - v28);
        v16 = src;
      }
      if ( !v16 )
        return;
    }
LABEL_16:
    j_j___libc_free_0(v16, v33 - v16);
    return;
  }
  v5 = a3[1];
  v6 = (_QWORD *)*a3;
  src = 0;
  v32 = 0;
  v7 = v5;
  v33 = 0;
  if ( (unsigned __int64)(8 * v5) > 0x7FFFFFFFFFFFFFF8LL )
    sub_4262D8((__int64)"cannot create std::vector larger than max_size()");
  v8 = 0;
  if ( v7 * 8 )
  {
    v8 = (_QWORD *)sub_22077B0(8 * v5);
    v9 = &v8[v7];
    src = v8;
    v33 = (char *)&v8[v7];
    do
    {
      if ( v8 )
        *v8 = *v6;
      ++v8;
      ++v6;
    }
    while ( v9 != v8 );
  }
  v32 = (char *)v8;
  v10 = *(_QWORD *)a1;
  v34 = 0;
  v35 = 0;
  v36 = 0;
  v11 = (*(__int64 (__fastcall **)(__int64))(v10 + 16))(a1);
  v12 = 0;
  if ( v11 )
  {
    v13 = (char *)src;
    v14 = v34;
    if ( v32 - (_BYTE *)src == v35 - (_QWORD)v34 )
    {
      if ( src == v32 )
      {
LABEL_42:
        v12 = 1;
      }
      else
      {
        while ( *((_DWORD *)v13 + 1) == *((_DWORD *)v14 + 1)
             && ((*(_DWORD *)v14 ^ *(_DWORD *)v13) & 0xFFFFFF) == 0
             && v13[3] == v14[3] )
        {
          v13 += 8;
          v14 += 8;
          if ( v32 == v13 )
            goto LABEL_42;
        }
        v12 = 0;
      }
    }
  }
  if ( (*(unsigned __int8 (__fastcall **)(__int64, __int64, _QWORD, __int64, char *, __int64 *))(*(_QWORD *)a1 + 120LL))(
         a1,
         a2,
         0,
         v12,
         &v29,
         &v30) )
  {
    sub_CD8D20(a1, (__int64)&src);
    (*(void (__fastcall **)(__int64, __int64))(*(_QWORD *)a1 + 128LL))(a1, v30);
  }
  else if ( v29 )
  {
    sub_CCBDF0((__int64)&src, &v34, v15);
  }
  if ( v34 )
    j_j___libc_free_0(v34, v36 - (_QWORD)v34);
  v16 = src;
  if ( src )
    goto LABEL_16;
}
