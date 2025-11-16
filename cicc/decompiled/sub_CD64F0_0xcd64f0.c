// Function: sub_CD64F0
// Address: 0xcd64f0
//
void *__fastcall sub_CD64F0(__int64 a1, __int64 a2)
{
  void *v4; // rax
  char v5; // al
  _BOOL8 v6; // rcx
  char v7; // al
  _BOOL8 v8; // rcx
  char v9; // al
  _BOOL8 v10; // rcx
  char v11; // al
  _BOOL8 v12; // rcx
  char v13; // al
  _BOOL8 v14; // rcx
  void *result; // rax
  char v16; // al
  __int64 v17; // rax
  void **v18; // rax
  void *v19; // r8
  __int64 v20; // rcx
  void **v21; // r13
  int v22; // eax
  void **v23; // rdx
  unsigned int v24; // ecx
  __int64 v25; // rax
  __int64 v26; // r8
  _BYTE *v27; // rdi
  _BYTE *v28; // r13
  _BYTE *v29; // rax
  size_t v30; // r13
  __int64 *v31; // r8
  __int64 v32; // rax
  char *v33; // rdi
  __int64 v34; // rsi
  unsigned int v35; // [rsp+0h] [rbp-70h]
  __int64 v36; // [rsp+0h] [rbp-70h]
  __int64 v37; // [rsp+8h] [rbp-68h]
  void *v38; // [rsp+8h] [rbp-68h]
  __int64 v39; // [rsp+8h] [rbp-68h]
  char v40; // [rsp+17h] [rbp-59h] BYREF
  __int64 v41; // [rsp+18h] [rbp-58h] BYREF
  void *src; // [rsp+20h] [rbp-50h] BYREF
  void **v43; // [rsp+28h] [rbp-48h]
  void **v44; // [rsp+30h] [rbp-40h]

  v4 = *(void **)a2;
  v43 = (void **)a2;
  src = v4;
  v5 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1);
  v6 = 0;
  if ( v5 )
    v6 = src == 0;
  if ( (*(unsigned __int8 (__fastcall **)(__int64, const char *, _QWORD, _BOOL8, char *, __int64 *))(*(_QWORD *)a1 + 120LL))(
         a1,
         "pgoAppHash",
         0,
         v6,
         &v40,
         &v41) )
  {
    sub_CCC650(a1, (__int64 *)&src);
    (*(void (__fastcall **)(__int64, __int64))(*(_QWORD *)a1 + 128LL))(a1, v41);
  }
  else if ( v40 )
  {
    src = 0;
  }
  if ( !(*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1) )
    *v43 = src;
  src = *(void **)(a2 + 8);
  v43 = (void **)(a2 + 8);
  v7 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1);
  v8 = 0;
  if ( v7 )
    v8 = src == 0;
  if ( (*(unsigned __int8 (__fastcall **)(__int64, const char *, _QWORD, _BOOL8, char *, __int64 *))(*(_QWORD *)a1 + 120LL))(
         a1,
         "pgoProfileHash",
         0,
         v8,
         &v40,
         &v41) )
  {
    sub_CCC650(a1, (__int64 *)&src);
    (*(void (__fastcall **)(__int64, __int64))(*(_QWORD *)a1 + 128LL))(a1, v41);
  }
  else if ( v40 )
  {
    src = 0;
  }
  if ( !(*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1) )
    *v43 = src;
  src = *(void **)(a2 + 16);
  v43 = (void **)(a2 + 16);
  v9 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1);
  v10 = 0;
  if ( v9 )
    v10 = src == 0;
  if ( (*(unsigned __int8 (__fastcall **)(__int64, const char *, _QWORD, _BOOL8, char *, __int64 *))(*(_QWORD *)a1 + 120LL))(
         a1,
         "pgoOptionsHash",
         0,
         v10,
         &v40,
         &v41) )
  {
    sub_CCC650(a1, (__int64 *)&src);
    (*(void (__fastcall **)(__int64, __int64))(*(_QWORD *)a1 + 128LL))(a1, v41);
  }
  else if ( v40 )
  {
    src = 0;
  }
  if ( !(*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1) )
    *v43 = src;
  LODWORD(src) = *(_DWORD *)(a2 + 24);
  v43 = (void **)(a2 + 24);
  v11 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1);
  v12 = 0;
  if ( v11 )
    v12 = (_DWORD)src == 0;
  if ( (*(unsigned __int8 (__fastcall **)(__int64, char *, _QWORD, _BOOL8, char *, __int64 *))(*(_QWORD *)a1 + 120LL))(
         a1,
         "M",
         0,
         v12,
         &v40,
         &v41) )
  {
    sub_CCC2C0(a1, (unsigned int *)&src);
    (*(void (__fastcall **)(__int64, __int64))(*(_QWORD *)a1 + 128LL))(a1, v41);
  }
  else if ( v40 )
  {
    LODWORD(src) = 0;
  }
  if ( !(*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1) )
    *(_DWORD *)v43 = (_DWORD)src;
  LODWORD(src) = *(_DWORD *)(a2 + 28);
  v43 = (void **)(a2 + 28);
  v13 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1);
  v14 = 0;
  if ( v13 )
    v14 = (_DWORD)src == 0;
  if ( (*(unsigned __int8 (__fastcall **)(__int64, const char *, _QWORD, _BOOL8, char *, __int64 *))(*(_QWORD *)a1 + 120LL))(
         a1,
         "numInvocations",
         0,
         v14,
         &v40,
         &v41) )
  {
    sub_CCC2C0(a1, (unsigned int *)&src);
    (*(void (__fastcall **)(__int64, __int64))(*(_QWORD *)a1 + 128LL))(a1, v41);
  }
  else if ( v40 )
  {
    LODWORD(src) = 0;
  }
  result = (void *)(*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1);
  if ( !(_BYTE)result )
  {
    result = v43;
    *(_DWORD *)v43 = (_DWORD)src;
  }
  if ( *(_DWORD *)(a2 + 24) )
  {
    v16 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1);
    src = 0;
    v43 = 0;
    v44 = 0;
    if ( !v16 )
    {
      if ( (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)a1 + 56LL))(a1) && (v28 = src, v43 == src) )
      {
        v29 = src;
      }
      else
      {
        if ( (*(unsigned __int8 (__fastcall **)(__int64, const char *, _QWORD, _QWORD, char *, __int64 *))(*(_QWORD *)a1 + 120LL))(
               a1,
               "BBweightData",
               0,
               0,
               &v40,
               &v41) )
        {
          sub_CD61D0(a1, (__int64 *)&src);
          (*(void (__fastcall **)(__int64, __int64))(*(_QWORD *)a1 + 128LL))(a1, v41);
        }
        v28 = v43;
        v29 = src;
      }
      v30 = v28 - v29;
      v31 = *(__int64 **)(sub_CB0A70(a1) + 8);
      v32 = *v31;
      v31[10] += v30;
      v33 = (char *)((v32 + 3) & 0xFFFFFFFFFFFFFFFCLL);
      if ( v31[1] >= (unsigned __int64)&v33[v30] && v32 )
        *v31 = (__int64)&v33[v30];
      else
        v33 = (char *)sub_9D1E70((__int64)v31, v30, v30, 2);
      *(_QWORD *)(a2 + 32) = v33;
      result = memcpy(v33, src, v30);
      goto LABEL_53;
    }
    v17 = *(unsigned int *)(a2 + 24);
    if ( !*(_DWORD *)(a2 + 24) )
    {
LABEL_49:
      result = (void *)(*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 56LL))(a1);
      if ( (_BYTE)result )
      {
        v27 = src;
        if ( v43 == src )
          goto LABEL_54;
      }
      result = (void *)(*(__int64 (__fastcall **)(__int64, const char *, _QWORD, _QWORD, char *, __int64 *))(*(_QWORD *)a1 + 120LL))(
                         a1,
                         "BBweightData",
                         0,
                         0,
                         &v40,
                         &v41);
      if ( (_BYTE)result )
      {
        sub_CD61D0(a1, (__int64 *)&src);
        result = (void *)(*(__int64 (__fastcall **)(__int64, __int64))(*(_QWORD *)a1 + 128LL))(a1, v41);
      }
LABEL_53:
      v27 = src;
LABEL_54:
      if ( v27 )
        return (void *)j_j___libc_free_0(v27, (char *)v44 - v27);
      return result;
    }
    v37 = 12 * v17;
    v18 = (void **)sub_22077B0(12 * v17);
    v19 = src;
    v20 = v37;
    v21 = v18;
    if ( (char *)v43 - (_BYTE *)src > 0 )
    {
      v36 = v37;
      v38 = src;
      memmove(v18, src, (char *)v43 - (_BYTE *)src);
      v19 = v38;
      v20 = v36;
      v34 = (char *)v44 - (_BYTE *)v38;
    }
    else
    {
      if ( !src )
        goto LABEL_41;
      v34 = (char *)v44 - (_BYTE *)src;
    }
    v39 = v20;
    j_j___libc_free_0(v19, v34);
    v20 = v39;
LABEL_41:
    v22 = *(_DWORD *)(a2 + 24);
    v23 = (void **)((char *)v21 + v20);
    src = v21;
    v43 = v21;
    v44 = (void **)((char *)v21 + v20);
    if ( v22 )
    {
      v24 = 0;
      while ( 1 )
      {
        v25 = *(_QWORD *)(a2 + 32);
        v26 = v25 + 12LL * v24;
        if ( v21 == v23 )
        {
          v35 = v24;
          sub_CD1470((__int64)&src, v21, v25 + 12LL * v24);
          v24 = v35 + 1;
          if ( *(_DWORD *)(a2 + 24) <= v35 + 1 )
            goto LABEL_49;
        }
        else
        {
          if ( v21 )
          {
            *v21 = *(void **)v26;
            *((_DWORD *)v21 + 2) = *(_DWORD *)(v26 + 8);
            v21 = v43;
          }
          ++v24;
          v43 = (void **)((char *)v21 + 12);
          if ( *(_DWORD *)(a2 + 24) <= v24 )
            goto LABEL_49;
        }
        v21 = v43;
        v23 = v44;
      }
    }
    goto LABEL_49;
  }
  return result;
}
