// Function: sub_1C1CDE0
// Address: 0x1c1cde0
//
__int64 __fastcall sub_1C1CDE0(__int64 a1, __int64 a2)
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
  __int64 result; // rax
  char v16; // al
  __int64 v17; // rax
  char *v18; // rax
  void *v19; // r8
  __int64 v20; // rcx
  char *v21; // r13
  int v22; // eax
  char *v23; // rsi
  unsigned int v24; // ecx
  __int64 v25; // rdx
  _BYTE *v26; // rdi
  _BYTE *v27; // rax
  _BYTE *v28; // rdx
  size_t v29; // r13
  __int64 v30; // rax
  void *v31; // rax
  void *v32; // r12
  __int64 v33; // rsi
  unsigned int v34; // [rsp+0h] [rbp-70h]
  __int64 v35; // [rsp+0h] [rbp-70h]
  __int64 v36; // [rsp+8h] [rbp-68h]
  void *v37; // [rsp+8h] [rbp-68h]
  __int64 v38; // [rsp+8h] [rbp-68h]
  char v39; // [rsp+17h] [rbp-59h] BYREF
  __int64 v40; // [rsp+18h] [rbp-58h] BYREF
  void *src; // [rsp+20h] [rbp-50h] BYREF
  char *v42; // [rsp+28h] [rbp-48h]
  char *v43; // [rsp+30h] [rbp-40h]

  v4 = *(void **)a2;
  v42 = (char *)a2;
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
         &v39,
         &v40) )
  {
    sub_1C141E0(a1, (__int64 *)&src);
    (*(void (__fastcall **)(__int64, __int64))(*(_QWORD *)a1 + 128LL))(a1, v40);
  }
  else if ( v39 )
  {
    src = 0;
  }
  if ( !(*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1) )
    *(_QWORD *)v42 = src;
  src = *(void **)(a2 + 8);
  v42 = (char *)(a2 + 8);
  v7 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1);
  v8 = 0;
  if ( v7 )
    v8 = src == 0;
  if ( (*(unsigned __int8 (__fastcall **)(__int64, const char *, _QWORD, _BOOL8, char *, __int64 *))(*(_QWORD *)a1 + 120LL))(
         a1,
         "pgoProfileHash",
         0,
         v8,
         &v39,
         &v40) )
  {
    sub_1C141E0(a1, (__int64 *)&src);
    (*(void (__fastcall **)(__int64, __int64))(*(_QWORD *)a1 + 128LL))(a1, v40);
  }
  else if ( v39 )
  {
    src = 0;
  }
  if ( !(*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1) )
    *(_QWORD *)v42 = src;
  src = *(void **)(a2 + 16);
  v42 = (char *)(a2 + 16);
  v9 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1);
  v10 = 0;
  if ( v9 )
    v10 = src == 0;
  if ( (*(unsigned __int8 (__fastcall **)(__int64, const char *, _QWORD, _BOOL8, char *, __int64 *))(*(_QWORD *)a1 + 120LL))(
         a1,
         "pgoOptionsHash",
         0,
         v10,
         &v39,
         &v40) )
  {
    sub_1C141E0(a1, (__int64 *)&src);
    (*(void (__fastcall **)(__int64, __int64))(*(_QWORD *)a1 + 128LL))(a1, v40);
  }
  else if ( v39 )
  {
    src = 0;
  }
  if ( !(*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1) )
    *(_QWORD *)v42 = src;
  LODWORD(src) = *(_DWORD *)(a2 + 24);
  v42 = (char *)(a2 + 24);
  v11 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1);
  v12 = 0;
  if ( v11 )
    v12 = (_DWORD)src == 0;
  if ( (*(unsigned __int8 (__fastcall **)(__int64, char *, _QWORD, _BOOL8, char *, __int64 *))(*(_QWORD *)a1 + 120LL))(
         a1,
         "M",
         0,
         v12,
         &v39,
         &v40) )
  {
    sub_1C14710(a1, (unsigned int *)&src);
    (*(void (__fastcall **)(__int64, __int64))(*(_QWORD *)a1 + 128LL))(a1, v40);
  }
  else if ( v39 )
  {
    LODWORD(src) = 0;
  }
  if ( !(*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1) )
    *(_DWORD *)v42 = (_DWORD)src;
  LODWORD(src) = *(_DWORD *)(a2 + 28);
  v42 = (char *)(a2 + 28);
  v13 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1);
  v14 = 0;
  if ( v13 )
    v14 = (_DWORD)src == 0;
  if ( (*(unsigned __int8 (__fastcall **)(__int64, const char *, _QWORD, _BOOL8, char *, __int64 *))(*(_QWORD *)a1 + 120LL))(
         a1,
         "numInvocations",
         0,
         v14,
         &v39,
         &v40) )
  {
    sub_1C14710(a1, (unsigned int *)&src);
    (*(void (__fastcall **)(__int64, __int64))(*(_QWORD *)a1 + 128LL))(a1, v40);
  }
  else if ( v39 )
  {
    LODWORD(src) = 0;
  }
  result = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1);
  if ( !(_BYTE)result )
  {
    result = (__int64)v42;
    *(_DWORD *)v42 = (_DWORD)src;
  }
  if ( *(_DWORD *)(a2 + 24) )
  {
    v16 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1);
    src = 0;
    v42 = 0;
    v43 = 0;
    if ( v16 )
    {
      v17 = *(unsigned int *)(a2 + 24);
      if ( !*(_DWORD *)(a2 + 24) )
      {
LABEL_49:
        result = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 56LL))(a1);
        if ( !(_BYTE)result || (v26 = src, v42 != src) )
        {
          result = (*(__int64 (__fastcall **)(__int64, const char *, _QWORD, _QWORD, char *, __int64 *))(*(_QWORD *)a1 + 120LL))(
                     a1,
                     "BBweightData",
                     0,
                     0,
                     &v39,
                     &v40);
          if ( (_BYTE)result )
          {
            sub_1C1CAC0(a1, (__int64 *)&src);
            result = (*(__int64 (__fastcall **)(__int64, __int64))(*(_QWORD *)a1 + 128LL))(a1, v40);
          }
          v26 = src;
        }
        if ( v26 )
          return j_j___libc_free_0(v26, v43 - v26);
        return result;
      }
      v36 = 12 * v17;
      v18 = (char *)sub_22077B0(12 * v17);
      v19 = src;
      v20 = v36;
      v21 = v18;
      if ( v42 - (_BYTE *)src > 0 )
      {
        v35 = v36;
        v37 = src;
        memmove(v18, src, v42 - (_BYTE *)src);
        v19 = v37;
        v20 = v35;
        v33 = v43 - (_BYTE *)v37;
      }
      else
      {
        if ( !src )
          goto LABEL_41;
        v33 = v43 - (_BYTE *)src;
      }
      v38 = v20;
      j_j___libc_free_0(v19, v33);
      v20 = v38;
LABEL_41:
      v22 = *(_DWORD *)(a2 + 24);
      v23 = &v21[v20];
      src = v21;
      v42 = v21;
      v43 = &v21[v20];
      if ( v22 )
      {
        v24 = 0;
        while ( 1 )
        {
          v25 = *(_QWORD *)(a2 + 32) + 12LL * v24;
          if ( v23 == v21 )
          {
            v34 = v24;
            sub_CD1470((__int64)&src, v23, v25);
            v24 = v34 + 1;
            if ( *(_DWORD *)(a2 + 24) <= v34 + 1 )
              goto LABEL_49;
          }
          else
          {
            if ( v21 )
            {
              *(_QWORD *)v21 = *(_QWORD *)v25;
              *((_DWORD *)v21 + 2) = *(_DWORD *)(v25 + 8);
              v21 = v42;
            }
            ++v24;
            v42 = v21 + 12;
            if ( *(_DWORD *)(a2 + 24) <= v24 )
              goto LABEL_49;
          }
          v21 = v42;
          v23 = v43;
        }
      }
      goto LABEL_49;
    }
    if ( (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)a1 + 56LL))(a1) && (v27 = src, v42 == src) )
    {
      v28 = src;
    }
    else
    {
      if ( (*(unsigned __int8 (__fastcall **)(__int64, const char *, _QWORD, _QWORD, char *, __int64 *))(*(_QWORD *)a1 + 120LL))(
             a1,
             "BBweightData",
             0,
             0,
             &v39,
             &v40) )
      {
        sub_1C1CAC0(a1, (__int64 *)&src);
        (*(void (__fastcall **)(__int64, __int64))(*(_QWORD *)a1 + 128LL))(a1, v40);
      }
      v27 = v42;
      v28 = src;
    }
    v29 = v27 - v28;
    v30 = sub_16E4080(a1);
    v31 = (void *)sub_145CBF0(*(__int64 **)(v30 + 8), v29, 4);
    v32 = src;
    *(_QWORD *)(a2 + 32) = v31;
    memcpy(v31, v32, v29);
    return j_j___libc_free_0(v32, v43 - (_BYTE *)v32);
  }
  return result;
}
