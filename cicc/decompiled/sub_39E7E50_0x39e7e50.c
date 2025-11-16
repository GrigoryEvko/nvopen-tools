// Function: sub_39E7E50
// Address: 0x39e7e50
//
_BYTE *__fastcall sub_39E7E50(__int64 a1, __int64 a2, __int64 a3, char a4)
{
  char v8; // al
  __int64 *v9; // rdi
  _BYTE *v10; // rax
  __int64 v11; // r15
  __int64 *v12; // rsi
  __int64 *v13; // rdi
  _BYTE *v14; // rax
  __int64 v15; // rdi
  __int64 v16; // rdx
  __int64 (__fastcall *v17)(__int64); // rax
  __int64 *v18; // rdi
  _BYTE *v19; // rax
  unsigned __int64 v20; // r13
  _BYTE *result; // rax
  size_t v22; // rdx
  __int64 v23; // rdi
  __int64 v24; // rdi
  __int64 v25; // r14
  char *v26; // rsi
  size_t v27; // rdx
  void *v28; // rdi
  __int64 *v29; // rax
  char *v30; // [rsp+0h] [rbp-50h] BYREF
  size_t v31; // [rsp+8h] [rbp-48h]
  __int64 v32; // [rsp+10h] [rbp-40h] BYREF

  sub_39E74C0(a1, a2, a3, a4);
  v8 = *(_BYTE *)(a1 + 680);
  if ( (v8 & 2) == 0 )
    goto LABEL_13;
  if ( a4 )
  {
    v9 = (__int64 *)(a1 + 592);
    if ( (v8 & 1) == 0 )
      v9 = sub_16E8D30();
    v10 = (_BYTE *)v9[3];
    if ( (_BYTE *)v9[2] == v10 )
    {
      sub_16E7EE0((__int64)v9, "\n", 1u);
    }
    else
    {
      *v10 = 10;
      ++v9[3];
    }
    v8 = *(_BYTE *)(a1 + 680);
  }
  v11 = *(_QWORD *)(a1 + 288);
  v12 = (__int64 *)(a1 + 592);
  if ( (v8 & 1) == 0 )
    v12 = sub_16E8D30();
  sub_39F1950(a2, v12, v11, "\n ", 2);
  v13 = (__int64 *)(a1 + 592);
  if ( (*(_BYTE *)(a1 + 680) & 1) != 0 )
  {
    v14 = *(_BYTE **)(a1 + 616);
    if ( *(_BYTE **)(a1 + 608) != v14 )
    {
LABEL_12:
      *v14 = 10;
      ++v13[3];
      goto LABEL_13;
    }
  }
  else
  {
    v13 = sub_16E8D30();
    v14 = (_BYTE *)v13[3];
    if ( (_BYTE *)v13[2] != v14 )
      goto LABEL_12;
  }
  sub_16E7EE0((__int64)v13, "\n", 1u);
LABEL_13:
  v15 = *(_QWORD *)(a1 + 16);
  v16 = *(_QWORD *)(a1 + 272);
  if ( v15 )
    (*(void (__fastcall **)(__int64, _QWORD, __int64, __int64, __int64))(*(_QWORD *)v15 + 32LL))(
      v15,
      *(_QWORD *)(a1 + 288),
      v16,
      a2,
      a3);
  else
    (*(void (__fastcall **)(_QWORD, __int64, __int64, const char *, _QWORD, __int64))(**(_QWORD **)(a1 + 288) + 16LL))(
      *(_QWORD *)(a1 + 288),
      a2,
      v16,
      byte_3F871B3,
      0,
      a3);
  if ( a4 )
  {
    v17 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)a3 + 24LL);
    if ( v17 != sub_168C3A0 )
    {
      ((void (__fastcall *)(char **, __int64, __int64))v17)(&v30, a3, a2);
      v22 = v31;
      if ( v31 )
      {
        v23 = a1 + 592;
        if ( (*(_BYTE *)(a1 + 680) & 1) == 0 )
        {
          v29 = sub_16E8D30();
          v22 = v31;
          v23 = (__int64)v29;
        }
        sub_16E7EE0(v23, v30, v22);
      }
      if ( v30 != (char *)&v32 )
        j_j___libc_free_0((unsigned __int64)v30);
    }
  }
  if ( *(_DWORD *)(a1 + 456) && *(_BYTE *)(*(_QWORD *)(a1 + 448) + *(unsigned int *)(a1 + 456) - 1LL) != 10 )
  {
    v18 = (__int64 *)(a1 + 592);
    if ( (*(_BYTE *)(a1 + 680) & 1) == 0 )
      v18 = sub_16E8D30();
    v19 = (_BYTE *)v18[3];
    if ( (_BYTE *)v18[2] == v19 )
    {
      sub_16E7EE0((__int64)v18, "\n", 1u);
    }
    else
    {
      *v19 = 10;
      ++v18[3];
    }
  }
  v20 = *(unsigned int *)(a1 + 312);
  if ( *(_DWORD *)(a1 + 312) )
  {
    v25 = *(_QWORD *)(a1 + 272);
    v26 = *(char **)(a1 + 304);
    v27 = *(unsigned int *)(a1 + 312);
    v28 = *(void **)(v25 + 24);
    if ( v20 > *(_QWORD *)(v25 + 16) - (_QWORD)v28 )
    {
      sub_16E7EE0(*(_QWORD *)(a1 + 272), v26, v27);
    }
    else
    {
      memcpy(v28, v26, v27);
      *(_QWORD *)(v25 + 24) += v20;
    }
  }
  *(_DWORD *)(a1 + 312) = 0;
  if ( (*(_BYTE *)(a1 + 680) & 1) != 0 )
    return sub_39E0440(a1);
  v24 = *(_QWORD *)(a1 + 272);
  result = *(_BYTE **)(v24 + 24);
  if ( (unsigned __int64)result >= *(_QWORD *)(v24 + 16) )
    return (_BYTE *)sub_16E7DE0(v24, 10);
  *(_QWORD *)(v24 + 24) = result + 1;
  *result = 10;
  return result;
}
