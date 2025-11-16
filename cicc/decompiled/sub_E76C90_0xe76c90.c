// Function: sub_E76C90
// Address: 0xe76c90
//
__int64 __fastcall sub_E76C90(__int64 a1, _QWORD *a2, __int64 a3)
{
  __int64 v6; // r9
  size_t v7; // r13
  __int64 v8; // rax
  char *v9; // rsi
  __int64 v10; // r8
  __int64 v11; // rax
  __int64 v12; // r13
  __int64 *v13; // r13
  __int64 v14; // rsi
  unsigned int v15; // esi
  __int64 v16; // rsi
  int v17; // r13d
  __int64 result; // rax
  __int64 v19; // r14
  __int64 v20; // r13
  _QWORD *v21; // rdx
  __int64 v22; // rcx
  const void *v23; // r8
  char *v24; // rdi
  __int64 v25; // rdi
  __int64 v26; // rdx
  __int64 *v27; // [rsp+8h] [rbp-158h]
  const void *v28; // [rsp+8h] [rbp-158h]
  char *v29; // [rsp+10h] [rbp-150h] BYREF
  size_t v30; // [rsp+18h] [rbp-148h]
  __int64 v31; // [rsp+20h] [rbp-140h]
  _BYTE v32[312]; // [rsp+28h] [rbp-138h] BYREF

  (*(void (__fastcall **)(_QWORD *, __int64, __int64))(*a2 + 536LL))(a2, 1, 1);
  sub_E98EB0(a2, 1, 0);
  sub_E98EB0(a2, (-(__int64)(*(_BYTE *)(a3 + 168) == 0) & 0xFFFFFFFFFFFFFFE9LL) + 31, 0);
  sub_E98EB0(a2, *(unsigned int *)(a1 + 16) + 1LL, 0);
  v7 = *(_QWORD *)(a1 + 408);
  v30 = 0;
  v29 = v32;
  v8 = a2[1];
  v31 = 256;
  v9 = *(char **)(v8 + 1528);
  v10 = *(_QWORD *)(v8 + 1536);
  if ( v7 )
  {
    v23 = *(const void **)(a1 + 400);
    v24 = v32;
    if ( v7 > 0x100 )
    {
      v28 = *(const void **)(a1 + 400);
      sub_C8D290((__int64)&v29, v32, v7, 1u, (__int64)v23, v6);
      v23 = v28;
      v24 = &v29[v30];
    }
    memcpy(v24, v23, v7);
    v25 = a2[1];
    v30 += v7;
    sub_E65530(v25, &v29);
    v9 = v29;
    v10 = v30;
    if ( !*(_BYTE *)(a3 + 168) )
      goto LABEL_3;
    v9 = sub_C948A0((char ***)(a3 + 96), v29, v30);
    v10 = v26;
  }
  if ( !*(_BYTE *)(a3 + 168) )
  {
LABEL_3:
    (*(void (__fastcall **)(_QWORD *, char *, __int64))(*a2 + 512LL))(a2, v9, v10);
    (*(void (__fastcall **)(_QWORD *, void *, __int64))(*a2 + 512LL))(a2, &unk_3F801CE, 1);
    v11 = *(_QWORD *)(a1 + 8);
    v12 = 32LL * *(unsigned int *)(a1 + 16);
    v27 = (__int64 *)(v11 + v12);
    if ( v11 + v12 != v11 )
    {
      v13 = *(__int64 **)(a1 + 8);
      do
      {
        v14 = *v13;
        v13 += 4;
        (*(void (__fastcall **)(_QWORD *, __int64, _QWORD))(*a2 + 512LL))(a2, v14, *(v13 - 3));
        (*(void (__fastcall **)(_QWORD *, void *, __int64))(*a2 + 512LL))(a2, &unk_3F801CE, 1);
      }
      while ( v27 != v13 );
    }
    goto LABEL_6;
  }
  sub_E76730(a3, a2, v9, v10);
  v19 = *(_QWORD *)(a1 + 8);
  v20 = v19 + 32LL * *(unsigned int *)(a1 + 16);
  while ( v20 != v19 )
  {
    v21 = *(_QWORD **)v19;
    v22 = *(_QWORD *)(v19 + 8);
    v19 += 32;
    sub_E76730(a3, a2, v21, v22);
  }
LABEL_6:
  (*(void (__fastcall **)(_QWORD *, __int64, __int64))(*a2 + 536LL))(
    a2,
    2 - ((*(_BYTE *)(a1 + 513) == 0) - 1LL) - ((*(_BYTE *)(a1 + 512) == 0) - 1LL),
    1);
  sub_E98EB0(a2, 1, 0);
  sub_E98EB0(a2, (-(__int64)(*(_BYTE *)(a3 + 168) == 0) & 0xFFFFFFFFFFFFFFE9LL) + 31, 0);
  sub_E98EB0(a2, 2, 0);
  sub_E98EB0(a2, 15, 0);
  if ( *(_BYTE *)(a1 + 513) )
  {
    sub_E98EB0(a2, 5, 0);
    sub_E98EB0(a2, 30, 0);
  }
  if ( *(_BYTE *)(a1 + 512) )
  {
    sub_E98EB0(a2, 8193, 0);
    sub_E98EB0(a2, (-(__int64)(*(_BYTE *)(a3 + 168) == 0) & 0xFFFFFFFFFFFFFFE9LL) + 31, 0);
  }
  v15 = 1;
  if ( *(_DWORD *)(a1 + 128) )
    v15 = *(_DWORD *)(a1 + 128);
  sub_E98EB0(a2, v15, 0);
  v16 = a1 + 432;
  if ( !*(_QWORD *)(a1 + 440) )
    v16 = *(_QWORD *)(a1 + 120) + 80LL;
  v17 = 1;
  sub_E76850(a2, v16, *(_BYTE *)(a1 + 513), *(_BYTE *)(a1 + 512), a3);
  result = 1;
  if ( *(_DWORD *)(a1 + 128) > 1u )
  {
    do
    {
      v16 = *(_QWORD *)(a1 + 120) + 80 * result;
      sub_E76850(a2, v16, *(_BYTE *)(a1 + 513), *(_BYTE *)(a1 + 512), a3);
      result = (unsigned int)(v17 + 1);
      v17 = result;
    }
    while ( (unsigned int)result < *(_DWORD *)(a1 + 128) );
  }
  if ( v29 != v32 )
    return _libc_free(v29, v16);
  return result;
}
