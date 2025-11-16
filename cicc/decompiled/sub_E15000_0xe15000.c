// Function: sub_E15000
// Address: 0xe15000
//
__int64 __fastcall sub_E15000(__int64 a1, __int64 a2)
{
  _BYTE *v3; // r13
  size_t v5; // r13
  _BYTE *v6; // r13
  __int64 result; // rax
  __int64 v8; // rax
  size_t v9; // rdx
  const void *v10; // r14
  char *v11; // rdi
  unsigned __int64 v12; // rsi
  unsigned __int64 v13; // rdx
  __int64 v14; // rax

  v3 = *(_BYTE **)(a1 + 16);
  if ( (char)(4 * v3[9]) >> 2 < (unsigned int)(((char)(4 * *(_BYTE *)(a1 + 9)) >> 2) + 1) )
  {
    (*(void (__fastcall **)(_QWORD))(*(_QWORD *)v3 + 32LL))(*(_QWORD *)(a1 + 16));
    if ( (v3[9] & 0xC0) != 0x40 )
      (*(void (__fastcall **)(_BYTE *, __int64))(*(_QWORD *)v3 + 40LL))(v3, a2);
    v5 = *(_QWORD *)(a1 + 24);
    if ( !v5 )
      goto LABEL_5;
LABEL_12:
    v8 = *(_QWORD *)(a2 + 8);
    v9 = *(_QWORD *)(a2 + 16);
    v10 = *(const void **)(a1 + 32);
    v11 = *(char **)a2;
    if ( v5 + v8 > v9 )
    {
      v12 = v5 + v8 + 992;
      v13 = 2 * v9;
      if ( v12 > v13 )
        *(_QWORD *)(a2 + 16) = v12;
      else
        *(_QWORD *)(a2 + 16) = v13;
      v14 = realloc(v11);
      *(_QWORD *)a2 = v14;
      v11 = (char *)v14;
      if ( !v14 )
        abort();
      v8 = *(_QWORD *)(a2 + 8);
    }
    memcpy(&v11[v8], v10, v5);
    *(_QWORD *)(a2 + 8) += v5;
    goto LABEL_5;
  }
  ++*(_DWORD *)(a2 + 32);
  sub_E14360(a2, 40);
  (*(void (__fastcall **)(_BYTE *, __int64))(*(_QWORD *)v3 + 32LL))(v3, a2);
  if ( (v3[9] & 0xC0) != 0x40 )
    (*(void (__fastcall **)(_BYTE *, __int64))(*(_QWORD *)v3 + 40LL))(v3, a2);
  --*(_DWORD *)(a2 + 32);
  sub_E14360(a2, 41);
  v5 = *(_QWORD *)(a1 + 24);
  if ( v5 )
    goto LABEL_12;
LABEL_5:
  v6 = *(_BYTE **)(a1 + 40);
  if ( (char)(4 * v6[9]) >> 2 >= (unsigned int)((char)(4 * *(_BYTE *)(a1 + 9)) >> 2) )
  {
    ++*(_DWORD *)(a2 + 32);
    sub_E14360(a2, 40);
    (*(void (__fastcall **)(_BYTE *, __int64))(*(_QWORD *)v6 + 32LL))(v6, a2);
    if ( (v6[9] & 0xC0) != 0x40 )
      (*(void (__fastcall **)(_BYTE *, __int64))(*(_QWORD *)v6 + 40LL))(v6, a2);
    --*(_DWORD *)(a2 + 32);
    return sub_E14360(a2, 41);
  }
  else
  {
    (*(void (__fastcall **)(_QWORD, __int64))(*(_QWORD *)v6 + 32LL))(*(_QWORD *)(a1 + 40), a2);
    result = v6[9] & 0xC0;
    if ( (v6[9] & 0xC0) != 0x40 )
      return (*(__int64 (__fastcall **)(_BYTE *, __int64))(*(_QWORD *)v6 + 40LL))(v6, a2);
  }
  return result;
}
