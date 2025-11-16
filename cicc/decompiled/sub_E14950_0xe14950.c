// Function: sub_E14950
// Address: 0xe14950
//
__int64 __fastcall sub_E14950(__int64 a1, __int64 a2)
{
  _BYTE *v3; // r13
  __int64 result; // rax
  size_t v5; // r13
  __int64 v6; // rax
  size_t v7; // rdx
  const void *v8; // r12
  char *v9; // rdi
  unsigned __int64 v10; // rsi
  unsigned __int64 v11; // rdx
  __int64 v12; // rax

  v3 = *(_BYTE **)(a1 + 16);
  if ( (char)(4 * v3[9]) >> 2 < (unsigned int)(((char)(4 * *(_BYTE *)(a1 + 9)) >> 2) + 1) )
  {
    (*(void (__fastcall **)(_QWORD))(*(_QWORD *)v3 + 32LL))(*(_QWORD *)(a1 + 16));
    result = v3[9] & 0xC0;
    if ( (v3[9] & 0xC0) != 0x40 )
      result = (*(__int64 (__fastcall **)(_BYTE *, __int64))(*(_QWORD *)v3 + 40LL))(v3, a2);
    v5 = *(_QWORD *)(a1 + 24);
    if ( !v5 )
      return result;
LABEL_9:
    v6 = *(_QWORD *)(a2 + 8);
    v7 = *(_QWORD *)(a2 + 16);
    v8 = *(const void **)(a1 + 32);
    v9 = *(char **)a2;
    if ( v5 + v6 > v7 )
    {
      v10 = v5 + v6 + 992;
      v11 = 2 * v7;
      if ( v10 > v11 )
        *(_QWORD *)(a2 + 16) = v10;
      else
        *(_QWORD *)(a2 + 16) = v11;
      v12 = realloc(v9);
      *(_QWORD *)a2 = v12;
      v9 = (char *)v12;
      if ( !v12 )
        abort();
      v6 = *(_QWORD *)(a2 + 8);
    }
    result = (__int64)memcpy(&v9[v6], v8, v5);
    *(_QWORD *)(a2 + 8) += v5;
    return result;
  }
  ++*(_DWORD *)(a2 + 32);
  sub_E14360(a2, 40);
  (*(void (__fastcall **)(_BYTE *, __int64))(*(_QWORD *)v3 + 32LL))(v3, a2);
  if ( (v3[9] & 0xC0) != 0x40 )
    (*(void (__fastcall **)(_BYTE *, __int64))(*(_QWORD *)v3 + 40LL))(v3, a2);
  --*(_DWORD *)(a2 + 32);
  result = sub_E14360(a2, 41);
  v5 = *(_QWORD *)(a1 + 24);
  if ( v5 )
    goto LABEL_9;
  return result;
}
