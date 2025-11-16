// Function: sub_E14A90
// Address: 0xe14a90
//
__int64 __fastcall sub_E14A90(__int64 a1, __int64 a2)
{
  size_t v3; // r13
  _BYTE *v5; // r13
  __int64 result; // rax
  __int64 v7; // rax
  size_t v8; // rdx
  const void *v9; // r14
  char *v10; // rdi
  unsigned __int64 v11; // rsi
  unsigned __int64 v12; // rdx
  __int64 v13; // rax

  v3 = *(_QWORD *)(a1 + 16);
  if ( v3 )
  {
    v7 = *(_QWORD *)(a2 + 8);
    v8 = *(_QWORD *)(a2 + 16);
    v9 = *(const void **)(a1 + 24);
    v10 = *(char **)a2;
    if ( v3 + v7 > v8 )
    {
      v11 = v3 + v7 + 992;
      v12 = 2 * v8;
      if ( v11 > v12 )
        *(_QWORD *)(a2 + 16) = v11;
      else
        *(_QWORD *)(a2 + 16) = v12;
      v13 = realloc(v10);
      *(_QWORD *)a2 = v13;
      v10 = (char *)v13;
      if ( !v13 )
        abort();
      v7 = *(_QWORD *)(a2 + 8);
    }
    memcpy(&v10[v7], v9, v3);
    *(_QWORD *)(a2 + 8) += v3;
  }
  v5 = *(_BYTE **)(a1 + 32);
  if ( (char)(4 * v5[9]) >> 2 >= (unsigned int)((char)(4 * *(_BYTE *)(a1 + 9)) >> 2) )
  {
    ++*(_DWORD *)(a2 + 32);
    sub_E14360(a2, 40);
    (*(void (__fastcall **)(_BYTE *, __int64))(*(_QWORD *)v5 + 32LL))(v5, a2);
    if ( (v5[9] & 0xC0) != 0x40 )
      (*(void (__fastcall **)(_BYTE *, __int64))(*(_QWORD *)v5 + 40LL))(v5, a2);
    --*(_DWORD *)(a2 + 32);
    return sub_E14360(a2, 41);
  }
  else
  {
    (*(void (__fastcall **)(_QWORD, __int64))(*(_QWORD *)v5 + 32LL))(*(_QWORD *)(a1 + 32), a2);
    result = v5[9] & 0xC0;
    if ( (v5[9] & 0xC0) != 0x40 )
      return (*(__int64 (__fastcall **)(_BYTE *, __int64))(*(_QWORD *)v5 + 40LL))(v5, a2);
  }
  return result;
}
