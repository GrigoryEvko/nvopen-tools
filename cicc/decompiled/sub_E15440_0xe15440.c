// Function: sub_E15440
// Address: 0xe15440
//
unsigned __int64 __fastcall sub_E15440(__int64 a1, __int64 a2)
{
  _BYTE *v4; // r12
  __int64 v5; // rsi
  unsigned __int64 v6; // rax
  void *v7; // rdi
  __int64 v8; // rdx
  unsigned __int64 v9; // rsi
  unsigned __int64 v10; // rax
  __int64 v11; // rax
  _BYTE *v12; // r12
  __int64 v13; // rsi
  unsigned __int64 result; // rax
  void *v15; // rdi
  __int64 v16; // rdx
  unsigned __int64 v17; // rsi
  unsigned __int64 v18; // rax

  v4 = *(_BYTE **)(a1 + 16);
  if ( (char)(4 * v4[9]) >> 2 >= (unsigned int)((char)(4 * *(_BYTE *)(a1 + 9)) >> 2) )
  {
    ++*(_DWORD *)(a2 + 32);
    sub_E14360(a2, 40);
    (*(void (__fastcall **)(_BYTE *, __int64))(*(_QWORD *)v4 + 32LL))(v4, a2);
    if ( (v4[9] & 0xC0) != 0x40 )
      (*(void (__fastcall **)(_BYTE *, __int64))(*(_QWORD *)v4 + 40LL))(v4, a2);
    --*(_DWORD *)(a2 + 32);
    sub_E14360(a2, 41);
  }
  else
  {
    (*(void (__fastcall **)(_QWORD))(*(_QWORD *)v4 + 32LL))(*(_QWORD *)(a1 + 16));
    if ( (v4[9] & 0xC0) != 0x40 )
      (*(void (__fastcall **)(_BYTE *, __int64))(*(_QWORD *)v4 + 40LL))(v4, a2);
  }
  v5 = *(_QWORD *)(a2 + 8);
  v6 = *(_QWORD *)(a2 + 16);
  ++*(_DWORD *)(a2 + 32);
  v7 = *(void **)a2;
  v8 = v5 + 1;
  if ( v5 + 1 > v6 )
  {
    v9 = v5 + 993;
    v10 = 2 * v6;
    if ( v9 > v10 )
      *(_QWORD *)(a2 + 16) = v9;
    else
      *(_QWORD *)(a2 + 16) = v10;
    v11 = realloc(v7);
    *(_QWORD *)a2 = v11;
    v7 = (void *)v11;
    if ( !v11 )
      goto LABEL_26;
    v5 = *(_QWORD *)(a2 + 8);
    v8 = v5 + 1;
  }
  *(_QWORD *)(a2 + 8) = v8;
  *((_BYTE *)v7 + v5) = 91;
  v12 = *(_BYTE **)(a1 + 24);
  if ( (unsigned int)((char)(4 * v12[9]) >> 2) > 0x12 )
  {
    ++*(_DWORD *)(a2 + 32);
    sub_E14360(a2, 40);
    (*(void (__fastcall **)(_BYTE *, __int64))(*(_QWORD *)v12 + 32LL))(v12, a2);
    if ( (v12[9] & 0xC0) != 0x40 )
      (*(void (__fastcall **)(_BYTE *, __int64))(*(_QWORD *)v12 + 40LL))(v12, a2);
    --*(_DWORD *)(a2 + 32);
    sub_E14360(a2, 41);
  }
  else
  {
    (*(void (__fastcall **)(_QWORD, __int64))(*(_QWORD *)v12 + 32LL))(*(_QWORD *)(a1 + 24), a2);
    if ( (v12[9] & 0xC0) != 0x40 )
      (*(void (__fastcall **)(_BYTE *, __int64))(*(_QWORD *)v12 + 40LL))(v12, a2);
  }
  v13 = *(_QWORD *)(a2 + 8);
  result = *(_QWORD *)(a2 + 16);
  --*(_DWORD *)(a2 + 32);
  v15 = *(void **)a2;
  v16 = v13 + 1;
  if ( v13 + 1 > result )
  {
    v17 = v13 + 993;
    v18 = 2 * result;
    if ( v17 > v18 )
      *(_QWORD *)(a2 + 16) = v17;
    else
      *(_QWORD *)(a2 + 16) = v18;
    result = realloc(v15);
    *(_QWORD *)a2 = result;
    v15 = (void *)result;
    if ( result )
    {
      v13 = *(_QWORD *)(a2 + 8);
      v16 = v13 + 1;
      goto LABEL_17;
    }
LABEL_26:
    abort();
  }
LABEL_17:
  *(_QWORD *)(a2 + 8) = v16;
  *((_BYTE *)v15 + v13) = 93;
  return result;
}
