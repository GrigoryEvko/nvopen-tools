// Function: sub_E15210
// Address: 0xe15210
//
unsigned __int64 __fastcall sub_E15210(__int64 a1, __int64 a2)
{
  __int64 v3; // rsi
  unsigned __int64 v4; // rax
  int v5; // r13d
  unsigned __int64 v6; // rsi
  unsigned __int64 v7; // rax
  __int64 v8; // rax
  __int64 v9; // rsi
  unsigned __int64 v10; // rax
  __int64 v11; // rdx
  unsigned __int64 v12; // rsi
  unsigned __int64 v13; // rax
  __int64 v14; // rax
  _BYTE *v15; // r12
  __int64 v16; // rsi
  unsigned __int64 result; // rax
  void *v18; // rdi
  __int64 v19; // rdx
  unsigned __int64 v20; // rsi
  unsigned __int64 v21; // rax

  sub_E12F20((__int64 *)a2, *(_QWORD *)(a1 + 16), *(const void **)(a1 + 24));
  v3 = *(_QWORD *)(a2 + 8);
  v4 = *(_QWORD *)(a2 + 16);
  v5 = *(_DWORD *)(a2 + 32);
  *(_DWORD *)(a2 + 32) = 0;
  if ( v3 + 1 <= v4 )
  {
    v8 = *(_QWORD *)a2;
  }
  else
  {
    v6 = v3 + 993;
    v7 = 2 * v4;
    if ( v6 > v7 )
      *(_QWORD *)(a2 + 16) = v6;
    else
      *(_QWORD *)(a2 + 16) = v7;
    v8 = realloc(*(void **)a2);
    *(_QWORD *)a2 = v8;
    if ( !v8 )
      goto LABEL_28;
    v3 = *(_QWORD *)(a2 + 8);
  }
  *(_BYTE *)(v8 + v3) = 60;
  ++*(_QWORD *)(a2 + 8);
  (*(void (__fastcall **)(_QWORD, __int64))(**(_QWORD **)(a1 + 32) + 32LL))(*(_QWORD *)(a1 + 32), a2);
  sub_E12F20((__int64 *)a2, 1u, ">");
  v9 = *(_QWORD *)(a2 + 8);
  v10 = *(_QWORD *)(a2 + 16);
  *(_DWORD *)(a2 + 32) = v5 + 1;
  v11 = v9 + 1;
  if ( v9 + 1 <= v10 )
  {
    v14 = *(_QWORD *)a2;
  }
  else
  {
    v12 = v9 + 993;
    v13 = 2 * v10;
    if ( v12 > v13 )
      *(_QWORD *)(a2 + 16) = v12;
    else
      *(_QWORD *)(a2 + 16) = v13;
    v14 = realloc(*(void **)a2);
    *(_QWORD *)a2 = v14;
    if ( !v14 )
      goto LABEL_28;
    v9 = *(_QWORD *)(a2 + 8);
    v11 = v9 + 1;
  }
  *(_QWORD *)(a2 + 8) = v11;
  *(_BYTE *)(v14 + v9) = 40;
  v15 = *(_BYTE **)(a1 + 40);
  if ( (unsigned int)((char)(4 * v15[9]) >> 2) > 0x12 )
  {
    ++*(_DWORD *)(a2 + 32);
    sub_E14360(a2, 40);
    (*(void (__fastcall **)(_BYTE *, __int64))(*(_QWORD *)v15 + 32LL))(v15, a2);
    if ( (v15[9] & 0xC0) != 0x40 )
      (*(void (__fastcall **)(_BYTE *, __int64))(*(_QWORD *)v15 + 40LL))(v15, a2);
    --*(_DWORD *)(a2 + 32);
    sub_E14360(a2, 41);
  }
  else
  {
    (*(void (__fastcall **)(_BYTE *, __int64))(*(_QWORD *)v15 + 32LL))(v15, a2);
    if ( (v15[9] & 0xC0) != 0x40 )
      (*(void (__fastcall **)(_BYTE *, __int64))(*(_QWORD *)v15 + 40LL))(v15, a2);
  }
  v16 = *(_QWORD *)(a2 + 8);
  result = *(_QWORD *)(a2 + 16);
  --*(_DWORD *)(a2 + 32);
  v18 = *(void **)a2;
  v19 = v16 + 1;
  if ( v16 + 1 > result )
  {
    v20 = v16 + 993;
    v21 = 2 * result;
    if ( v20 > v21 )
      *(_QWORD *)(a2 + 16) = v20;
    else
      *(_QWORD *)(a2 + 16) = v21;
    result = realloc(v18);
    *(_QWORD *)a2 = result;
    v18 = (void *)result;
    if ( result )
    {
      v16 = *(_QWORD *)(a2 + 8);
      v19 = v16 + 1;
      goto LABEL_21;
    }
LABEL_28:
    abort();
  }
LABEL_21:
  *(_QWORD *)(a2 + 8) = v19;
  *((_BYTE *)v18 + v16) = 41;
  return result;
}
