// Function: sub_E16FF0
// Address: 0xe16ff0
//
__int64 __fastcall sub_E16FF0(__int64 a1, __int64 a2)
{
  __int64 v4; // rsi
  unsigned __int64 v5; // rax
  char *v6; // rdi
  unsigned __int64 v7; // rsi
  unsigned __int64 v8; // rax
  __int64 v9; // rax
  char *v10; // rdi
  __int64 v11; // rsi
  unsigned __int64 v12; // rax
  char *v13; // rdi
  unsigned __int64 v14; // rsi
  unsigned __int64 v15; // rax
  __int64 v16; // rax
  _BYTE *v17; // r13
  __int64 result; // rax

  if ( *(_BYTE *)(a1 + 56) )
    sub_E12F20((__int64 *)a2, 2u, "::");
  v4 = *(_QWORD *)(a2 + 8);
  v5 = *(_QWORD *)(a2 + 16);
  v6 = *(char **)a2;
  if ( v4 + 3 > v5 )
  {
    v7 = v4 + 995;
    v8 = 2 * v5;
    if ( v7 <= v8 )
      *(_QWORD *)(a2 + 16) = v8;
    else
      *(_QWORD *)(a2 + 16) = v7;
    v9 = realloc(v6);
    *(_QWORD *)a2 = v9;
    v6 = (char *)v9;
    if ( !v9 )
      goto LABEL_23;
    v4 = *(_QWORD *)(a2 + 8);
  }
  v10 = &v6[v4];
  *(_WORD *)v10 = 25966;
  v10[2] = 119;
  *(_QWORD *)(a2 + 8) += 3LL;
  if ( *(_BYTE *)(a1 + 57) )
    sub_E12F20((__int64 *)a2, 2u, "[]");
  if ( *(_QWORD *)(a1 + 24) )
  {
    ++*(_DWORD *)(a2 + 32);
    sub_E14360(a2, 40);
    sub_E161C0((_QWORD *)(a1 + 16), (char **)a2);
    --*(_DWORD *)(a2 + 32);
    sub_E14360(a2, 41);
  }
  v11 = *(_QWORD *)(a2 + 8);
  v12 = *(_QWORD *)(a2 + 16);
  v13 = *(char **)a2;
  if ( v11 + 1 > v12 )
  {
    v14 = v11 + 993;
    v15 = 2 * v12;
    if ( v14 > v15 )
      *(_QWORD *)(a2 + 16) = v14;
    else
      *(_QWORD *)(a2 + 16) = v15;
    v16 = realloc(v13);
    *(_QWORD *)a2 = v16;
    v13 = (char *)v16;
    if ( v16 )
    {
      v11 = *(_QWORD *)(a2 + 8);
      goto LABEL_17;
    }
LABEL_23:
    abort();
  }
LABEL_17:
  v13[v11] = 32;
  ++*(_QWORD *)(a2 + 8);
  v17 = *(_BYTE **)(a1 + 32);
  (*(void (__fastcall **)(_BYTE *, __int64))(*(_QWORD *)v17 + 32LL))(v17, a2);
  result = v17[9] & 0xC0;
  if ( (v17[9] & 0xC0) != 0x40 )
    result = (*(__int64 (__fastcall **)(_BYTE *, __int64))(*(_QWORD *)v17 + 40LL))(v17, a2);
  if ( *(_QWORD *)(a1 + 48) )
  {
    ++*(_DWORD *)(a2 + 32);
    sub_E14360(a2, 40);
    sub_E161C0((_QWORD *)(a1 + 40), (char **)a2);
    --*(_DWORD *)(a2 + 32);
    return sub_E14360(a2, 41);
  }
  return result;
}
