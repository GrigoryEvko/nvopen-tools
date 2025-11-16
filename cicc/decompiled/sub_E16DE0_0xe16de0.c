// Function: sub_E16DE0
// Address: 0xe16de0
//
__int64 __fastcall sub_E16DE0(_QWORD *a1, __int64 a2)
{
  __int64 v4; // rsi
  unsigned __int64 v5; // rax
  void *v6; // rdi
  __int64 v7; // rdx
  unsigned __int64 v8; // rsi
  unsigned __int64 v9; // rax
  __int64 v10; // rax
  __int64 v11; // rsi
  unsigned __int64 v12; // rax
  __int64 v13; // rdx
  unsigned __int64 v14; // rsi
  unsigned __int64 v15; // rax
  __int64 v16; // rax
  _QWORD *v17; // r13
  _QWORD *i; // r14
  _BYTE *v19; // r12
  __int64 v20; // rsi
  unsigned __int64 v21; // rax
  __int64 v22; // rdx
  unsigned __int64 v23; // rsi
  unsigned __int64 v24; // rax
  __int64 result; // rax

  sub_E12F20((__int64 *)a2, 8u, "requires");
  if ( a1[3] )
  {
    sub_E14360(a2, 32);
    ++*(_DWORD *)(a2 + 32);
    sub_E14360(a2, 40);
    sub_E161C0(a1 + 2, (char **)a2);
    --*(_DWORD *)(a2 + 32);
    sub_E14360(a2, 41);
  }
  v4 = *(_QWORD *)(a2 + 8);
  v5 = *(_QWORD *)(a2 + 16);
  v6 = *(void **)a2;
  v7 = v4 + 1;
  if ( v4 + 1 > v5 )
  {
    v8 = v4 + 993;
    v9 = 2 * v5;
    if ( v8 > v9 )
      *(_QWORD *)(a2 + 16) = v8;
    else
      *(_QWORD *)(a2 + 16) = v9;
    v10 = realloc(v6);
    *(_QWORD *)a2 = v10;
    v6 = (void *)v10;
    if ( !v10 )
      goto LABEL_28;
    v4 = *(_QWORD *)(a2 + 8);
    v7 = v4 + 1;
  }
  *(_QWORD *)(a2 + 8) = v7;
  *((_BYTE *)v6 + v4) = 32;
  v11 = *(_QWORD *)(a2 + 8);
  v12 = *(_QWORD *)(a2 + 16);
  ++*(_DWORD *)(a2 + 32);
  v13 = v11 + 1;
  if ( v11 + 1 <= v12 )
  {
    v16 = *(_QWORD *)a2;
  }
  else
  {
    v14 = v11 + 993;
    v15 = 2 * v12;
    if ( v14 > v15 )
      *(_QWORD *)(a2 + 16) = v14;
    else
      *(_QWORD *)(a2 + 16) = v15;
    v16 = realloc(*(void **)a2);
    *(_QWORD *)a2 = v16;
    if ( !v16 )
      goto LABEL_28;
    v11 = *(_QWORD *)(a2 + 8);
    v13 = v11 + 1;
  }
  *(_QWORD *)(a2 + 8) = v13;
  *(_BYTE *)(v16 + v11) = 123;
  v17 = (_QWORD *)a1[4];
  for ( i = &v17[a1[5]]; i != v17; ++v17 )
  {
    v19 = (_BYTE *)*v17;
    (*(void (__fastcall **)(_QWORD, __int64))(*(_QWORD *)*v17 + 32LL))(*v17, a2);
    if ( (v19[9] & 0xC0) != 0x40 )
      (*(void (__fastcall **)(_BYTE *, __int64))(*(_QWORD *)v19 + 40LL))(v19, a2);
  }
  sub_E14360(a2, 32);
  v20 = *(_QWORD *)(a2 + 8);
  v21 = *(_QWORD *)(a2 + 16);
  --*(_DWORD *)(a2 + 32);
  v22 = v20 + 1;
  if ( v20 + 1 <= v21 )
  {
    result = *(_QWORD *)a2;
    goto LABEL_24;
  }
  v23 = v20 + 993;
  v24 = 2 * v21;
  if ( v23 > v24 )
    *(_QWORD *)(a2 + 16) = v23;
  else
    *(_QWORD *)(a2 + 16) = v24;
  result = realloc(*(void **)a2);
  *(_QWORD *)a2 = result;
  if ( !result )
LABEL_28:
    abort();
  v20 = *(_QWORD *)(a2 + 8);
  v22 = v20 + 1;
LABEL_24:
  *(_QWORD *)(a2 + 8) = v22;
  *(_BYTE *)(result + v20) = 125;
  return result;
}
