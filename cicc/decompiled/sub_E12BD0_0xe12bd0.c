// Function: sub_E12BD0
// Address: 0xe12bd0
//
unsigned __int64 __fastcall sub_E12BD0(__int64 a1, __int64 a2)
{
  __int64 v4; // rsi
  unsigned __int64 v5; // rax
  void *v6; // rdi
  __int64 v7; // rdx
  unsigned __int64 v8; // rsi
  unsigned __int64 v9; // rax
  __int64 v10; // rax
  _BYTE *v11; // r13
  __int64 v12; // rsi
  unsigned __int64 v13; // rax
  void *v14; // rdi
  __int64 v15; // rdx
  unsigned __int64 v16; // rsi
  unsigned __int64 v17; // rax
  __int64 v18; // rax
  __int64 v19; // rsi
  unsigned __int64 v20; // rax
  __int64 v21; // rdx
  unsigned __int64 v22; // rsi
  unsigned __int64 v23; // rax
  __int64 v24; // rax
  _BYTE *v25; // r12
  __int64 v26; // rsi
  unsigned __int64 result; // rax
  void *v28; // rdi
  __int64 v29; // rdx
  unsigned __int64 v30; // rsi
  unsigned __int64 v31; // rax

  ++*(_DWORD *)(a2 + 32);
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
      goto LABEL_31;
    v4 = *(_QWORD *)(a2 + 8);
    v7 = v4 + 1;
  }
  *(_QWORD *)(a2 + 8) = v7;
  *((_BYTE *)v6 + v4) = 40;
  v11 = *(_BYTE **)(a1 + 16);
  (*(void (__fastcall **)(_BYTE *, __int64))(*(_QWORD *)v11 + 32LL))(v11, a2);
  if ( (v11[9] & 0xC0) != 0x40 )
    (*(void (__fastcall **)(_BYTE *, __int64))(*(_QWORD *)v11 + 40LL))(v11, a2);
  v12 = *(_QWORD *)(a2 + 8);
  v13 = *(_QWORD *)(a2 + 16);
  --*(_DWORD *)(a2 + 32);
  v14 = *(void **)a2;
  v15 = v12 + 1;
  if ( v12 + 1 > v13 )
  {
    v16 = v12 + 993;
    v17 = 2 * v13;
    if ( v16 > v17 )
      *(_QWORD *)(a2 + 16) = v16;
    else
      *(_QWORD *)(a2 + 16) = v17;
    v18 = realloc(v14);
    *(_QWORD *)a2 = v18;
    v14 = (void *)v18;
    if ( !v18 )
      goto LABEL_31;
    v12 = *(_QWORD *)(a2 + 8);
    v15 = v12 + 1;
  }
  *(_QWORD *)(a2 + 8) = v15;
  *((_BYTE *)v14 + v12) = 41;
  v19 = *(_QWORD *)(a2 + 8);
  v20 = *(_QWORD *)(a2 + 16);
  ++*(_DWORD *)(a2 + 32);
  v21 = v19 + 1;
  if ( v19 + 1 <= v20 )
  {
    v24 = *(_QWORD *)a2;
  }
  else
  {
    v22 = v19 + 993;
    v23 = 2 * v20;
    if ( v22 > v23 )
      *(_QWORD *)(a2 + 16) = v22;
    else
      *(_QWORD *)(a2 + 16) = v23;
    v24 = realloc(*(void **)a2);
    *(_QWORD *)a2 = v24;
    if ( !v24 )
      goto LABEL_31;
    v19 = *(_QWORD *)(a2 + 8);
    v21 = v19 + 1;
  }
  *(_QWORD *)(a2 + 8) = v21;
  *(_BYTE *)(v24 + v19) = 40;
  v25 = *(_BYTE **)(a1 + 24);
  (*(void (__fastcall **)(_BYTE *, __int64))(*(_QWORD *)v25 + 32LL))(v25, a2);
  if ( (v25[9] & 0xC0) != 0x40 )
    (*(void (__fastcall **)(_BYTE *, __int64))(*(_QWORD *)v25 + 40LL))(v25, a2);
  v26 = *(_QWORD *)(a2 + 8);
  result = *(_QWORD *)(a2 + 16);
  --*(_DWORD *)(a2 + 32);
  v28 = *(void **)a2;
  v29 = v26 + 1;
  if ( v26 + 1 > result )
  {
    v30 = v26 + 993;
    v31 = 2 * result;
    if ( v30 > v31 )
      *(_QWORD *)(a2 + 16) = v30;
    else
      *(_QWORD *)(a2 + 16) = v31;
    result = realloc(v28);
    *(_QWORD *)a2 = result;
    v28 = (void *)result;
    if ( result )
    {
      v26 = *(_QWORD *)(a2 + 8);
      v29 = v26 + 1;
      goto LABEL_26;
    }
LABEL_31:
    abort();
  }
LABEL_26:
  *(_QWORD *)(a2 + 8) = v29;
  *((_BYTE *)v28 + v26) = 41;
  return result;
}
