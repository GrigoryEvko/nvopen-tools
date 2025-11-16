// Function: sub_E147D0
// Address: 0xe147d0
//
__int64 *__fastcall sub_E147D0(__int64 a1, __int64 a2)
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
  __int64 *v20; // rax

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
      goto LABEL_18;
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
    if ( v18 )
    {
      v12 = *(_QWORD *)(a2 + 8);
      v15 = v12 + 1;
      goto LABEL_13;
    }
LABEL_18:
    abort();
  }
LABEL_13:
  *(_QWORD *)(a2 + 8) = v15;
  *((_BYTE *)v14 + v12) = 41;
  if ( **(_BYTE **)(a1 + 32) != 110 )
    return sub_E12F20((__int64 *)a2, *(_QWORD *)(a1 + 24), *(const void **)(a1 + 32));
  v20 = (__int64 *)sub_E14360(a2, 45);
  return sub_E12F20(v20, *(_QWORD *)(a1 + 24) - 1LL, (const void *)(*(_QWORD *)(a1 + 32) + 1LL));
}
