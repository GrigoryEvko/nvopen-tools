// Function: sub_E11BE0
// Address: 0xe11be0
//
__int64 __fastcall sub_E11BE0(__int64 a1, __int64 a2)
{
  _BYTE *v4; // r13
  char v5; // r13
  __int64 v6; // rsi
  unsigned __int64 v7; // rax
  void *v8; // rdi
  __int64 v9; // rdx
  unsigned __int64 v10; // rsi
  unsigned __int64 v11; // rax
  __int64 v12; // rax
  _BYTE *v13; // r13
  __int64 result; // rax

  v4 = *(_BYTE **)(a1 + 16);
  if ( !v4 )
    goto LABEL_14;
  (*(void (__fastcall **)(_QWORD))(*(_QWORD *)v4 + 32LL))(*(_QWORD *)(a1 + 16));
  if ( (v4[9] & 0xC0) != 0x40 )
    (*(void (__fastcall **)(_BYTE *, __int64))(*(_QWORD *)v4 + 40LL))(v4, a2);
  if ( !*(_QWORD *)(a1 + 16) )
  {
LABEL_14:
    if ( !*(_BYTE *)(a1 + 32) )
      goto LABEL_12;
    goto LABEL_15;
  }
  v5 = 46;
  if ( *(_BYTE *)(a1 + 32) )
LABEL_15:
    v5 = 58;
  v6 = *(_QWORD *)(a2 + 8);
  v7 = *(_QWORD *)(a2 + 16);
  v8 = *(void **)a2;
  v9 = v6 + 1;
  if ( v6 + 1 > v7 )
  {
    v10 = v6 + 993;
    v11 = 2 * v7;
    if ( v10 > v11 )
      *(_QWORD *)(a2 + 16) = v10;
    else
      *(_QWORD *)(a2 + 16) = v11;
    v12 = realloc(v8);
    *(_QWORD *)a2 = v12;
    v8 = (void *)v12;
    if ( !v12 )
      abort();
    v6 = *(_QWORD *)(a2 + 8);
    v9 = v6 + 1;
  }
  *(_QWORD *)(a2 + 8) = v9;
  *((_BYTE *)v8 + v6) = v5;
LABEL_12:
  v13 = *(_BYTE **)(a1 + 24);
  (*(void (__fastcall **)(_BYTE *, __int64))(*(_QWORD *)v13 + 32LL))(v13, a2);
  result = v13[9] & 0xC0;
  if ( (v13[9] & 0xC0) != 0x40 )
    return (*(__int64 (__fastcall **)(_BYTE *, __int64))(*(_QWORD *)v13 + 40LL))(v13, a2);
  return result;
}
