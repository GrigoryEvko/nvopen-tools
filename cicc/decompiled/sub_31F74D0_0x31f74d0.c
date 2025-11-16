// Function: sub_31F74D0
// Address: 0x31f74d0
//
__int64 __fastcall sub_31F74D0(__int64 a1, __int64 a2, const void *a3, size_t a4)
{
  unsigned __int8 v6; // al
  __int64 v7; // rdi
  __int64 v8; // rax
  unsigned __int8 v9; // dl
  unsigned int v10; // r14d
  const void *v12; // rax
  __int64 v13; // rdx
  _BYTE **v14; // rax
  _BYTE *v15; // rax

  v6 = *(_BYTE *)(a1 - 16);
  if ( (v6 & 2) != 0 )
    v7 = *(_QWORD *)(a1 - 32);
  else
    v7 = a1 - 16 - 8LL * ((v6 >> 2) & 0xF);
  v8 = *(_QWORD *)(v7 + 24);
  if ( !v8 )
    goto LABEL_6;
  v9 = *(_BYTE *)(v8 - 16);
  if ( (v9 & 2) != 0 )
  {
    if ( !*(_DWORD *)(v8 - 24) )
    {
LABEL_6:
      v10 = 0;
      goto LABEL_7;
    }
    v14 = *(_BYTE ***)(v8 - 32);
  }
  else
  {
    if ( (*(_WORD *)(v8 - 16) & 0x3C0) == 0 )
      goto LABEL_6;
    v14 = (_BYTE **)(v8 - 16 - 8LL * ((v9 >> 2) & 0xF));
  }
  v15 = *v14;
  if ( !v15 || *v15 != 14 )
    goto LABEL_6;
  v10 = 1;
  if ( (v15[23] & 4) == 0 )
  {
    if ( a2 )
      goto LABEL_8;
    return 0;
  }
LABEL_7:
  if ( !a2 )
    return v10;
LABEL_8:
  if ( (*(_BYTE *)(a2 + 23) & 4) == 0 )
    return v10;
  v12 = (const void *)sub_A547D0(a2, 2);
  if ( a4 != v13 || a4 && memcmp(a3, v12, a4) )
    return v10;
  return v10 | 2;
}
