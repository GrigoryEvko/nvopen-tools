// Function: sub_BDCEC0
// Address: 0xbdcec0
//
char __fastcall sub_BDCEC0(__int64 a1, __int64 a2, _BYTE *a3, __int64 a4)
{
  char v6; // al
  unsigned __int8 v7; // dl
  const char *v8; // r13
  const char *v9; // rax
  char v10; // dl
  __int64 v11; // rdx
  const char *i; // rbx
  const char *v13; // r12
  _BYTE *v14; // rax
  const char *v16; // [rsp+0h] [rbp-60h] BYREF
  char v17; // [rsp+20h] [rbp-40h]
  char v18; // [rsp+21h] [rbp-3Fh]

  v6 = a3[32];
  v7 = *(_BYTE *)a4;
  if ( (v6 & 0xF) != 1 )
  {
    v8 = (const char *)a4;
    if ( v7 > 3u )
    {
LABEL_10:
      if ( v7 == 5 )
        sub_BDC820(a1, a4);
      v11 = 32LL * (*(_DWORD *)(a4 + 4) & 0x7FFFFFF);
      v9 = (const char *)(a4 - v11);
      if ( (*(_BYTE *)(a4 + 7) & 0x40) != 0 )
      {
        v9 = *(const char **)(a4 - 8);
        v8 = &v9[v11];
      }
      for ( i = v9; v8 != i; i += 32 )
      {
        LOBYTE(v9) = **(_BYTE **)i;
        if ( (_BYTE)v9 == 1 || (unsigned __int8)v9 <= 0x15u )
          LOBYTE(v9) = sub_BDCEC0(a1, a2, a3);
      }
      return (char)v9;
    }
    if ( (*(_BYTE *)(a4 + 32) & 0xF) == 1 || (LOBYTE(v9) = sub_B2FC80(a4), (_BYTE)v9) )
    {
      v18 = 1;
      v9 = "Alias must point to a definition";
      goto LABEL_20;
    }
    v7 = *(_BYTE *)a4;
LABEL_6:
    if ( v7 != 1 )
      return (char)v9;
    sub_AE6EC0(a2, a4);
    if ( v10 )
    {
      v8 = (const char *)a4;
      if ( !(unsigned __int8)sub_B2F6B0(a4) )
      {
        v7 = *(_BYTE *)a4;
        goto LABEL_10;
      }
      v18 = 1;
      v9 = "Alias cannot point to an interposable alias";
    }
    else
    {
      v18 = 1;
      v9 = "Aliases cannot form a cycle";
    }
LABEL_20:
    v13 = *(const char **)a1;
    v16 = v9;
    v17 = 3;
    if ( v13 )
      goto LABEL_21;
LABEL_30:
    *(_BYTE *)(a1 + 152) = 1;
    return (char)v9;
  }
  if ( v7 <= 3u )
  {
    LOBYTE(v9) = *(_BYTE *)(a4 + 32) & 0xF;
    if ( (_BYTE)v9 == 1 )
      goto LABEL_6;
  }
  v13 = *(const char **)a1;
  v9 = "available_externally alias must point to available_externally global value";
  v18 = 1;
  v16 = "available_externally alias must point to available_externally global value";
  v17 = 3;
  if ( !v13 )
    goto LABEL_30;
LABEL_21:
  sub_CA0E80(&v16, v13);
  v14 = (_BYTE *)*((_QWORD *)v13 + 4);
  if ( (unsigned __int64)v14 >= *((_QWORD *)v13 + 3) )
  {
    sub_CB5D20(v13, 10);
  }
  else
  {
    *((_QWORD *)v13 + 4) = v14 + 1;
    *v14 = 10;
  }
  v9 = *(const char **)a1;
  *(_BYTE *)(a1 + 152) = 1;
  if ( v9 )
    LOBYTE(v9) = (unsigned __int8)sub_BDBD80(a1, a3);
  return (char)v9;
}
