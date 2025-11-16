// Function: sub_FF7D20
// Address: 0xff7d20
//
__int64 __fastcall sub_FF7D20(__int64 a1, __int64 a2, __int64 *a3)
{
  unsigned __int64 v4; // rax
  __int64 v6; // r14
  char *v7; // r15
  char v10; // al
  __int64 v11; // rax
  __int64 v12; // rax
  unsigned int v13; // ebx
  __int64 v14; // rdi
  int v15; // eax
  bool v16; // al
  __int64 v17; // rax
  unsigned int v18; // esi
  _DWORD *v19; // r8
  _DWORD *v20; // rdi
  __int64 v21; // rax
  char *v22; // rax
  unsigned __int8 v23; // dl
  __int64 v24; // rax
  __int64 v25; // rsi
  unsigned __int64 v26; // rdx
  __int64 v27; // rax
  bool v28; // al
  unsigned int v29; // [rsp+18h] [rbp-38h] BYREF
  unsigned int v30[13]; // [rsp+1Ch] [rbp-34h] BYREF

  v4 = *(_QWORD *)(a2 + 48) & 0xFFFFFFFFFFFFFFF8LL;
  if ( v4 == a2 + 48 || !v4 || (unsigned int)*(unsigned __int8 *)(v4 - 24) - 30 > 0xA )
    BUG();
  if ( *(_BYTE *)(v4 - 24) != 31 )
    return 0;
  if ( (*(_DWORD *)(v4 - 20) & 0x7FFFFFF) != 3 )
    return 0;
  v6 = *(_QWORD *)(v4 - 120);
  if ( *(_BYTE *)v6 != 82 )
    return 0;
  v7 = *(char **)(v6 - 32);
  v10 = *v7;
  if ( (unsigned __int8)*v7 <= 0x1Cu )
  {
    if ( v10 != 17 )
      return 0;
  }
  else
  {
    if ( v10 != 78 )
      return 0;
    v7 = (char *)*((_QWORD *)v7 - 4);
    if ( !v7 )
      goto LABEL_63;
    if ( *v7 != 17 )
      return 0;
  }
  v11 = *(_QWORD *)(v6 - 64);
  if ( *(_BYTE *)v11 != 57 )
    goto LABEL_13;
  if ( (*(_BYTE *)(v11 + 7) & 0x40) != 0 )
    v21 = *(_QWORD *)(v11 - 8);
  else
    v21 = v11 - 32LL * (*(_DWORD *)(v11 + 4) & 0x7FFFFFF);
  v22 = *(char **)(v21 + 32);
  v23 = *v22;
  if ( (unsigned __int8)*v22 <= 0x1Cu )
  {
    if ( v23 != 17 )
      goto LABEL_13;
    goto LABEL_35;
  }
  if ( v23 != 78 )
    goto LABEL_13;
  v22 = (char *)*((_QWORD *)v22 - 4);
  if ( !v22 )
LABEL_63:
    BUG();
  if ( *v22 != 17 )
    goto LABEL_13;
LABEL_35:
  if ( *((_DWORD *)v22 + 8) > 0x40u )
  {
    if ( (unsigned int)sub_C44630((__int64)(v22 + 24)) == 1 )
      return 0;
  }
  else
  {
    v24 = *((_QWORD *)v22 + 3);
    if ( v24 && (v24 & (v24 - 1)) == 0 )
      return 0;
  }
LABEL_13:
  v29 = 523;
  if ( a3 )
  {
    v12 = *(_QWORD *)(v6 - 64);
    if ( *(_BYTE *)v12 == 85 )
    {
      v25 = *(_QWORD *)(v12 - 32);
      if ( v25 )
      {
        if ( !*(_BYTE *)v25 && *(_QWORD *)(v25 + 24) == *(_QWORD *)(v12 + 80) )
        {
          sub_981210(*a3, v25, &v29);
          LOBYTE(v26) = 0;
          if ( v29 - 458 <= 0xD )
            v26 = (0x2809uLL >> ((unsigned __int8)v29 + 54)) & 1;
          if ( v29 == 357 || v29 == 186 || (_BYTE)v26 )
          {
            v27 = qword_4F8E610;
            v18 = *(_WORD *)(v6 + 2) & 0x3F;
            if ( !qword_4F8E610 )
              return 0;
            v19 = &unk_4F8E608;
            v20 = &unk_4F8E608;
            do
            {
              if ( v18 > *(_DWORD *)(v27 + 32) )
              {
                v27 = *(_QWORD *)(v27 + 24);
              }
              else
              {
                v20 = (_DWORD *)v27;
                v27 = *(_QWORD *)(v27 + 16);
              }
            }
            while ( v27 );
LABEL_24:
            if ( v20 != v19 && v18 >= v20[8] )
              goto LABEL_26;
            return 0;
          }
        }
      }
    }
  }
  v13 = *((_DWORD *)v7 + 8);
  v14 = (__int64)(v7 + 24);
  if ( v13 <= 0x40 )
  {
    v16 = *((_QWORD *)v7 + 3) == 0;
  }
  else
  {
    v15 = sub_C444A0(v14);
    v14 = (__int64)(v7 + 24);
    v16 = v13 == v15;
  }
  if ( v16 )
  {
    v17 = qword_4F8E6D0;
    v18 = *(_WORD *)(v6 + 2) & 0x3F;
    if ( !qword_4F8E6D0 )
      return 0;
    v19 = &unk_4F8E6C8;
    v20 = &unk_4F8E6C8;
    do
    {
      if ( v18 > *(_DWORD *)(v17 + 32) )
      {
        v17 = *(_QWORD *)(v17 + 24);
      }
      else
      {
        v20 = (_DWORD *)v17;
        v17 = *(_QWORD *)(v17 + 16);
      }
    }
    while ( v17 );
    goto LABEL_24;
  }
  if ( v13 > 0x40 )
  {
    if ( (unsigned int)sub_C444A0(v14) != v13 - 1 )
    {
      v28 = v13 == (unsigned int)sub_C445E0(v14);
      goto LABEL_60;
    }
LABEL_65:
    v30[0] = *(_WORD *)(v6 + 2) & 0x3F;
    v20 = (_DWORD *)sub_FF14C0((__int64)&unk_4F8E640, v30);
    if ( v20 == (_DWORD *)&unk_4F8E648 )
      return 0;
    goto LABEL_26;
  }
  if ( *((_QWORD *)v7 + 3) == 1 )
    goto LABEL_65;
  if ( v13 )
  {
    v28 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v13) == *((_QWORD *)v7 + 3);
LABEL_60:
    if ( !v28 )
      return 0;
  }
  v30[0] = *(_WORD *)(v6 + 2) & 0x3F;
  v20 = (_DWORD *)sub_FF14C0((__int64)&unk_4F8E680, v30);
  if ( v20 == (_DWORD *)&unk_4F8E688 )
    return 0;
LABEL_26:
  sub_FF6650(a1, a2, (__int64)(v20 + 10));
  return 1;
}
