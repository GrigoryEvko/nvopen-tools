// Function: sub_5D0090
// Address: 0x5d0090
//
__int64 __fastcall sub_5D0090(__int64 a1, _BYTE *a2)
{
  char *v2; // rbx
  size_t v3; // rcx
  char v4; // al
  size_t v5; // rbx
  const char *v6; // r14
  __int64 i; // r15
  int v8; // r12d
  const char *v9; // r13
  bool v10; // zf
  bool v11; // zf
  bool v12; // zf
  char *v13; // rax
  char v14; // al
  int v16; // eax
  _BYTE *v17; // r14
  __int64 v18; // r12
  char v19; // al
  char v20; // al
  _BYTE *v21; // r12
  unsigned int v22; // r15d
  __int64 v23; // rbx
  _BYTE *v24; // rax
  char v25; // dl
  __int64 v26; // rax
  __int64 v27; // rax
  __int64 v28; // [rsp+0h] [rbp-60h]
  int v29; // [rsp+Ch] [rbp-54h]
  size_t n; // [rsp+18h] [rbp-48h]
  char *s2; // [rsp+28h] [rbp-38h]

  v2 = *(char **)(*(_QWORD *)(a1 + 32) + 40LL);
  v28 = *(_QWORD *)(a1 + 32);
  s2 = v2;
  v3 = strlen(v2);
  n = v3;
  v4 = *v2;
  if ( v3 > 4 && v4 == 95 )
  {
    if ( v2[1] != 95 || v2[v3 - 1] != 95 || v2[v3 - 2] != 95 )
      goto LABEL_4;
    n = v3 - 4;
    v4 = v2[2];
    s2 = v2 + 2;
  }
  if ( v4 == 86 )
  {
    v16 = s2[1];
    if ( s2[1] == 49 )
    {
      if ( s2[2] == 54 )
      {
        v29 = 16;
        v6 = s2 + 3;
        v5 = n - 3;
        goto LABEL_5;
      }
    }
    else if ( (((_BYTE)v16 - 52) & 0xFB) != 0 && (unsigned __int8)(v16 - 49) > 1u )
    {
      goto LABEL_4;
    }
    v29 = v16 - 48;
    v6 = s2 + 2;
    v5 = n - 2;
    goto LABEL_5;
  }
LABEL_4:
  v5 = n;
  v6 = s2;
  v29 = 0;
LABEL_5:
  for ( i = 1; i != 15; ++i )
  {
    LOBYTE(v8) = i;
    v9 = (const char *)*(&off_4B6DF00 + i);
    if ( !strncmp(v9, v6, v5) && strlen(v9) == v5 )
    {
      if ( (_DWORD)i == 14 )
        LOBYTE(v8) = 13;
      goto LABEL_27;
    }
  }
  if ( memcmp("byte", s2, 4u) )
  {
    if ( !memcmp("word", s2, 4u) && n == 4 )
    {
      v8 = unk_4F06A00;
    }
    else
    {
LABEL_11:
      v10 = memcmp("unwind_word", s2, 0xBu) == 0;
      if ( n == 11 && v10 )
      {
        v8 = unk_4F069FF;
      }
      else
      {
        v11 = memcmp("libgcc_cmp_return", s2, 0x11u) == 0;
        if ( n == 17 && v11 )
        {
          v8 = unk_4F069FE;
        }
        else
        {
          v12 = memcmp("libgcc_shift_count", s2, 0x12u) == 0;
          if ( n == 18 && v12 )
          {
            v8 = unk_4F069FD;
          }
          else
          {
            if ( memcmp("pointer", s2, 7u) || n != 7 || !unk_4F06A70 )
            {
LABEL_19:
              v13 = sub_5C79F0(a1);
              sub_6851A0(1099, v28 + 24, v13);
              v14 = *(_BYTE *)(a1 + 10);
              *(_BYTE *)(a1 + 8) = 0;
              if ( (unsigned __int8)(v14 - 2) > 1u )
                return sub_72C930();
              return (__int64)a2;
            }
            v8 = unk_4F069FC;
          }
        }
      }
    }
    if ( v8 != 15 )
      goto LABEL_27;
    goto LABEL_19;
  }
  if ( n != 4 )
    goto LABEL_11;
  LOBYTE(v8) = 1;
LABEL_27:
  if ( (unsigned int)sub_8D3D40(a2) )
    return (__int64)a2;
  v18 = sub_5CFCE0((__int64)a2, v8, a1 + 56);
  if ( (unsigned __int8)(*(_BYTE *)(a1 + 10) - 2) > 1u )
  {
    v17 = (_BYTE *)v18;
    v23 = *(_QWORD *)(a1 + 48);
    if ( (unsigned int)sub_8D2870(a2) && (*(_BYTE *)(v23 + 8) & 0x20) != 0 )
    {
      v24 = a2;
      if ( a2[140] == 12 )
      {
        do
          v24 = (_BYTE *)*((_QWORD *)v24 + 20);
        while ( v24[140] == 12 );
        a2 = v24;
      }
      v17 = (_BYTE *)v18;
      if ( (a2[141] & 0x20) == 0 )
        a2[143] |= 8u;
    }
  }
  else
  {
    v17 = a2;
    if ( a2[140] == 2 && (a2[161] & 8) != 0 )
    {
      v25 = *(_BYTE *)(v18 + 140);
      if ( v25 == 12 )
      {
        v26 = v18;
        do
        {
          v26 = *(_QWORD *)(v26 + 160);
          v25 = *(_BYTE *)(v26 + 140);
        }
        while ( v25 == 12 );
      }
      v17 = a2;
      if ( v25 )
        a2[160] = *(_BYTE *)(v18 + 160);
    }
  }
  if ( !v29 )
    return (__int64)v17;
  while ( 1 )
  {
    v19 = *(_BYTE *)(v18 + 140);
    if ( v19 != 12 )
      break;
    v18 = *(_QWORD *)(v18 + 160);
  }
  if ( !v19 )
    return (__int64)v17;
  v20 = v17[140];
  if ( v20 == 12 )
  {
    v21 = v17;
    do
      v21 = (_BYTE *)*((_QWORD *)v21 + 20);
    while ( v21[140] == 12 );
    goto LABEL_40;
  }
  v21 = v17;
  if ( (v20 & 0xFB) == 8 )
  {
LABEL_40:
    v22 = sub_8D4C10(v17, unk_4F077C4 != 2);
    v20 = v21[140];
    goto LABEL_41;
  }
  v22 = 0;
LABEL_41:
  if ( (unsigned __int8)(v20 - 2) > 1u )
  {
    *(_BYTE *)(a1 + 8) = 0;
    return (__int64)v17;
  }
  if ( (unsigned __int8)(*(_BYTE *)(a1 + 10) - 2) <= 1u )
  {
    sub_684AA0(qword_4F077A8 < 0x9C40u ? 5 : 8, 1923, a1 + 56);
    *(_BYTE *)(a1 + 8) = 0;
    return (__int64)v17;
  }
  v27 = sub_72B5A0(v21, v29, 0);
  *(_QWORD *)(v27 + 64) = *(_QWORD *)(a1 + 56);
  return sub_73C570(v27, v22, -1);
}
