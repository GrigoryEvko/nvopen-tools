// Function: sub_644100
// Address: 0x644100
//
__int64 __fastcall sub_644100(__int64 a1, __int64 a2)
{
  char v3; // r14
  int v4; // r13d
  __int64 v5; // rbx
  __int64 v6; // r8
  __int64 v7; // rdi
  __int64 v8; // r8
  unsigned __int64 v9; // rax
  char v10; // al
  char v11; // al
  __int64 result; // rax
  int v13; // edx
  char v14; // al
  __int64 v15; // rax
  __int64 v16; // rsi
  char v17; // al
  __int64 v18; // rdi
  char v19; // al
  __int64 v20; // rdi
  int v21; // eax
  char v22; // al
  __int64 v23; // rax
  __int64 v24; // rax
  char v25; // bl
  __int64 v26; // rax
  int v27; // [rsp+8h] [rbp-38h]
  __int64 v28; // [rsp+8h] [rbp-38h]
  char v29; // [rsp+8h] [rbp-38h]

  v3 = *(_BYTE *)(a1 + 44);
  v4 = unk_4D03F98;
  unk_4D03F98 = 0;
  if ( (*(_BYTE *)(*(_QWORD *)a1 + 17LL) & 0x20) != 0 )
    goto LABEL_18;
  v5 = *(_QWORD *)(a1 + 48);
  v6 = a2;
  if ( !v5 )
  {
    v11 = *(_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 4);
    if ( (((v11 - 15) & 0xFD) == 0 || v11 == 2) && v3 != 1 )
      goto LABEL_18;
  }
  if ( *(_BYTE *)(qword_4F04C68[0] + 776LL * (int)dword_4F04C5C + 4) == 1 )
    goto LABEL_18;
  if ( *(_DWORD *)(a1 + 40) != dword_4F04C34 && (*(_BYTE *)(a1 + 64) & 0x30) == 0 )
  {
    *(_DWORD *)(a1 + 80) = 0;
    v27 = 0;
    goto LABEL_7;
  }
  if ( v3 == 2 )
    goto LABEL_65;
  if ( dword_4F077C4 != 2 )
    goto LABEL_26;
  if ( v5 )
  {
    if ( dword_4D04824 || (*(_BYTE *)(v5 + 64) & 2) == 0 )
      goto LABEL_59;
LABEL_65:
    *(_DWORD *)(a1 + 80) = 1;
    v27 = 0;
    goto LABEL_7;
  }
  v18 = *(_QWORD *)(a1 + 56);
  if ( (*(_BYTE *)(v18 + 140) & 0xFB) != 8 )
  {
LABEL_59:
    if ( unk_4F07778 <= 201102 && !dword_4F07774 )
      goto LABEL_26;
    v17 = *(_BYTE *)(a1 + 64);
    if ( (v17 & 0x40) == 0 )
      goto LABEL_26;
    if ( !v5 )
    {
      v28 = v6;
      v24 = sub_6440B0(0x5Au, v6);
      v6 = v28;
      if ( v24 && v3 == 1 )
        goto LABEL_26;
      v17 = *(_BYTE *)(a1 + 64);
    }
    if ( v17 < 0 && HIDWORD(qword_4F077B4) )
    {
LABEL_26:
      *(_DWORD *)(a1 + 80) = 2;
      v27 = 0;
      goto LABEL_7;
    }
    goto LABEL_65;
  }
  v19 = sub_8D4C10(v18, 0);
  v6 = a2;
  if ( (v19 & 2) != 0 )
    goto LABEL_77;
  v20 = *(_QWORD *)(a1 + 56);
  v21 = dword_4F077C4;
  if ( (*(_BYTE *)(v20 + 140) & 0xFB) != 8 )
    goto LABEL_78;
  v22 = sub_8D4C10(v20, dword_4F077C4 != 2);
  v6 = a2;
  if ( (v22 & 1) == 0
    || (*(_BYTE *)(a2 + 8) & 2) != 0
    || (*(_BYTE *)(a2 + 134) & 1) != 0
    || dword_4F04C5C != dword_4F04C34
    || *(_BYTE *)(a1 + 44)
    || (*(_WORD *)(a1 + 64) & 0x180) == 0x180 )
  {
LABEL_77:
    v21 = dword_4F077C4;
LABEL_78:
    if ( v21 != 2 )
      goto LABEL_26;
    goto LABEL_59;
  }
  *(_DWORD *)(a1 + 80) = 1;
  v27 = 1;
LABEL_7:
  v7 = a1;
  sub_642710((__int64 *)a1, v6);
  v8 = *(_QWORD *)(a1 + 8);
  if ( v8 )
  {
    LOBYTE(v9) = *(_BYTE *)(v8 + 80);
  }
  else
  {
    v8 = *(_QWORD *)(a1 + 24);
    if ( !v8 )
      goto LABEL_18;
    v9 = *(unsigned __int8 *)(v8 + 80);
    if ( (unsigned __int8)v9 > 0x14u )
      goto LABEL_18;
    v16 = 1181824;
    if ( !_bittest64(&v16, v9) )
      goto LABEL_18;
  }
  if ( (v5 == 0) == ((_BYTE)v9 == 7) )
  {
    if ( (unsigned __int8)(v9 - 10) <= 1u && *(_QWORD *)(v8 + 96) )
    {
      if ( (*(_BYTE *)(a1 + 64) & 8) == 0 )
      {
        v15 = *(_QWORD *)(v8 + 88);
        if ( (*(_BYTE *)(v15 + 193) & 0x20) == 0 && !*(_DWORD *)(v15 + 160) && !*(_QWORD *)(v15 + 344) )
        {
          v25 = *(_BYTE *)(v15 + 192) >> 7;
          v29 = *(_BYTE *)(v15 + 172);
          if ( v29 == 2 || v3 != 2 )
          {
            v26 = *(_QWORD *)(a1 + 48);
            if ( (*(_BYTE *)(v26 + 64) & 2) != 0 && !v25 )
            {
              sub_685490(657, *(_QWORD *)a1 + 8LL, v8);
              v26 = *(_QWORD *)(a1 + 48);
            }
          }
          else
          {
            sub_685490(553, *(_QWORD *)a1 + 8LL, v8);
            v26 = *(_QWORD *)(a1 + 48);
          }
          v3 = v29;
          *(_BYTE *)(v26 + 64) = (2 * v25) | *(_BYTE *)(v26 + 64) & 0xFD;
        }
      }
      *(_BYTE *)(a1 + 44) = v3;
      if ( v3 == 2 || !dword_4D04824 && (*(_BYTE *)(*(_QWORD *)(a1 + 48) + 64LL) & 2) != 0 )
        goto LABEL_50;
      goto LABEL_13;
    }
    if ( v27 && *(_BYTE *)(*(_QWORD *)(v8 + 88) + 136LL) != 2 )
    {
LABEL_13:
      v10 = *(_BYTE *)(a1 + 64);
LABEL_14:
      *(_DWORD *)(a1 + 80) = 2;
      goto LABEL_67;
    }
    if ( v3 == 1 || v5 && !v3 )
    {
      v13 = *(_DWORD *)(qword_4F04C68[0] + 776LL * dword_4F04C34);
      if ( *(_DWORD *)(v8 + 40) == v13 )
        goto LABEL_33;
      if ( *(_DWORD *)(v8 + 40) <= v13 )
        goto LABEL_18;
      if ( (_BYTE)v9 == 7 )
      {
        v23 = *(_QWORD *)(v8 + 88);
        if ( (*(_BYTE *)(v23 + 89) & 1) != 0 )
          goto LABEL_18;
      }
      else
      {
LABEL_33:
        if ( (_BYTE)v9 == 11 )
        {
          v14 = *(_BYTE *)(*(_QWORD *)(v8 + 88) + 172LL);
          goto LABEL_36;
        }
        if ( (_BYTE)v9 == 20 )
        {
          v14 = *(_BYTE *)(*(_QWORD *)(*(_QWORD *)(v8 + 88) + 176LL) + 172LL);
          goto LABEL_36;
        }
        if ( (_BYTE)v9 != 7 )
        {
          LOBYTE(v7) = v5 == 0;
          sub_721090(v7);
        }
        v23 = *(_QWORD *)(v8 + 88);
      }
      v14 = *(_BYTE *)(v23 + 136);
LABEL_36:
      if ( v14 == 2 )
      {
LABEL_50:
        *(_DWORD *)(a1 + 80) = 1;
        goto LABEL_51;
      }
      if ( v14 == 1 )
      {
        v10 = *(_BYTE *)(a1 + 64);
        if ( (v10 & 0x40) != 0 && dword_4F077BC && (*(_BYTE *)(v8 + 81) & 4) != 0 )
          goto LABEL_14;
      }
    }
  }
LABEL_18:
  result = *(unsigned int *)(a1 + 80);
  if ( (_DWORD)result )
  {
    if ( (_DWORD)result == 1 )
    {
LABEL_51:
      *(_BYTE *)(a1 + 44) = 2;
      goto LABEL_23;
    }
    if ( (_DWORD)result != 2 )
    {
LABEL_23:
      result = sub_6418E0(a1);
      goto LABEL_20;
    }
    v10 = *(_BYTE *)(a1 + 64);
LABEL_67:
    *(_BYTE *)(a1 + 44) = (v10 & 8) == 0;
    goto LABEL_23;
  }
  *(_QWORD *)(a1 + 8) = 0;
LABEL_20:
  unk_4D03F98 = v4;
  return result;
}
