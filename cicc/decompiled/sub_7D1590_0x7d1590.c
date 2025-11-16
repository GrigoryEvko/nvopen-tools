// Function: sub_7D1590
// Address: 0x7d1590
//
__int64 __fastcall sub_7D1590(char a1, int a2, __int64 *a3, __int64 a4)
{
  __int64 v4; // r8
  __int64 v5; // r15
  __int64 v8; // rdi
  __int64 v9; // rsi
  char v10; // al
  __int64 v11; // rbx
  __int64 v12; // r12
  int v13; // edx
  __int64 v14; // r14
  __int64 v15; // rax
  __int64 v16; // rax
  __int64 v17; // rsi
  char v18; // di
  char v19; // al
  _BOOL4 v20; // eax
  __int64 result; // rax
  _BOOL4 v22; // edx
  __int64 v23; // rsi
  __int64 v24; // rax
  char v25; // dl
  int v26; // eax
  int v28; // [rsp+Ch] [rbp-44h]
  __int64 v29; // [rsp+10h] [rbp-40h]
  __int64 v30; // [rsp+10h] [rbp-40h]

  v4 = (__int64)a3;
  v5 = 0;
  if ( a2 != -1 )
    v5 = qword_4F04C68[0] + 776LL * a2;
  v28 = *(_DWORD *)(a4 + 136);
  if ( *(_DWORD *)(a4 + 100) && (a1 & 0xEF) == 1 )
  {
    *(_DWORD *)(a4 + 136) = 0;
    if ( !*(_DWORD *)(a4 + 52) )
      goto LABEL_6;
LABEL_44:
    v12 = 0;
LABEL_45:
    if ( (*(_BYTE *)(a4 + 121) & 0x20) == 0 || (result = 0, !*(_DWORD *)(a4 + 24)) && !*(_DWORD *)(a4 + 32) )
    {
      result = 0;
      if ( (unsigned __int8)(a1 - 6) <= 1u && dword_4F077C4 == 2 )
      {
        *(_DWORD *)(a4 + 76) = 1;
        v22 = 1;
        if ( (dword_4F04C44 & unk_4F04C48) != -1 && !*(_DWORD *)(a4 + 92) )
        {
          v23 = *(_QWORD *)(v5 + 208);
          if ( (unsigned __int8)(*(_BYTE *)(v23 + 140) - 9) <= 2u )
            v22 = (*(_DWORD *)(v23 + 176) & 0x11000) != 4096;
        }
        *(_DWORD *)(a4 + 80) = v22;
        result = 0;
        *(_DWORD *)(a4 + 96) = 1;
        *(_QWORD *)(a4 + 112) = v12;
      }
    }
    goto LABEL_55;
  }
  if ( *(_DWORD *)(a4 + 52) )
    goto LABEL_44;
LABEL_6:
  v8 = *(_QWORD *)(v5 + 184);
  v9 = *a3;
  if ( unk_4D03F98 )
  {
    if ( v8 )
    {
      if ( *(_QWORD *)(v9 + 64) )
      {
        v10 = *(_BYTE *)(v8 + 28);
        if ( v10 == 3 || !v10 )
        {
          sub_824D70(v8);
          v4 = (__int64)a3;
          v9 = *a3;
        }
      }
    }
  }
  v5 = 0;
  if ( a2 != -1 )
    v5 = qword_4F04C68[0] + 776LL * a2;
  v11 = *(_QWORD *)(v9 + 24);
  if ( v11 )
  {
    v12 = 0;
    while ( 1 )
    {
      v13 = *(_DWORD *)(v11 + 40);
      if ( v13 == *(_DWORD *)v5 )
        break;
      v12 = v11;
      if ( !*(_QWORD *)(v11 + 8) )
        goto LABEL_66;
      v11 = *(_QWORD *)(v11 + 8);
    }
    v14 = 0;
    while ( 1 )
    {
      v16 = *(unsigned __int8 *)(v11 + 80);
      v17 = v11;
      v18 = *(_BYTE *)(v11 + 80);
      if ( (_BYTE)v16 == 16 )
      {
        v17 = **(_QWORD **)(v11 + 88);
        v18 = *(_BYTE *)(v17 + 80);
      }
      if ( v18 == 24 )
        v17 = *(_QWORD *)(v17 + 88);
      if ( v14 && *(_DWORD *)(v14 + 40) != v13 )
        goto LABEL_54;
      if ( dword_4F04BA0[v16] == *(_DWORD *)(a4 + 124) )
        break;
LABEL_19:
      v15 = *(_QWORD *)(v11 + 8);
      if ( !v15 )
      {
        v12 = v11;
        goto LABEL_54;
      }
      v13 = *(_DWORD *)(v15 + 40);
      v12 = v11;
      if ( v13 != *(_DWORD *)v5 )
        goto LABEL_54;
      v11 = *(_QWORD *)(v11 + 8);
    }
    v19 = *(_BYTE *)(v17 + 83);
    if ( (v19 & 0x40) != 0 || (*(_BYTE *)(v11 + 83) & 0x40) != 0 )
    {
      if ( !dword_4F077BC || (_DWORD)qword_4F077B4 || (unsigned __int64)(qword_4F077A8 - 50000LL) > 0x270F )
        goto LABEL_34;
      v25 = *(_BYTE *)(v17 + 80);
      if ( v25 != 20 )
      {
        if ( v25 == 17 )
        {
          v30 = v4;
          v26 = sub_8780F0(v17);
          v4 = v30;
          if ( v26 )
          {
LABEL_35:
            v19 = *(_BYTE *)(v17 + 83);
            goto LABEL_36;
          }
        }
LABEL_34:
        if ( !*(_DWORD *)(a4 + 24) && !*(_DWORD *)(a4 + 32) )
          goto LABEL_19;
        goto LABEL_35;
      }
    }
LABEL_36:
    if ( v19 >= 0
      || unk_4F04C48 == -1
      || (v24 = *(_QWORD *)(qword_4F04C68[0] + 776LL * unk_4F04C48 + 368), v17 != v24)
      || !v24 )
    {
      v29 = v4;
      v20 = sub_7CF0D0(v11, v17, (_DWORD *)a4);
      v4 = v29;
      if ( v20 )
      {
        if ( !*(_DWORD *)(a4 + 4) || *(_BYTE *)(v17 + 80) != 3 )
        {
          v14 = v11;
          goto LABEL_54;
        }
        if ( !v14 )
          v14 = v11;
      }
    }
    goto LABEL_19;
  }
LABEL_66:
  v12 = v11;
  v14 = 0;
LABEL_54:
  result = sub_7D1520(v5, v14, v4, (int *)a4);
  if ( !result )
    goto LABEL_45;
LABEL_55:
  *(_DWORD *)(a4 + 136) = v28;
  return result;
}
