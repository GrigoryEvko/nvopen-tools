// Function: sub_890C40
// Address: 0x890c40
//
__int64 __fastcall sub_890C40(__int64 a1, __int64 a2, __int64 a3, int a4, char a5)
{
  __int64 v9; // r14
  __int64 v10; // rax
  char v11; // al
  char v13; // r13
  __int64 v15; // rsi
  __int64 v16; // rdx
  __int64 v17; // rcx
  __int64 v18; // r8
  __int64 v19; // r9
  __int64 v20; // r14
  __int64 v21; // rsi
  __int64 v22; // r14
  __int64 v23; // rax
  int v24; // eax
  __int64 v25; // [rsp+8h] [rbp-48h]
  int v26; // [rsp+14h] [rbp-3Ch]

  v9 = qword_4F04C68[0];
  v26 = *(_DWORD *)(a1 + 204);
  v25 = *(_QWORD *)(qword_4F04C68[0] + 776LL * dword_4F04C34 + 224);
  v10 = sub_8807C0(a2);
  if ( (*(_BYTE *)(a3 + 18) & 2) == 0 && v25 == v10 )
  {
    if ( v10 )
    {
      v11 = *(_BYTE *)(a2 + 80);
      if ( (unsigned __int8)(v11 - 4) > 1u && v11 != 19 && !*(_DWORD *)(a1 + 16) )
      {
        sub_6851C0(0x2F4u, (_DWORD *)(a3 + 8));
        goto LABEL_17;
      }
    }
  }
  if ( !a4 && !*(_QWORD *)(a1 + 24) )
  {
    if ( *(_DWORD *)(a1 + 16) || dword_4F077BC && qword_4F077A8 <= 0x76BFu )
      return 0;
    sub_6854C0(0x2F3u, (FILE *)(a3 + 8), a2);
    goto LABEL_17;
  }
  v13 = (*(_QWORD *)(a1 + 240) != 0) & (a5 ^ 1);
  if ( v13 )
  {
    sub_6851C0(0x1ABu, (_DWORD *)(a3 + 8));
LABEL_17:
    *(_DWORD *)(a1 + 52) = 1;
    return 1;
  }
  v15 = v9 + 776LL * v26;
  if ( !(unsigned int)sub_85ED80(a2, v15) )
  {
    if ( *(_DWORD *)(a1 + 28) )
      sub_6853B0(8u, 0x94Eu, (FILE *)(a3 + 8), a2);
    else
      sub_6854C0(0x227u, (FILE *)(a3 + 8), a2);
    goto LABEL_17;
  }
  v20 = *(_QWORD *)(a3 + 32);
  if ( (*(_BYTE *)(a3 + 18) & 2) != 0 )
  {
    if ( *(_BYTE *)(v20 + 140) == 14 )
      v20 = sub_7CFE40(*(_QWORD *)(a3 + 32), v15, v16, v17, v18, v19);
    v21 = *(_QWORD *)(a2 + 64);
    if ( v21 != v20 )
      v13 = (unsigned int)sub_8D97D0(v20, v21, 0, v17, v18) == 0;
    v22 = *(_QWORD *)v20;
  }
  else
  {
    if ( !v20 )
    {
LABEL_27:
      switch ( *(_BYTE *)(a2 + 80) )
      {
        case 4:
        case 5:
          v23 = *(_QWORD *)(*(_QWORD *)(a2 + 96) + 80LL);
          break;
        case 6:
          v23 = *(_QWORD *)(*(_QWORD *)(a2 + 96) + 32LL);
          break;
        case 9:
        case 0xA:
          v23 = *(_QWORD *)(*(_QWORD *)(a2 + 96) + 56LL);
          break;
        case 0x13:
        case 0x14:
        case 0x15:
        case 0x16:
          v23 = *(_QWORD *)(a2 + 88);
          break;
        default:
          return 0;
      }
      if ( !v23 || (*(_BYTE *)(v23 + 160) & 2) == 0 )
        return 0;
      sub_6854C0(0x3F1u, (FILE *)(a3 + 8), a2);
      goto LABEL_17;
    }
    v24 = 1;
    if ( *(_QWORD *)(a2 + 64) != v20 )
      v24 = sub_880800(a2, *(_QWORD *)(v20 + 128));
    v22 = *(_QWORD *)v20;
    v13 = v24 == 0;
  }
  if ( !v22 || !v13 || (unsigned int)sub_880920(a2) )
    goto LABEL_27;
  if ( dword_4F077BC && !(_DWORD)qword_4F077B4 )
    return 0;
  sub_686A10(0x2E6u, (_DWORD *)(a3 + 8), *(_QWORD *)(*(_QWORD *)a3 + 8LL), v22);
  return 0;
}
