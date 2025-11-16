// Function: sub_8779B0
// Address: 0x8779b0
//
__int64 __fastcall sub_8779B0(
        __int64 a1,
        unsigned int (__fastcall *a2)(__int64, __int64),
        unsigned int (__fastcall *a3)(__int64, __int64))
{
  __int64 v3; // rdx
  __int64 v5; // r13
  __int64 v6; // r14
  __int64 v7; // rdi
  __int64 v8; // rdi
  __int64 v9; // rdx
  char v10; // al
  __int64 v11; // r15
  bool v12; // si
  __int64 v13; // r9
  unsigned int v14; // eax
  __int64 v15; // r9
  __int64 v17; // rax
  __int64 v18; // rdi
  __int64 v19; // rax
  __int64 v20; // rax
  __int64 v21; // rax
  unsigned int v22; // eax
  __int64 v23; // rsi
  __int64 v24; // rdi
  __int64 v25; // [rsp+0h] [rbp-50h]
  __int64 v26; // [rsp+8h] [rbp-48h]
  __int64 v27; // [rsp+8h] [rbp-48h]
  __int64 v28; // [rsp+8h] [rbp-48h]

  v3 = (int)dword_4F04C2C;
  if ( dword_4F04C2C == -1 )
    return 0;
  v5 = 0;
  v6 = 0;
  while ( 1 )
  {
    while ( 1 )
    {
      v9 = qword_4F04C68[0] + 776 * v3;
      v10 = *(_BYTE *)(v9 + 4);
      v11 = v9;
      v12 = v10 == 14 || v10 == 17;
      if ( v12 )
        break;
      if ( (unsigned __int8)(v10 - 9) <= 1u )
        goto LABEL_13;
      v13 = *(_QWORD *)(v9 + 208);
      if ( !v6 )
        goto LABEL_40;
      if ( !v5 || (*(_BYTE *)(v5 + 194) & 0x40) == 0 )
      {
        if ( v13 != v6 )
        {
          if ( !v13 )
            goto LABEL_13;
          if ( !dword_4F07588 )
            goto LABEL_13;
          v20 = *(_QWORD *)(v13 + 32);
          if ( *(_QWORD *)(v6 + 32) != v20 || !v20 )
            goto LABEL_13;
        }
LABEL_40:
        v27 = *(_QWORD *)(v9 + 208);
        if ( a3(a1, v9) )
          return 1;
        v15 = v27;
        if ( !v5 || (*(_BYTE *)(v5 + 194) & 0x40) == 0 )
          goto LABEL_21;
        goto LABEL_43;
      }
      v26 = *(_QWORD *)(v9 + 208);
      v14 = a3(a1, v9);
      v15 = v26;
      if ( v14 )
        return 1;
      if ( (*(_BYTE *)(v5 + 194) & 0x40) == 0 )
        goto LABEL_21;
LABEL_43:
      v21 = v5;
      do
        v21 = *(_QWORD *)(v21 + 232);
      while ( (*(_BYTE *)(v21 + 194) & 0x40) != 0 );
      v25 = v15;
      v28 = *(_QWORD *)(v11 + 208);
      *(_QWORD *)(v11 + 208) = *(_QWORD *)(*(_QWORD *)(v21 + 40) + 32LL);
      v22 = a3(a1, v11);
      v15 = v25;
      *(_QWORD *)(v11 + 208) = v28;
      if ( v22 )
        return 1;
LABEL_21:
      v6 = 0;
      if ( (*(_BYTE *)(v15 + 89) & 4) == 0 )
        goto LABEL_13;
      v3 = *(int *)(v11 + 448);
      v6 = *(_QWORD *)(*(_QWORD *)(v15 + 40) + 32LL);
      if ( (_DWORD)v3 == -1 )
        return 0;
    }
    if ( v10 == 14 )
    {
      v7 = *(_QWORD *)(v9 + 368);
      v5 = *(_QWORD *)(v9 + 216);
      if ( v7 )
      {
        switch ( *(_BYTE *)(v7 + 80) )
        {
          case 4:
          case 5:
            v19 = *(_QWORD *)(*(_QWORD *)(v7 + 96) + 80LL);
            break;
          case 6:
            v19 = *(_QWORD *)(*(_QWORD *)(v7 + 96) + 32LL);
            break;
          case 9:
          case 0xA:
            v19 = *(_QWORD *)(*(_QWORD *)(v7 + 96) + 56LL);
            break;
          case 0x13:
          case 0x14:
          case 0x15:
          case 0x16:
            v19 = *(_QWORD *)(v7 + 88);
            break;
          default:
            BUG();
        }
        v7 = *(_QWORD *)(v19 + 72);
        v12 = v7 == 0;
      }
    }
    else
    {
      v7 = 0;
      v5 = *(_QWORD *)(*(_QWORD *)(v9 + 184) + 32LL);
    }
    if ( v5 && v12 )
      break;
    if ( a2(v7, a1) )
      return 1;
    if ( v5 )
    {
      if ( (*(_BYTE *)(v5 + 194) & 0x40) != 0 )
        goto LABEL_27;
      goto LABEL_11;
    }
LABEL_13:
    v3 = *(int *)(v11 + 448);
    if ( (_DWORD)v3 == -1 )
      return 0;
  }
  v8 = 0;
  if ( (*(_BYTE *)(v5 + 194) & 0x40) == 0 )
    v8 = *(_QWORD *)(v5 + 232);
  if ( !a2(v8, a1) )
  {
    if ( (*(_BYTE *)(v5 + 194) & 0x40) != 0 )
    {
LABEL_27:
      v17 = v5;
      do
        v17 = *(_QWORD *)(v17 + 232);
      while ( (*(_BYTE *)(v17 + 194) & 0x40) != 0 );
      if ( (*(_BYTE *)(v17 + 195) & 1) == 0 )
        goto LABEL_49;
      v18 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)v17 + 96LL) + 32LL);
      switch ( *(_BYTE *)(v18 + 80) )
      {
        case 4:
        case 5:
          v23 = *(_QWORD *)(*(_QWORD *)(v18 + 96) + 80LL);
          break;
        case 6:
          v23 = *(_QWORD *)(*(_QWORD *)(v18 + 96) + 32LL);
          break;
        case 9:
        case 0xA:
          v23 = *(_QWORD *)(*(_QWORD *)(v18 + 96) + 56LL);
          break;
        case 0x13:
        case 0x14:
        case 0x15:
        case 0x16:
          v23 = *(_QWORD *)(v18 + 88);
          break;
        default:
          BUG();
      }
      v24 = *(_QWORD *)(v23 + 72);
      if ( !v24 )
LABEL_49:
        v24 = *(_QWORD *)(v17 + 232);
      if ( a2(v24, a1) )
        return 1;
    }
LABEL_11:
    if ( (*(_BYTE *)(v5 + 89) & 4) == 0 )
      return 0;
    v6 = *(_QWORD *)(*(_QWORD *)(v5 + 40) + 32LL);
    goto LABEL_13;
  }
  return 1;
}
