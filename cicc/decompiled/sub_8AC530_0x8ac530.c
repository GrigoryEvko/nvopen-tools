// Function: sub_8AC530
// Address: 0x8ac530
//
__int64 __fastcall sub_8AC530(__int64 a1, int a2, char a3)
{
  char v3; // cl
  __int64 v7; // rsi
  __int64 v8; // rdx
  __int64 v9; // r9
  __int64 result; // rax
  int v11; // r13d
  _BOOL4 v12; // esi
  char v13; // di
  __int64 v14; // r8
  int v15; // r12d
  __int64 v16; // r12
  bool v17; // zf
  __int64 v18; // rdi
  int v19; // eax
  char v20; // al
  char v21; // r13
  __int64 v22; // r8
  __int64 v23; // rdx
  int v24; // r13d
  __int64 v25; // rsi
  __int64 v26; // [rsp+0h] [rbp-50h]
  __int64 v27; // [rsp+0h] [rbp-50h]
  __int64 v28; // [rsp+8h] [rbp-48h]
  __int64 v29; // [rsp+8h] [rbp-48h]
  __int64 v30; // [rsp+10h] [rbp-40h]
  char v31; // [rsp+18h] [rbp-38h]
  __int64 v32; // [rsp+18h] [rbp-38h]
  __int64 v33; // [rsp+18h] [rbp-38h]
  __int64 v34; // [rsp+18h] [rbp-38h]
  char v35; // [rsp+18h] [rbp-38h]
  __int64 v36; // [rsp+18h] [rbp-38h]
  __int64 v37; // [rsp+18h] [rbp-38h]

  v3 = a3 & 1;
  v7 = *(_QWORD *)(a1 + 32);
  v8 = *(_QWORD *)(a1 + 24);
  switch ( *(_BYTE *)(v7 + 80) )
  {
    case 4:
    case 5:
      v9 = *(_QWORD *)(*(_QWORD *)(v7 + 96) + 80LL);
      break;
    case 6:
      v9 = *(_QWORD *)(*(_QWORD *)(v7 + 96) + 32LL);
      break;
    case 9:
    case 0xA:
      v9 = *(_QWORD *)(*(_QWORD *)(v7 + 96) + 56LL);
      break;
    case 0x13:
    case 0x14:
    case 0x15:
    case 0x16:
      v9 = *(_QWORD *)(v7 + 88);
      break;
    default:
      v9 = 0;
      break;
  }
  result = *(unsigned __int8 *)(v8 + 80);
  v11 = a3 & 8;
  if ( (unsigned __int8)(result - 10) > 1u )
  {
    if ( (a3 & 8) != 0 )
    {
      v12 = 0;
      v11 = 0;
      if ( (_BYTE)result != 17 )
        goto LABEL_6;
      v16 = *(_QWORD *)(v8 + 88);
      v31 = v3;
      if ( (*(_BYTE *)(v16 + 89) & 4) == 0 )
        goto LABEL_39;
LABEL_33:
      v18 = *(_QWORD *)(*(_QWORD *)(v16 + 40) + 32LL);
      if ( v18 )
      {
        v26 = v9;
        v28 = v8;
        v19 = sub_8D23B0(v18);
        v8 = v28;
        v9 = v26;
        if ( v19 )
        {
          v3 = v31;
          if ( dword_4F077BC && !*(_QWORD *)(*(_QWORD *)(*(_QWORD *)(v16 + 152) + 168LL) + 40LL) )
            v3 = 0;
LABEL_6:
          v13 = 0;
          v14 = 0;
          result = v11 | (unsigned int)dword_4F601E0 | dword_4D03FE8[0];
          v15 = v11 | dword_4F601E0 | dword_4D03FE8[0];
          if ( !v15 )
            goto LABEL_7;
          goto LABEL_25;
        }
      }
LABEL_39:
      v3 = 0;
      goto LABEL_6;
    }
LABEL_40:
    v12 = 0;
    goto LABEL_6;
  }
  v16 = *(_QWORD *)(v8 + 88);
  if ( (*(_BYTE *)(v16 + 193) & 2) == 0 )
  {
    if ( dword_4F077BC )
      v3 = 1;
    if ( (a3 & 8) == 0 )
      goto LABEL_40;
    v12 = 0;
    v11 = 0;
LABEL_32:
    v31 = v3;
    if ( (*(_BYTE *)(v16 + 89) & 4) == 0 )
      goto LABEL_39;
    goto LABEL_33;
  }
  v17 = v11 == 0;
  v11 = 1;
  v12 = !v17;
  if ( !v17 )
    goto LABEL_32;
LABEL_25:
  v14 = *(_QWORD *)(a1 + 16);
  if ( !v14 )
  {
    v27 = v9;
    v29 = v8;
    v35 = v3;
    result = sub_892270((_QWORD *)a1);
    v14 = *(_QWORD *)(a1 + 16);
    v9 = v27;
    v8 = v29;
    v3 = v35;
  }
  v15 = v11;
  v13 = 1;
LABEL_7:
  if ( *(char *)(v9 + 160) < 0 )
    return result;
  result = dword_4D04734;
  if ( dword_4D04734 == 4 )
    goto LABEL_17;
  if ( dword_4D04734 == 1 )
  {
    if ( !a2 )
    {
      result = *(unsigned __int8 *)(v8 + 80);
      if ( (unsigned __int8)(result - 10) <= 1u || (_BYTE)result == 17 )
      {
        result = *(unsigned __int16 *)(v9 + 384);
        if ( unk_4F04C48 != -1 )
        {
          result = (unsigned int)(result + 1);
          v25 = *(_QWORD *)(a1 + 32);
          *(_WORD *)(v9 + 384) = result;
          if ( (_WORD)result == 200 )
          {
            v37 = v9;
            sub_6854E0(0x257u, v25);
            result = *(unsigned __int16 *)(v37 + 384);
          }
        }
        if ( (__int16)result > 199 )
          return result;
      }
LABEL_17:
      if ( (*(_BYTE *)(a1 + 81) & 1) != 0 )
      {
        result = (unsigned int)dword_4F601E0;
        if ( dword_4F601E0 && (*(_BYTE *)(a1 + 80) & 1) != 0 )
        {
          result = *(_QWORD *)(a1 + 16);
          if ( (*(_BYTE *)(result + 28) & 1) == 0 )
            return sub_899AF0();
        }
        return result;
      }
      goto LABEL_43;
    }
    if ( v8 == *(_QWORD *)(a1 + 32) )
      return result;
  }
  else
  {
    if ( *(_QWORD *)(a1 + 32) == v8 )
      return result;
    if ( !a2 )
    {
      if ( (a3 & 2) != 0 )
      {
        result = (__int64)dword_4D03FE8;
        if ( dword_4D03FE8[0] )
        {
          *(_BYTE *)(v14 + 28) &= ~4u;
          *(_DWORD *)(v14 + 24) = 0;
        }
        else if ( (*(_BYTE *)(a1 + 80) & 1) != 0 )
        {
          --*(_DWORD *)(v14 + 24);
        }
        *(_BYTE *)(a1 + 80) &= ~1u;
      }
      goto LABEL_17;
    }
  }
  if ( !v12 && qword_4D03B78 | *(_QWORD *)(qword_4F04C68[0] + 776LL * unk_4F04C24 + 720) )
  {
    v32 = v8;
    result = (__int64)sub_878440();
    v17 = qword_4F601B8 == 0;
    *(_QWORD *)(result + 8) = v32;
    if ( v17 )
      qword_4F601B8 = result;
    if ( qword_4F601B0 )
      *(_QWORD *)qword_4F601B0 = result;
    qword_4F601B0 = result;
    goto LABEL_17;
  }
  v20 = *(_BYTE *)(a1 + 80);
  v21 = v13 & (v3 ^ 1);
  *(_BYTE *)(a1 + 80) = v20 | 1;
  result = v20 & 1;
  if ( (_DWORD)result )
  {
    if ( !v21 )
      return result;
    v34 = v8;
    v24 = 0;
    result = sub_8919F0(a1, 0);
    v23 = v34;
    if ( !(_DWORD)result )
      return result;
  }
  else
  {
    if ( v13 )
      ++*(_DWORD *)(v14 + 24);
    v30 = v8;
    v33 = v14;
    result = sub_893360();
    v22 = v33;
    *(_QWORD *)(a1 + 48) = result;
    if ( !v21 )
    {
      if ( !dword_4F601E0 )
      {
        if ( (*(_BYTE *)(a1 + 81) & 1) != 0 )
          return result;
LABEL_43:
        if ( qword_4F601F0 )
        {
          result = qword_4F601E8;
          *(_QWORD *)(qword_4F601E8 + 8) = a1;
        }
        else
        {
          qword_4F601F0 = a1;
        }
        qword_4F601E8 = a1;
        *(_BYTE *)(a1 + 81) |= 1u;
        return result;
      }
LABEL_66:
      if ( (*(_BYTE *)(v22 + 28) & 1) == 0 && (a3 & 4) == 0 )
      {
        result = sub_8A9E70(a1, 0);
        if ( (_DWORD)result )
          result = sub_8AB5A0(a1);
      }
      goto LABEL_17;
    }
    result = sub_8919F0(a1, 0);
    v23 = v30;
    if ( !(_DWORD)result )
    {
      if ( !dword_4F601E0 )
        goto LABEL_17;
      goto LABEL_66;
    }
    v24 = 1;
  }
  if ( (*(_BYTE *)(v22 + 28) & 1) == 0 && (v36 = v23, result = sub_8A9E70(a1, 0), v23 = v36, (_DWORD)result)
    || v15 && (result = (__int64)qword_4D03FD0, *qword_4D03FD0) && (*(_BYTE *)(v23 + 81) & 2) == 0 )
  {
    result = sub_8AB5A0(a1);
  }
  if ( v24 )
    goto LABEL_17;
  return result;
}
