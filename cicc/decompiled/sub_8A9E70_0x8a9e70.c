// Function: sub_8A9E70
// Address: 0x8a9e70
//
__int64 __fastcall sub_8A9E70(__int64 a1, char a2)
{
  char v2; // al
  char v4; // dl
  __int64 result; // rax
  __int64 v7; // rdi
  __int64 v8; // r14
  char v9; // al
  _BYTE *v10; // r15
  _BOOL4 v11; // r13d
  __int64 v12; // rax
  __int64 v13; // rdx
  unsigned __int8 v14; // cl
  int v15; // edx
  char v16; // si
  int v17; // eax
  int v18; // r8d
  char v19; // dl
  __int64 v20; // rdx
  __int64 v22; // r14
  __int64 v23; // rax
  bool v24; // cl
  int v25; // eax
  bool v26; // cl
  int v27; // eax
  __int64 v28; // rax
  __int64 v29; // rax
  bool v30; // si
  char v31; // cl
  char v32; // cl
  char v33; // al
  bool v34; // zf
  char v35; // [rsp-45h] [rbp-45h]
  int v36; // [rsp-44h] [rbp-44h]
  int v37; // [rsp-40h] [rbp-40h]
  _BOOL4 v38; // [rsp-40h] [rbp-40h]
  bool v39; // [rsp-40h] [rbp-40h]
  unsigned __int8 v40; // [rsp-3Ch] [rbp-3Ch]
  bool v41; // [rsp-3Ch] [rbp-3Ch]
  _BOOL4 v42; // [rsp-3Ch] [rbp-3Ch]
  int v43; // [rsp-3Ch] [rbp-3Ch]

  v2 = *(_BYTE *)(*(_QWORD *)(a1 + 16) + 28LL);
  if ( (v2 & 1) != 0 )
    return 0;
  v4 = *(_BYTE *)(a1 + 80);
  if ( (v4 & 0x20) != 0 )
    return 0;
  if ( (v4 & 8) == 0 )
  {
    if ( (*(_BYTE *)(a1 + 80) & 1) != 0 )
    {
      if ( !unk_4D04734 && !((v2 & 8) != 0 ? 1 : sub_891B10(a1)) )
        return 0;
    }
    else if ( unk_4D04734 != 1 )
    {
      return 0;
    }
  }
  v7 = *(_QWORD *)(a1 + 24);
  v8 = *(_QWORD *)(a1 + 32);
  v9 = *(_BYTE *)(v7 + 80);
  v10 = *(_BYTE **)(v7 + 88);
  if ( v9 == 9 )
  {
LABEL_17:
    if ( (unsigned __int8)(*(_BYTE *)(v8 + 80) - 19) <= 3u )
    {
      v13 = *(_QWORD *)(v8 + 88);
      if ( *(_QWORD *)(v13 + 88) )
      {
        if ( (*(_BYTE *)(v13 + 160) & 1) == 0 )
          v8 = *(_QWORD *)(v13 + 88);
      }
    }
    v14 = v10[170] >> 7;
    v40 = v14;
    v15 = v14;
    v11 = (*(_BYTE *)(v7 + 81) & 2) != 0;
    v16 = v10[172] & 0x20;
    if ( (*(_BYTE *)(v8 + 81) & 2) == 0 )
    {
      v37 = v14;
      v17 = sub_890400(a1);
      v15 = v37;
      if ( !v17 )
      {
        v18 = 0;
        if ( v16 )
          v18 = v10[176] & 1;
        if ( !v40 )
        {
          if ( (a2 & 1 & ((*(_BYTE *)(a1 + 80) >> 1) ^ 1)) == 0 )
            goto LABEL_27;
          if ( !unk_4D0472C )
          {
            v17 = 0;
            goto LABEL_27;
          }
          v36 = v18;
          v35 = a2 & 1 & ((*(_BYTE *)(a1 + 80) >> 1) ^ 1);
          sub_899B10(*(_QWORD *)(a1 + 32));
          v32 = 0;
          v17 = 0;
          v18 = v36;
          if ( (v10[172] & 0x20) == 0 )
          {
            v33 = *(_BYTE *)(v8 + 81) & 2;
            v34 = v33 == 0;
            LOBYTE(v17) = v33 != 0;
            if ( !v34 )
              v32 = v35;
            v17 = (unsigned __int8)v17;
          }
          v31 = ((*(_BYTE *)(a1 + 80) >> 1) ^ 1) & v32;
LABEL_72:
          if ( v31 )
            return 1;
LABEL_27:
          if ( v17 | v11 )
            return 0;
          goto LABEL_28;
        }
        goto LABEL_65;
      }
    }
    goto LABEL_32;
  }
  if ( v9 == 7 )
  {
    v12 = *(_QWORD *)(a1 + 40);
    if ( !v12 )
    {
      v12 = sub_8A9D50(v7, *(_QWORD *)(v8 + 88), 0);
      if ( !v12 )
        v12 = *(_QWORD *)(a1 + 32);
      *(_QWORD *)(a1 + 40) = v12;
      v7 = *(_QWORD *)(a1 + 24);
      *(_QWORD *)(a1 + 32) = v12;
    }
    v8 = v12;
    goto LABEL_17;
  }
  v11 = 0;
  if ( (v10[195] & 2) != 0 )
    v11 = (*(_BYTE *)(v7 + 81) & 2) != 0;
  switch ( *(_BYTE *)(v8 + 80) )
  {
    case 4:
    case 5:
      v22 = *(_QWORD *)(*(_QWORD *)(v8 + 96) + 80LL);
      break;
    case 6:
      v22 = *(_QWORD *)(*(_QWORD *)(v8 + 96) + 32LL);
      break;
    case 9:
    case 0xA:
      v22 = *(_QWORD *)(*(_QWORD *)(v8 + 96) + 56LL);
      break;
    case 0x13:
    case 0x14:
    case 0x15:
    case 0x16:
      v22 = *(_QWORD *)(v8 + 88);
      break;
    default:
      v22 = 0;
      break;
  }
  v38 = (v10[195] & 2) != 0;
  v41 = (v10[195] & 2) != 0;
  v23 = sub_892400(v22);
  v24 = v41;
  v15 = v38;
  if ( !*(_QWORD *)(v23 + 8) )
  {
    v42 = v38;
    v39 = v24;
    v25 = sub_890400(a1);
    v15 = v42;
    v18 = v25;
    if ( !v25 && (v10[206] & 0x18) == 0 )
    {
      v26 = v39;
      if ( !*(_QWORD *)(v22 + 104) )
        goto LABEL_49;
      v27 = sub_825090();
      v15 = v42;
      if ( !v27 )
      {
        v26 = v39;
        v18 = 0;
LABEL_49:
        if ( !v26 )
        {
          v19 = *(_BYTE *)(a1 + 80);
          if ( (v19 & 2) != 0 || (a2 & 1) == 0 || !unk_4D0472C )
          {
            if ( !v11 )
              goto LABEL_30;
            return 0;
          }
          v43 = v18;
          sub_899B10(*(_QWORD *)(a1 + 32));
          v29 = sub_892400(v22);
          v18 = v43;
          v30 = *(_QWORD *)(v29 + 8) != 0;
          v17 = v30;
          v31 = v30 & ((*(_BYTE *)(a1 + 80) >> 1) ^ 1);
          goto LABEL_72;
        }
LABEL_65:
        if ( !v11 )
        {
LABEL_28:
          if ( !v18 )
          {
            v19 = *(_BYTE *)(a1 + 80);
LABEL_30:
            if ( (v19 & 0x18) == 8 )
            {
              v20 = *(_QWORD *)(a1 + 32);
              switch ( *(_BYTE *)(v20 + 80) )
              {
                case 4:
                case 5:
                  v28 = *(_QWORD *)(*(_QWORD *)(v20 + 96) + 80LL);
                  break;
                case 6:
                  v28 = *(_QWORD *)(*(_QWORD *)(v20 + 96) + 32LL);
                  break;
                case 9:
                case 0xA:
                  v28 = *(_QWORD *)(*(_QWORD *)(v20 + 96) + 56LL);
                  break;
                case 0x13:
                case 0x14:
                case 0x15:
                case 0x16:
                  v28 = *(_QWORD *)(v20 + 88);
                  break;
                default:
                  BUG();
              }
              if ( (*(_BYTE *)(*(_QWORD *)(v28 + 104) + 121LL) & 3) != 1 && (*(_BYTE *)(a1 + 81) & 2) == 0 )
              {
                sub_6853B0(7u, 0x1E9u, (FILE *)(a1 + 84), *(_QWORD *)(a1 + 24));
                *(_BYTE *)(a1 + 81) |= 2u;
                return 0;
              }
            }
          }
          return 0;
        }
LABEL_33:
        if ( (*(_WORD *)(a1 + 80) & 0x208) == 8 )
        {
          sub_6853B0(unk_4F07471, 0x1EAu, (FILE *)(a1 + 84), *(_QWORD *)(a1 + 24));
          *(_BYTE *)(a1 + 81) |= 2u;
        }
        return 0;
      }
    }
  }
LABEL_32:
  if ( v15 )
    goto LABEL_33;
  result = 1;
  if ( (*(_BYTE *)(a1 + 80) & 2) != 0 )
    return 0;
  return result;
}
