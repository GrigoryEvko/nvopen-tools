// Function: sub_97FAA0
// Address: 0x97faa0
//
bool __fastcall sub_97FAA0(__int64 a1, __int64 a2, unsigned int a3, __int64 *a4)
{
  __int64 v6; // rbx
  int v7; // r13d
  unsigned int v8; // r13d
  _QWORD *v10; // rdx
  __int64 v11; // rsi
  unsigned __int8 v12; // al
  __int64 v13; // rax
  unsigned int v14; // eax
  int v15; // ecx
  __int64 v16; // r12
  _BYTE *v17; // r9
  __int64 v18; // rbx
  char v19; // al
  __int64 v20; // rsi
  int v21; // esi
  __int64 v22; // rdx
  _QWORD *v23; // rdx
  __int64 v24; // rax
  __int64 v25; // rcx
  char v26; // dl
  _QWORD *v27; // rax
  char v28; // al
  __int64 v29; // rdx
  unsigned __int64 v30; // rax
  unsigned __int8 v31; // al
  __int64 v32; // r12
  int v33; // [rsp+Ch] [rbp-54h]
  int v34; // [rsp+Ch] [rbp-54h]
  int v35; // [rsp+Ch] [rbp-54h]
  int v36; // [rsp+Ch] [rbp-54h]
  int v37; // [rsp+Ch] [rbp-54h]
  int v38; // [rsp+Ch] [rbp-54h]
  int v39; // [rsp+Ch] [rbp-54h]
  unsigned int v40; // [rsp+10h] [rbp-50h]
  unsigned int v41; // [rsp+14h] [rbp-4Ch]
  _BYTE *v42; // [rsp+18h] [rbp-48h]
  _BYTE *v43; // [rsp+18h] [rbp-48h]
  _BYTE *v44; // [rsp+18h] [rbp-48h]
  _BYTE *v45; // [rsp+18h] [rbp-48h]
  _BYTE *v46; // [rsp+18h] [rbp-48h]
  _BYTE *v47; // [rsp+18h] [rbp-48h]
  _BYTE *v48; // [rsp+18h] [rbp-48h]
  __int64 v49; // [rsp+20h] [rbp-40h] BYREF
  __int64 v50; // [rsp+28h] [rbp-38h]

  v6 = a3;
  v7 = *(_DWORD *)(a2 + 12);
  if ( a3 > 0x82 )
  {
    if ( a3 - 189 <= 2 )
    {
      v10 = *(_QWORD **)(a2 + 16);
      v11 = *v10;
      v12 = *(_BYTE *)(*v10 + 8LL);
      if ( v12 > 3u && v12 != 5 && (v12 & 0xFD) != 4 )
        return 0;
      v13 = v10[1];
      if ( v7 == 2 )
      {
        if ( *(_BYTE *)(v13 + 8) != 16 || *(_QWORD *)(v13 + 32) != 2 )
          return 0;
        return **(_QWORD **)(v13 + 16) == v11;
      }
      else
      {
        if ( v7 != 3 || v13 != v11 )
          return 0;
        return v10[2] == v11;
      }
    }
    goto LABEL_18;
  }
  if ( a3 <= 0x80 )
  {
    if ( a3 - 62 <= 3 )
    {
      v8 = sub_97FA80(a1, (__int64)a4);
      switch ( (_DWORD)v6 )
      {
        case '@':
          if ( *(_DWORD *)(a2 + 12) != 3
            || !(unsigned __int8)sub_BCAC40(*(_QWORD *)(*(_QWORD *)(a2 + 16) + 8LL), v8)
            || !(unsigned __int8)sub_BCAC40(*(_QWORD *)(*(_QWORD *)(a2 + 16) + 16LL), v8) )
          {
            return 0;
          }
          break;
        case 'A':
          if ( *(_DWORD *)(a2 + 12) != 4
            || !(unsigned __int8)sub_BCAC40(*(_QWORD *)(*(_QWORD *)(a2 + 16) + 8LL), v8)
            || !(unsigned __int8)sub_BCAC40(*(_QWORD *)(*(_QWORD *)(a2 + 16) + 16LL), v8)
            || !(unsigned __int8)sub_BCAC40(*(_QWORD *)(*(_QWORD *)(a2 + 16) + 24LL), 8) )
          {
            return 0;
          }
          break;
        case '?':
          if ( *(_DWORD *)(a2 + 12) != 3
            || !(unsigned __int8)sub_BCAC40(*(_QWORD *)(*(_QWORD *)(a2 + 16) + 8LL), v8)
            || !(unsigned __int8)sub_BCAC40(*(_QWORD *)(*(_QWORD *)(a2 + 16) + 16LL), 8) )
          {
            return 0;
          }
          break;
        default:
          if ( *(_DWORD *)(a2 + 12) != 2 || !(unsigned __int8)sub_BCAC40(*(_QWORD *)(*(_QWORD *)(a2 + 16) + 8LL), v8) )
            return 0;
          break;
      }
      v32 = *a4;
      v49 = sub_BCE3C0(v32, 0);
      v50 = sub_BCD140(v32, v8);
      return **(_QWORD **)(a2 + 16) == sub_BD0B90(v32, &v49, 2, 0);
    }
LABEL_18:
    v41 = *(_DWORD *)(a1 + 172);
    v14 = sub_97FA80(a1, (__int64)a4);
    v15 = v7 - 1;
    v40 = v14;
    v16 = **(_QWORD **)(a2 + 16);
    v17 = (char *)&unk_3F1B860 + 8 * v6;
    v18 = 0;
    v19 = *v17;
    v20 = v16;
    while ( 1 )
    {
      if ( v19 == 18 )
        return *(_DWORD *)(a2 + 8) >> 8 != 0;
      if ( v19 != 19 )
      {
        if ( v16 )
        {
          switch ( v19 )
          {
            case 0:
              v28 = *(_BYTE *)(v16 + 8) == 7;
              goto LABEL_36;
            case 1:
              v39 = v15;
              v48 = v17;
              v28 = sub_BCAC40(v16, 8);
              v17 = v48;
              v15 = v39;
              goto LABEL_36;
            case 2:
              v37 = v15;
              v46 = v17;
              v28 = sub_BCAC40(v16, 16);
              v17 = v46;
              v15 = v37;
              goto LABEL_36;
            case 3:
              v36 = v15;
              v45 = v17;
              v28 = sub_BCAC40(v16, 32);
              v17 = v45;
              v15 = v36;
              goto LABEL_36;
            case 4:
              v38 = v15;
              v47 = v17;
              v28 = sub_BCAC40(v16, v41);
              v17 = v47;
              v15 = v38;
              goto LABEL_36;
            case 5:
            case 6:
              if ( *(_BYTE *)(v16 + 8) != 12 )
                return 0;
              v34 = v15;
              v43 = v17;
              v49 = sub_BCAE30(v16);
              v50 = v29;
              v30 = sub_CA1930(&v49);
              v17 = v43;
              v15 = v34;
              if ( v30 < v41 )
                return 0;
              goto LABEL_24;
            case 7:
              v28 = *(_BYTE *)(v16 + 8) == 12;
              goto LABEL_36;
            case 8:
            case 9:
              v33 = v15;
              v42 = v17;
              v28 = sub_BCAC40(v16, 64);
              v17 = v42;
              v15 = v33;
              goto LABEL_36;
            case 10:
            case 11:
              v35 = v15;
              v44 = v17;
              v28 = sub_BCAC40(v16, v40);
              v17 = v44;
              v15 = v35;
              goto LABEL_36;
            case 12:
              if ( *(_BYTE *)(v16 + 8) != 2 )
                return 0;
              goto LABEL_24;
            case 13:
              v28 = *(_BYTE *)(v16 + 8) == 3;
              goto LABEL_36;
            case 14:
            case 15:
              v31 = *(_BYTE *)(v16 + 8);
              if ( v31 > 3u && v31 != 5 && (v31 & 0xFD) != 4 )
                return 0;
              goto LABEL_24;
            case 16:
              v28 = *(_BYTE *)(v16 + 8) == 14;
              goto LABEL_36;
            case 17:
              v28 = *(_BYTE *)(v16 + 8) == 15;
LABEL_36:
              if ( !v28 )
                return 0;
              goto LABEL_24;
            default:
              BUG();
          }
        }
        return 0;
      }
      if ( v16 != v20 )
        return 0;
LABEL_24:
      v21 = v18 + 1;
      v22 = (_DWORD)v18 == v15 ? 0LL : *(_QWORD *)(*(_QWORD *)(a2 + 16) + 8 * v18 + 8);
      if ( v18 == 7 )
        break;
      v19 = v17[++v18];
      if ( !v19 )
        break;
      v20 = v16;
      v16 = v22;
    }
    if ( v7 != v21 )
      return 0;
    return *(_DWORD *)(a2 + 8) >> 8 == 0;
  }
  if ( v7 != 2 )
    return 0;
  v23 = *(_QWORD **)(a2 + 16);
  v24 = *v23;
  v25 = v23[1];
  v26 = *(_BYTE *)(*v23 + 8LL);
  if ( v26 == 15 )
  {
    if ( *(_DWORD *)(v24 + 12) != 2 )
      return 0;
    v27 = *(_QWORD **)(v24 + 16);
    if ( v25 != *v27 )
      return 0;
    return v27[1] == v25;
  }
  else
  {
    if ( v26 != 17 || *(_DWORD *)(v24 + 32) != 2 )
      return 0;
    return *(_QWORD *)(v24 + 24) == v25;
  }
}
