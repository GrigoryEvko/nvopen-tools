// Function: sub_1110120
// Address: 0x1110120
//
__int64 __fastcall sub_1110120(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  unsigned __int16 v5; // r12
  _BYTE *v6; // r13
  _BYTE *v9; // r15
  __int16 v10; // ax
  _BYTE *v11; // rax
  __int64 result; // rax
  _BYTE *v13; // r15
  char v14; // al
  char v15; // al
  _BYTE *v16; // r13
  __int64 v17; // rdi
  unsigned __int8 v18; // al
  __int16 v19; // r12
  __int64 v20; // rcx
  __int64 v21; // r8
  __int64 v22; // r9
  __int64 v23; // rdx
  __int64 v24; // rbx
  __int64 v25; // rsi
  __int64 v26; // rsi
  __int64 v27; // r12
  __int64 v28; // r12
  __int64 v29; // rcx
  __int64 v30; // r8
  __int64 v31; // r9
  __int64 v32; // rdx
  char v33; // al
  _BYTE *v34; // r13
  __int16 v35; // r15
  bool v36; // al
  bool v37; // al
  unsigned int v38; // edx
  unsigned int v39; // eax
  __int64 v40; // rsi
  unsigned int v41; // ecx
  unsigned int v42; // eax
  __int64 v43; // rsi
  unsigned int v44; // r15d
  unsigned int v45; // eax
  __int64 v46; // rsi
  unsigned int v47; // ecx
  unsigned int v48; // eax
  __int64 v49; // rsi
  unsigned int v50; // r15d
  unsigned int v51; // edx
  unsigned int v52; // edx
  __int64 v53; // [rsp+8h] [rbp-48h]
  int v54; // [rsp+8h] [rbp-48h]
  int v55; // [rsp+8h] [rbp-48h]
  int v56; // [rsp+8h] [rbp-48h]
  int v57; // [rsp+8h] [rbp-48h]
  int v58; // [rsp+8h] [rbp-48h]
  int v59; // [rsp+8h] [rbp-48h]
  __int64 v60[7]; // [rsp+18h] [rbp-38h] BYREF

  v5 = *(_WORD *)(a2 + 2);
  if ( (v5 & 1) != 0 )
    return 0;
  v6 = *(_BYTE **)(a2 - 32);
  v9 = v6;
  if ( *v6 != 18 )
  {
    if ( *v6 == 17 )
    {
      switch ( (v5 >> 4) & 0x1F )
      {
        case 0:
          goto LABEL_50;
        case 3:
        case 0xA:
          if ( *((_DWORD *)v6 + 8) <= 0x40u )
          {
            v37 = *((_QWORD *)v6 + 3) == 0;
          }
          else
          {
            v55 = *((_DWORD *)v6 + 8);
            v37 = v55 == (unsigned int)sub_C444A0((__int64)(v6 + 24));
          }
          goto LABEL_74;
        case 5:
          if ( sub_AD7930(*(_BYTE **)(a2 - 32), a2, a3, a4, a5) )
          {
            v5 = *(_WORD *)(a2 + 2);
            if ( (v5 & 0x1F0) != 0 )
              goto LABEL_9;
            v9 = *(_BYTE **)(a2 - 32);
LABEL_50:
            v14 = *v9;
            v6 = v9;
            if ( *v9 == 18 )
              return 0;
          }
          else
          {
            v6 = *(_BYTE **)(a2 - 32);
            v14 = *v6;
            if ( *v6 == 18 )
            {
              v5 = *(_WORD *)(a2 + 2);
              goto LABEL_21;
            }
          }
          if ( v14 != 17 )
            return 0;
          v5 = *(_WORD *)(a2 + 2);
          goto LABEL_53;
        case 7:
          v45 = *((_DWORD *)v6 + 8);
          v46 = *((_QWORD *)v6 + 3);
          v47 = v45 - 1;
          if ( v45 <= 0x40 )
          {
            v37 = v46 == (1LL << v47) - 1;
            goto LABEL_74;
          }
          if ( (*(_QWORD *)(v46 + 8LL * (v47 >> 6)) & (1LL << v47)) == 0
            && v47 == (unsigned int)sub_C445E0((__int64)(v6 + 24)) )
          {
            goto LABEL_9;
          }
LABEL_89:
          v48 = *((_DWORD *)v6 + 8);
          v49 = *((_QWORD *)v6 + 3);
          v50 = v48 - 1;
          if ( v48 <= 0x40 )
          {
            v35 = 7;
            v36 = 1LL << ((unsigned __int8)v48 - 1) == v49;
            goto LABEL_59;
          }
          if ( (*(_QWORD *)(v49 + 8LL * (v50 >> 6)) & (1LL << v50)) != 0
            && v50 == (unsigned int)sub_C44590((__int64)(v6 + 24)) )
          {
            goto LABEL_25;
          }
          return 0;
        case 8:
          v39 = *((_DWORD *)v6 + 8);
          v40 = *((_QWORD *)v6 + 3);
          v41 = v39 - 1;
          if ( v39 <= 0x40 )
          {
            v37 = v40 == 1LL << v41;
LABEL_74:
            if ( !v37 )
            {
LABEL_53:
              v35 = (v5 >> 4) & 0x1F;
              switch ( v35 )
              {
                case 1:
                case 2:
                case 5:
                case 6:
                  if ( *((_DWORD *)v6 + 8) <= 0x40u )
                    goto LABEL_108;
                  v54 = *((_DWORD *)v6 + 8);
                  v36 = v54 == (unsigned int)sub_C444A0((__int64)(v6 + 24));
                  goto LABEL_59;
                case 3:
                  v51 = *((_DWORD *)v6 + 8);
                  if ( !v51 )
                    goto LABEL_25;
                  if ( v51 <= 0x40 )
                  {
                    v36 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v51) == *((_QWORD *)v6 + 3);
                  }
                  else
                  {
                    v58 = *((_DWORD *)v6 + 8);
                    v36 = v58 == (unsigned int)sub_C445E0((__int64)(v6 + 24));
                  }
                  goto LABEL_59;
                case 7:
                  goto LABEL_89;
                case 8:
                  goto LABEL_82;
                case 9:
                  if ( *((_DWORD *)v6 + 8) <= 0x40u )
                  {
LABEL_108:
                    v36 = *((_QWORD *)v6 + 3) == 0;
                    goto LABEL_59;
                  }
                  v57 = *((_DWORD *)v6 + 8);
                  if ( v57 == (unsigned int)sub_C444A0((__int64)(v6 + 24)) )
                    goto LABEL_60;
                  return 0;
                case 10:
                  v52 = *((_DWORD *)v6 + 8);
                  if ( !v52 )
                    goto LABEL_25;
                  if ( v52 <= 0x40 )
                  {
                    if ( *((_QWORD *)v6 + 3) == 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v52) )
                      goto LABEL_60;
                  }
                  else
                  {
                    v59 = *((_DWORD *)v6 + 8);
                    if ( v59 == (unsigned int)sub_C445E0((__int64)(v6 + 24)) )
                      goto LABEL_25;
                  }
                  break;
                default:
                  return 0;
              }
              return 0;
            }
            goto LABEL_9;
          }
          if ( (*(_QWORD *)(v40 + 8LL * (v41 >> 6)) & (1LL << v41)) != 0
            && (unsigned int)sub_C44590((__int64)(v6 + 24)) == v41 )
          {
            goto LABEL_9;
          }
LABEL_82:
          v42 = *((_DWORD *)v6 + 8);
          v43 = *((_QWORD *)v6 + 3);
          v44 = v42 - 1;
          if ( v42 > 0x40 )
          {
            if ( (*(_QWORD *)(v43 + 8LL * (v44 >> 6)) & (1LL << v44)) != 0
              || v44 != (unsigned int)sub_C445E0((__int64)(v6 + 24)) )
            {
              return 0;
            }
            goto LABEL_25;
          }
          v35 = 8;
          v36 = (1LL << ((unsigned __int8)v42 - 1)) - 1 == v43;
LABEL_59:
          if ( !v36 )
            return 0;
LABEL_60:
          v17 = *(_QWORD *)(a2 + 8);
          v18 = *(_BYTE *)(v17 + 8);
          if ( v18 != 12 )
            goto LABEL_26;
          if ( v35 != 5 )
            goto LABEL_62;
          return 0;
        case 9:
          v38 = *((_DWORD *)v6 + 8);
          if ( !v38 )
            goto LABEL_9;
          if ( v38 <= 0x40 )
          {
            v37 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v38) == *((_QWORD *)v6 + 3);
          }
          else
          {
            v56 = *((_DWORD *)v6 + 8);
            v37 = v56 == (unsigned int)sub_C445E0((__int64)(v6 + 24));
          }
          goto LABEL_74;
        default:
          goto LABEL_53;
      }
    }
    return 0;
  }
  v10 = (v5 >> 4) & 0x1F;
  if ( v10 == 13 )
  {
    if ( *((void **)v6 + 3) != sub_C33340() )
    {
      if ( (v6[44] & 8) != 0 )
        return 0;
LABEL_16:
      v13 = v6 + 24;
      goto LABEL_17;
    }
    v13 = (_BYTE *)*((_QWORD *)v6 + 4);
    if ( (v13[20] & 8) != 0 )
      return 0;
LABEL_17:
    if ( (v13[20] & 7) == 0 )
    {
LABEL_9:
      *(_WORD *)(a2 + 2) = v5 & 0xFE0F;
      return a2;
    }
    return 0;
  }
  if ( ((v5 >> 4) & 0x1Fu) > 0xD )
  {
    if ( v10 != 14 )
      goto LABEL_21;
    if ( *((void **)v6 + 3) != sub_C33340() )
    {
      if ( (v6[44] & 8) == 0 )
        return 0;
      goto LABEL_16;
    }
    v13 = (_BYTE *)*((_QWORD *)v6 + 4);
    if ( (v13[20] & 8) == 0 )
      return 0;
    goto LABEL_17;
  }
  if ( (unsigned __int16)(v10 - 11) <= 1u )
  {
    v11 = *((void **)v6 + 3) == sub_C33340() ? (_BYTE *)*((_QWORD *)v6 + 4) : v6 + 24;
    if ( (v11[20] & 7) == 1 )
      goto LABEL_9;
  }
LABEL_21:
  if ( ((v5 >> 4) & 0x1F) == 0xB )
  {
    if ( *((void **)v6 + 3) == sub_C33340() )
    {
      v16 = (_BYTE *)*((_QWORD *)v6 + 4);
      if ( (v16[20] & 7) != 3 )
        return 0;
    }
    else
    {
      v15 = v6[44];
      v16 = v6 + 24;
      if ( (v15 & 7) != 3 )
        return 0;
    }
    if ( (v16[20] & 8) == 0 )
      return 0;
  }
  else
  {
    if ( ((v5 >> 4) & 0x1F) != 0xC )
      return 0;
    if ( *((void **)v6 + 3) == sub_C33340() )
    {
      v34 = (_BYTE *)*((_QWORD *)v6 + 4);
      if ( (v34[20] & 7) != 3 )
        return 0;
    }
    else
    {
      v33 = v6[44];
      v34 = v6 + 24;
      if ( (v33 & 7) != 3 )
        return 0;
    }
    if ( (v34[20] & 8) != 0 )
      return 0;
  }
LABEL_25:
  v17 = *(_QWORD *)(a2 + 8);
  v18 = *(_BYTE *)(v17 + 8);
  if ( v18 == 12 )
  {
LABEL_62:
    *(_WORD *)(a2 + 2) = v5 & 0xFE0F | 0x50;
    v20 = sub_AD64C0(v17, 0, 0);
    result = a2;
    if ( (*(_BYTE *)(a2 + 7) & 0x40) != 0 )
      goto LABEL_29;
    goto LABEL_63;
  }
LABEL_26:
  if ( v18 > 3u && v18 != 5 && (v18 & 0xFD) != 4 || ((v5 >> 4) & 0x1F) == 0xB )
    return 0;
  v19 = v5 & 0xFE0F;
  LOBYTE(v19) = v19 | 0xB0;
  *(_WORD *)(a2 + 2) = v19;
  v20 = (__int64)sub_AD9290(v17, 1);
  result = a2;
  if ( (*(_BYTE *)(a2 + 7) & 0x40) != 0 )
  {
LABEL_29:
    v23 = *(_QWORD *)(result - 8);
    v24 = *(_QWORD *)(v23 + 32);
    goto LABEL_30;
  }
LABEL_63:
  v23 = result - 32LL * (*(_DWORD *)(result + 4) & 0x7FFFFFF);
  v24 = *(_QWORD *)(v23 + 32);
LABEL_30:
  if ( v24 )
  {
    v25 = *(_QWORD *)(v23 + 40);
    **(_QWORD **)(v23 + 48) = v25;
    if ( v25 )
      *(_QWORD *)(v25 + 16) = *(_QWORD *)(v23 + 48);
  }
  *(_QWORD *)(v23 + 32) = v20;
  if ( v20 )
  {
    v26 = *(_QWORD *)(v20 + 16);
    *(_QWORD *)(v23 + 40) = v26;
    if ( v26 )
    {
      v21 = v23 + 40;
      *(_QWORD *)(v26 + 16) = v23 + 40;
    }
    *(_QWORD *)(v23 + 48) = v20 + 16;
    v23 += 32;
    *(_QWORD *)(v20 + 16) = v23;
  }
  if ( *(_BYTE *)v24 > 0x1Cu )
  {
    v27 = *(_QWORD *)(a1 + 40);
    v53 = result;
    v60[0] = v24;
    v28 = v27 + 2096;
    sub_110FAE0(v28, v60, v23, v20, v21, v22);
    v32 = *(_QWORD *)(v24 + 16);
    result = v53;
    if ( v32 )
    {
      if ( !*(_QWORD *)(v32 + 8) )
      {
        sub_110FAE0(v28, v60, *(_QWORD *)(v32 + 24), v29, v30, v31);
        return v53;
      }
    }
  }
  return result;
}
