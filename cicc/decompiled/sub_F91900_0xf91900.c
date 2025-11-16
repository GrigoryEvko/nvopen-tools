// Function: sub_F91900
// Address: 0xf91900
//
__int64 __fastcall sub_F91900(unsigned __int8 *a1, unsigned __int8 *a2, bool a3)
{
  unsigned __int8 *v3; // r12
  unsigned __int8 *v4; // r13
  unsigned __int64 v5; // rbx
  unsigned __int8 *v6; // r14
  unsigned int v7; // r15d
  unsigned int v9; // eax
  unsigned __int8 *v10; // rdx
  char *v11; // rdi
  int v12; // eax
  unsigned __int8 *v13; // rdx
  unsigned __int64 v14; // rdx
  __int64 v15; // rsi
  unsigned __int8 *v16; // rcx
  __int64 v17; // rax
  unsigned __int8 *v18; // rax
  int v19; // esi
  unsigned int v20; // eax
  __int64 v21; // rax
  char v22; // dl
  __int64 v23; // rax
  unsigned int v24; // r12d
  __int64 v25; // rax
  int v26; // eax
  __int64 v27; // rax
  int v28; // r13d
  __int64 v29; // rax
  int v30; // r10d
  unsigned __int8 *v31; // rax
  bool v32; // al
  __int64 v33; // rax
  __int64 v34; // rax
  __int64 v35; // rax
  __int64 v36; // rdx
  __int64 v37; // rax
  __int64 v38; // rdx
  __int64 v39; // rax
  __int64 v40; // rbx
  __int64 v41; // rax
  unsigned int v42; // r12d
  __int64 v43; // rax
  unsigned __int8 *v44; // [rsp+0h] [rbp-50h]
  unsigned __int8 *v45; // [rsp+8h] [rbp-48h]
  bool v46; // [rsp+8h] [rbp-48h]
  int v47; // [rsp+8h] [rbp-48h]
  bool v48; // [rsp+14h] [rbp-3Ch]
  bool v49; // [rsp+18h] [rbp-38h]
  char v50; // [rsp+18h] [rbp-38h]
  __int64 v51; // [rsp+18h] [rbp-38h]

  while ( 1 )
  {
    v48 = a3;
    v49 = a3;
    if ( *a1 > 0x15u )
      return 0;
    v3 = a2;
    if ( !*((_QWORD *)a2 + 2) )
      return 0;
    v4 = a1;
    if ( !sub_AC30F0((__int64)a1) && (unsigned int)*a1 - 12 > 1 )
      return 0;
    v5 = *((_QWORD *)a2 + 2);
    if ( !v5 )
      return 0;
    while ( 2 )
    {
      v6 = *(unsigned __int8 **)(v5 + 24);
      switch ( *v6 )
      {
        case 0x1Eu:
        case 0x22u:
        case 0x28u:
        case 0x30u:
        case 0x31u:
        case 0x33u:
        case 0x34u:
        case 0x3Du:
        case 0x3Eu:
        case 0x3Fu:
        case 0x4Eu:
        case 0x55u:
          if ( *((_QWORD *)v6 + 5) != *((_QWORD *)a2 + 5) )
            return 0;
          if ( a2 == v6 )
            return 0;
          LOBYTE(v9) = sub_B445A0(*(_QWORD *)(v5 + 24), (__int64)a2);
          v7 = v9;
          if ( (_BYTE)v9 )
            return 0;
          v10 = (unsigned __int8 *)*((_QWORD *)a2 + 4);
          v44 = v6 + 24;
          if ( v6 + 24 == v10 )
            goto LABEL_20;
          break;
        default:
          v5 = *(_QWORD *)(v5 + 8);
          if ( !v5 )
            return 0;
          continue;
      }
      break;
    }
    while ( 1 )
    {
      v11 = (char *)(v10 - 24);
      v45 = v10;
      if ( !v10 )
        v11 = 0;
      if ( !(unsigned __int8)sub_98CD80(v11) )
        break;
      v10 = (unsigned __int8 *)*((_QWORD *)v45 + 1);
      if ( v44 == v10 )
        goto LABEL_20;
    }
    if ( v44 != v45 )
      return 0;
LABEL_20:
    v12 = *v6;
    if ( (_BYTE)v12 != 63 )
      break;
    v13 = *(unsigned __int8 **)&v6[-32 * (*((_DWORD *)v6 + 1) & 0x7FFFFFF)];
    v46 = v13 != 0 && a2 == v13;
    if ( !v46 )
      goto LABEL_22;
    if ( !(unsigned __int8)sub_B4DCF0((__int64)v6) )
    {
      if ( !sub_B4DE30((__int64)v6) )
        goto LABEL_88;
      v41 = *(_QWORD *)(*(_QWORD *)&v6[-32 * (*((_DWORD *)v6 + 1) & 0x7FFFFFF)] + 8LL);
      if ( (unsigned int)*(unsigned __int8 *)(v41 + 8) - 17 <= 1 )
        v41 = **(_QWORD **)(v41 + 16);
      v42 = *(_DWORD *)(v41 + 8);
      v43 = sub_B43CB0((__int64)v6);
      if ( sub_B2F070(v43, v42 >> 8) )
LABEL_88:
        v49 = v46;
    }
    a3 = v49;
    a2 = v6;
    a1 = v4;
  }
  if ( (_BYTE)v12 == 30 )
  {
    v21 = sub_B43CB0((__int64)v6);
    v22 = sub_B2D630(v21, 40);
    if ( (unsigned int)*v4 - 12 > 1 )
    {
      v50 = v22;
      v32 = sub_AC30F0((__int64)v4);
      if ( v50 )
      {
        if ( v32 )
        {
          v33 = sub_B43CB0((__int64)v6);
          if ( (unsigned __int8)sub_B2D630(v33, 43) )
            return !v48;
        }
      }
    }
    else
    {
      if ( v22 )
        return 1;
      sub_AC30F0((__int64)v4);
    }
    v12 = *v6;
  }
  if ( (_BYTE)v12 != 61 )
  {
    if ( (_BYTE)v12 != 62 )
    {
      if ( (_BYTE)v12 == 85 )
      {
        v17 = *((_QWORD *)v6 - 4);
        if ( v17 )
        {
          if ( !*(_BYTE *)v17
            && *(_QWORD *)(v17 + 24) == *((_QWORD *)v6 + 10)
            && (*(_BYTE *)(v17 + 33) & 0x20) != 0
            && *(_DWORD *)(v17 + 36) == 11 )
          {
            v18 = *(unsigned __int8 **)&v6[-32 * (*((_DWORD *)v6 + 1) & 0x7FFFFFF)];
            if ( a2 == v18 )
            {
              if ( v18 )
                return 1;
            }
          }
        }
        goto LABEL_37;
      }
      v14 = (unsigned int)(v12 - 34);
      if ( (unsigned __int8)(v12 - 34) > 0x33u )
        goto LABEL_24;
LABEL_23:
      v15 = 0x8000000000041LL;
      if ( !_bittest64(&v15, v14) )
      {
LABEL_24:
        if ( (unsigned int)(unsigned __int8)v12 - 42 > 0x11 )
          return v7;
        v16 = (unsigned __int8 *)*((_QWORD *)v6 - 4);
        if ( v3 != v16 || !v16 || (unsigned __int8)(v12 - 51) > 1u && (unsigned int)(unsigned __int8)v12 - 48 > 1 )
          return v7;
        return 1;
      }
LABEL_37:
      if ( sub_AC30F0((__int64)v4) )
      {
        v34 = sub_B43CB0((__int64)v6);
        if ( sub_B2F070(v34, 0) )
          return 0;
      }
      if ( v3 == *((unsigned __int8 **)v6 - 4) )
        return 1;
      v19 = *v6;
      if ( v5 < (unsigned __int64)&v6[-32 * (*((_DWORD *)v6 + 1) & 0x7FFFFFF)] )
        return v7;
      if ( v19 == 40 )
      {
        v20 = sub_B491D0((__int64)v6);
      }
      else
      {
        v20 = 0;
        if ( v19 != 85 )
        {
          v20 = 2;
          if ( v19 != 34 )
LABEL_99:
            BUG();
        }
      }
      v51 = -32 - 32LL * v20;
      if ( (v6[7] & 0x80u) != 0 )
      {
        v35 = sub_BD2BC0((__int64)v6);
        if ( (v6[7] & 0x80u) == 0 )
        {
          if ( (unsigned int)((v35 + v36) >> 4) )
            goto LABEL_99;
        }
        else if ( (unsigned int)((v35 + v36 - sub_BD2BC0((__int64)v6)) >> 4) )
        {
          if ( (v6[7] & 0x80u) == 0 )
            goto LABEL_99;
          v47 = *(_DWORD *)(sub_BD2BC0((__int64)v6) + 8);
          if ( (v6[7] & 0x80u) == 0 )
            BUG();
          v37 = sub_BD2BC0((__int64)v6);
          v39 = (unsigned int)(*(_DWORD *)(v37 + v38 - 4) - v47);
LABEL_78:
          if ( v5 < (unsigned __int64)&v6[v51 - 32 * v39] )
          {
            v40 = (__int64)(v5 - (_QWORD)&v6[-32 * (*((_DWORD *)v6 + 1) & 0x7FFFFFF)]) >> 5;
            if ( sub_AC30F0((__int64)v4)
              && ((unsigned __int8)sub_B49B80((__int64)v6, v40, 40)
               || (unsigned __int8)sub_B49B80((__int64)v6, v40, 90)
               || (unsigned __int8)sub_B49B80((__int64)v6, v40, 91))
              && (unsigned __int8)sub_B49B80((__int64)v6, v40, 43) )
            {
              return !v48;
            }
            if ( (unsigned int)*v4 - 12 <= 1
              && ((unsigned __int8)sub_B49B80((__int64)v6, v40, 40)
               || (unsigned __int8)sub_B49B80((__int64)v6, v40, 90)
               || (unsigned __int8)sub_B49B80((__int64)v6, v40, 91)) )
            {
              return 1;
            }
          }
          LOBYTE(v12) = *v6;
          goto LABEL_24;
        }
      }
      v39 = 0;
      goto LABEL_78;
    }
    if ( (v6[2] & 1) == 0 )
    {
      v27 = *(_QWORD *)(*((_QWORD *)v6 - 4) + 8LL);
      if ( (unsigned int)*(unsigned __int8 *)(v27 + 8) - 17 <= 1 )
        v27 = **(_QWORD **)(v27 + 16);
      v28 = *(_DWORD *)(v27 + 8) >> 8;
      v29 = sub_B43CB0((__int64)v6);
      if ( !sub_B2F070(v29, v28) )
      {
        v31 = (unsigned __int8 *)*((_QWORD *)v6 - 4);
        LOBYTE(v30) = a2 == v31;
        LOBYTE(v31) = v31 != 0;
        return (unsigned int)v31 & v30;
      }
      return 0;
    }
LABEL_22:
    v14 = (unsigned int)(v12 - 34);
    goto LABEL_23;
  }
  if ( (v6[2] & 1) != 0 )
    goto LABEL_22;
  v23 = *(_QWORD *)(*((_QWORD *)v6 - 4) + 8LL);
  if ( (unsigned int)*(unsigned __int8 *)(v23 + 8) - 17 <= 1 )
    v23 = **(_QWORD **)(v23 + 16);
  v24 = *(_DWORD *)(v23 + 8);
  v25 = sub_B43CB0((__int64)v6);
  LOBYTE(v26) = sub_B2F070(v25, v24 >> 8);
  return v26 ^ 1u;
}
