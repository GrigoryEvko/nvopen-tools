// Function: sub_10BA570
// Address: 0x10ba570
//
_BYTE *__fastcall sub_10BA570(char *a1)
{
  _BYTE *v1; // rbx
  unsigned __int8 v2; // r12
  int v3; // ecx
  char v4; // al
  int v5; // esi
  _BYTE **v6; // rdx
  _BYTE *v7; // rdx
  _BYTE *result; // rax
  __int64 v9; // rax
  __int64 v10; // rdx
  __int64 v11; // r13
  int v12; // ecx
  unsigned int v13; // edi
  unsigned __int8 v14; // dl
  char v15; // r13
  unsigned __int8 **v16; // rdx
  unsigned __int8 *v17; // r13
  unsigned __int8 v18; // al
  _BYTE *v19; // r13
  bool v20; // al
  _BYTE *v21; // rdx
  unsigned __int8 **v22; // rdx
  unsigned __int8 *v23; // r14
  unsigned __int8 v24; // al
  unsigned __int8 v25; // al
  _BYTE *v26; // r14
  __int64 v27; // r13
  __int64 v28; // rdx
  void **v29; // rax
  void **v30; // r12
  void **v31; // rax
  __int64 v32; // r14
  __int64 v33; // rdx
  void **v34; // rax
  void **v35; // r12
  void **v36; // r12
  int v37; // r14d
  unsigned int v38; // r15d
  void **v39; // rax
  void **v40; // r12
  char v41; // al
  _BYTE *v42; // r12
  int v43; // r13d
  unsigned int i; // r15d
  _BYTE *v45; // rax
  _BYTE *v46; // r12
  char v47; // al
  _BYTE *v48; // r12
  char v49; // [rsp+Fh] [rbp-31h]

  v1 = a1;
  v2 = *a1;
  if ( (unsigned __int8)*a1 <= 0x1Cu )
    return v1;
  v3 = v2;
  v4 = *a1;
  v5 = v2 - 29;
  switch ( v2 )
  {
    case ')':
    case '+':
    case '-':
    case '/':
    case '2':
    case '5':
    case 'J':
    case 'K':
    case 'S':
      goto LABEL_10;
    case 'T':
    case 'U':
    case 'V':
      v11 = *((_QWORD *)a1 + 1);
      v12 = *(unsigned __int8 *)(v11 + 8);
      v13 = v12 - 17;
      v14 = *(_BYTE *)(v11 + 8);
      if ( (unsigned int)(v12 - 17) <= 1 )
        v14 = *(_BYTE *)(**(_QWORD **)(v11 + 16) + 8LL);
      if ( v14 <= 3u || v14 == 5 || (v14 & 0xFD) == 4 )
        goto LABEL_11;
      if ( (_BYTE)v12 == 15 )
      {
        if ( (*(_BYTE *)(v11 + 9) & 4) == 0 )
          goto LABEL_7;
        if ( !sub_BCB420(*((_QWORD *)v1 + 1)) )
          goto LABEL_53;
        v2 = *v1;
        v11 = **(_QWORD **)(v11 + 16);
        v4 = *v1;
        v12 = *(unsigned __int8 *)(v11 + 8);
        v13 = v12 - 17;
      }
      else if ( (_BYTE)v12 == 16 )
      {
        do
        {
          v11 = *(_QWORD *)(v11 + 24);
          LOBYTE(v12) = *(_BYTE *)(v11 + 8);
        }
        while ( (_BYTE)v12 == 16 );
        v13 = (unsigned __int8)v12 - 17;
      }
      if ( v13 <= 1 )
        LOBYTE(v12) = *(_BYTE *)(**(_QWORD **)(v11 + 16) + 8LL);
      if ( (unsigned __int8)v12 > 3u && (_BYTE)v12 != 5 && (v12 & 0xFD) != 4 )
      {
LABEL_7:
        if ( v4 != 85 )
          return v1;
        goto LABEL_14;
      }
      v3 = v2;
      if ( v2 > 0x1Cu )
LABEL_10:
        v5 = v3 - 29;
      else
        v5 = *((unsigned __int16 *)v1 + 1);
LABEL_11:
      if ( v5 == 12 )
      {
        if ( (v1[7] & 0x40) != 0 )
          v6 = (_BYTE **)*((_QWORD *)v1 - 1);
        else
          v6 = (_BYTE **)&v1[-32 * (*((_DWORD *)v1 + 1) & 0x7FFFFFF)];
        v7 = *v6;
        if ( v7 )
          goto LABEL_6;
      }
      else
      {
        if ( v5 != 16 )
          goto LABEL_13;
        v15 = v1[7] & 0x40;
        if ( (v1[1] & 0x10) != 0 )
        {
          if ( v15 )
            v16 = (unsigned __int8 **)*((_QWORD *)v1 - 1);
          else
            v16 = (unsigned __int8 **)&v1[-32 * (*((_DWORD *)v1 + 1) & 0x7FFFFFF)];
          v17 = *v16;
          v18 = **v16;
          if ( v18 == 18 )
          {
            if ( *((void **)v17 + 3) == sub_C33340() )
              v19 = (_BYTE *)*((_QWORD *)v17 + 4);
            else
              v19 = v17 + 24;
            v20 = (v19[20] & 7) == 3;
            goto LABEL_48;
          }
          v32 = *((_QWORD *)v17 + 1);
          v33 = (unsigned int)*(unsigned __int8 *)(v32 + 8) - 17;
          if ( (unsigned int)v33 <= 1 && v18 <= 0x15u )
          {
            v34 = (void **)sub_AD7630((__int64)v17, 0, v33);
            v35 = v34;
            if ( v34 && *(_BYTE *)v34 == 18 )
            {
              if ( v34[3] == sub_C33340() )
                v36 = (void **)v35[4];
              else
                v36 = v35 + 3;
              v20 = (*((_BYTE *)v36 + 20) & 7) == 3;
LABEL_48:
              if ( v20 )
              {
LABEL_49:
                v15 = v1[7] & 0x40;
                goto LABEL_50;
              }
              goto LABEL_75;
            }
            if ( *(_BYTE *)(v32 + 8) == 17 )
            {
              v37 = *(_DWORD *)(v32 + 32);
              if ( v37 )
              {
                v49 = 0;
                v38 = 0;
                while ( 1 )
                {
                  v39 = (void **)sub_AD69F0(v17, v38);
                  v40 = v39;
                  if ( !v39 )
                    break;
                  v41 = *(_BYTE *)v39;
                  if ( v41 != 13 )
                  {
                    if ( v41 != 18 )
                      break;
                    v42 = v40[3] == sub_C33340() ? v40[4] : v40 + 3;
                    if ( (v42[20] & 7) != 3 )
                      break;
                    v49 = 1;
                  }
                  if ( v37 == ++v38 )
                    goto LABEL_102;
                }
              }
            }
LABEL_75:
            v2 = *v1;
          }
        }
        else
        {
          if ( v15 )
            v22 = (unsigned __int8 **)*((_QWORD *)v1 - 1);
          else
            v22 = (unsigned __int8 **)&v1[-32 * (*((_DWORD *)v1 + 1) & 0x7FFFFFF)];
          v23 = *v22;
          v24 = **v22;
          if ( v24 == 18 )
          {
            if ( *((void **)v23 + 3) == sub_C33340() )
            {
              v26 = (_BYTE *)*((_QWORD *)v23 + 4);
              if ( (v26[20] & 7) != 3 )
                goto LABEL_13;
            }
            else
            {
              v25 = v23[44];
              v26 = v23 + 24;
              if ( (v25 & 7) != 3 )
                goto LABEL_13;
            }
            if ( (v26[20] & 8) != 0 )
            {
LABEL_50:
              if ( v15 )
                v21 = (_BYTE *)*((_QWORD *)v1 - 1);
              else
                v21 = &v1[-32 * (*((_DWORD *)v1 + 1) & 0x7FFFFFF)];
              v7 = (_BYTE *)*((_QWORD *)v21 + 4);
              if ( !v7 )
              {
LABEL_53:
                v4 = *v1;
                goto LABEL_7;
              }
LABEL_6:
              v4 = *v7;
              v1 = v7;
              goto LABEL_7;
            }
          }
          else
          {
            v27 = *((_QWORD *)v23 + 1);
            v28 = (unsigned int)*(unsigned __int8 *)(v27 + 8) - 17;
            if ( (unsigned int)v28 <= 1 && v24 <= 0x15u )
            {
              v29 = (void **)sub_AD7630((__int64)v23, 0, v28);
              v30 = v29;
              if ( !v29 || *(_BYTE *)v29 != 18 )
              {
                if ( *(_BYTE *)(v27 + 8) == 17 )
                {
                  v43 = *(_DWORD *)(v27 + 32);
                  if ( v43 )
                  {
                    v49 = 0;
                    for ( i = 0; i != v43; ++i )
                    {
                      v45 = (_BYTE *)sub_AD69F0(v23, i);
                      v46 = v45;
                      if ( !v45 )
                        goto LABEL_75;
                      v47 = *v45;
                      if ( v47 != 13 )
                      {
                        if ( v47 != 18 )
                          goto LABEL_75;
                        if ( *((void **)v46 + 3) == sub_C33340() )
                        {
                          v48 = (_BYTE *)*((_QWORD *)v46 + 4);
                          if ( (v48[20] & 7) != 3 )
                            goto LABEL_75;
                        }
                        else
                        {
                          if ( (v46[44] & 7) != 3 )
                            goto LABEL_75;
                          v48 = v46 + 24;
                        }
                        if ( (v48[20] & 8) == 0 )
                          goto LABEL_75;
                        v49 = 1;
                      }
                    }
LABEL_102:
                    if ( v49 )
                      goto LABEL_49;
                  }
                }
                goto LABEL_75;
              }
              if ( v29[3] == sub_C33340() )
              {
                v31 = (void **)v30[4];
                if ( (*((_BYTE *)v31 + 20) & 7) != 3 )
                  goto LABEL_75;
              }
              else
              {
                if ( (*((_BYTE *)v30 + 44) & 7) != 3 )
                  goto LABEL_75;
                v31 = v30 + 3;
              }
              if ( (*((_BYTE *)v31 + 20) & 8) != 0 )
                goto LABEL_49;
              goto LABEL_75;
            }
          }
        }
      }
LABEL_13:
      if ( v2 != 85 )
        return v1;
LABEL_14:
      v9 = *((_QWORD *)v1 - 4);
      if ( v9
        && !*(_BYTE *)v9
        && *(_QWORD *)(v9 + 24) == *((_QWORD *)v1 + 10)
        && *(_DWORD *)(v9 + 36) == 170
        && (result = *(_BYTE **)&v1[-32 * (*((_DWORD *)v1 + 1) & 0x7FFFFFF)]) != 0 )
      {
        if ( *result != 85 )
          return result;
      }
      else
      {
        result = v1;
      }
      v10 = *((_QWORD *)result - 4);
      if ( v10 && !*(_BYTE *)v10 && *(_QWORD *)(v10 + 24) == *((_QWORD *)result + 10) && *(_DWORD *)(v10 + 36) == 26 )
      {
        if ( *(_QWORD *)&result[-32 * (*((_DWORD *)result + 1) & 0x7FFFFFF)] )
          return *(_BYTE **)&result[-32 * (*((_DWORD *)result + 1) & 0x7FFFFFF)];
      }
      return result;
    default:
      goto LABEL_7;
  }
}
