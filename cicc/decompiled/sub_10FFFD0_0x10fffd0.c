// Function: sub_10FFFD0
// Address: 0x10fffd0
//
__int64 __fastcall sub_10FFFD0(_QWORD **a1, unsigned __int8 *a2)
{
  int v2; // eax
  int v3; // ecx
  __int64 v6; // r12
  int v7; // edx
  unsigned int v8; // esi
  unsigned __int8 v9; // al
  __int64 *v10; // rax
  unsigned __int8 *v11; // rbx
  __int64 v12; // rax
  char v13; // r12
  unsigned __int8 *v14; // rdx
  unsigned __int8 *v15; // r12
  unsigned __int8 v16; // al
  _BYTE *v17; // r12
  bool v18; // al
  unsigned __int8 *v19; // rbx
  unsigned __int8 *v20; // rdx
  unsigned __int8 *v21; // r14
  unsigned __int8 v22; // al
  unsigned __int8 v23; // al
  _BYTE *v24; // r14
  __int64 v25; // r15
  __int64 v26; // rdx
  void **v27; // rax
  void **v28; // r14
  void **v29; // r14
  __int64 v30; // r15
  __int64 v31; // rdx
  void **v32; // rax
  void **v33; // r12
  void **v34; // rax
  int v35; // r14d
  unsigned int v36; // r15d
  void **v37; // rax
  void **v38; // rdx
  char v39; // al
  _BYTE *v40; // rdx
  int v41; // r12d
  unsigned int i; // r15d
  void **v43; // rax
  void **v44; // rdx
  char v45; // al
  void **v46; // rdx
  char v47; // [rsp-41h] [rbp-41h]
  void **v48; // [rsp-40h] [rbp-40h]
  void **v49; // [rsp-40h] [rbp-40h]

  v2 = *a2;
  if ( (unsigned __int8)v2 <= 0x1Cu )
    return 0;
  v3 = v2 - 29;
  switch ( *a2 )
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
      goto LABEL_3;
    case '*':
    case ',':
    case '.':
    case '0':
    case '1':
    case '3':
    case '4':
    case '6':
    case '7':
    case '8':
    case '9':
    case ':':
    case ';':
    case '<':
    case '=':
    case '>':
    case '?':
    case '@':
    case 'A':
    case 'B':
    case 'C':
    case 'D':
    case 'E':
    case 'F':
    case 'G':
    case 'H':
    case 'I':
    case 'L':
    case 'M':
    case 'N':
    case 'O':
    case 'P':
    case 'Q':
    case 'R':
      return 0;
    case 'T':
    case 'U':
    case 'V':
      v6 = *((_QWORD *)a2 + 1);
      v7 = *(unsigned __int8 *)(v6 + 8);
      v8 = v7 - 17;
      v9 = *(_BYTE *)(v6 + 8);
      if ( (unsigned int)(v7 - 17) <= 1 )
        v9 = *(_BYTE *)(**(_QWORD **)(v6 + 16) + 8LL);
      if ( v9 <= 3u || v9 == 5 || (v9 & 0xFD) == 4 )
        goto LABEL_4;
      if ( (_BYTE)v7 == 15 )
      {
        if ( (*(_BYTE *)(v6 + 9) & 4) == 0 || !sub_BCB420(*((_QWORD *)a2 + 1)) )
          return 0;
        v10 = *(__int64 **)(v6 + 16);
        v6 = *v10;
        v7 = *(unsigned __int8 *)(*v10 + 8);
        v8 = v7 - 17;
      }
      else if ( (_BYTE)v7 == 16 )
      {
        do
        {
          v6 = *(_QWORD *)(v6 + 24);
          LOBYTE(v7) = *(_BYTE *)(v6 + 8);
        }
        while ( (_BYTE)v7 == 16 );
        v8 = (unsigned __int8)v7 - 17;
      }
      if ( v8 <= 1 )
        LOBYTE(v7) = *(_BYTE *)(**(_QWORD **)(v6 + 16) + 8LL);
      if ( (unsigned __int8)v7 > 3u && (_BYTE)v7 != 5 && (v7 & 0xFD) != 4 )
        return 0;
      v2 = *a2;
      if ( (unsigned __int8)v2 > 0x1Cu )
      {
LABEL_3:
        v3 = v2 - 29;
LABEL_4:
        if ( v3 == 12 )
        {
LABEL_22:
          if ( (a2[7] & 0x40) != 0 )
            v11 = (unsigned __int8 *)*((_QWORD *)a2 - 1);
          else
            v11 = &a2[-32 * (*((_DWORD *)a2 + 1) & 0x7FFFFFF)];
          v12 = *(_QWORD *)v11;
          if ( *(_QWORD *)v11 )
            goto LABEL_25;
          return 0;
        }
      }
      else
      {
        v3 = *((unsigned __int16 *)a2 + 1);
        if ( v3 == 12 )
          goto LABEL_22;
      }
      if ( v3 != 16 )
        return 0;
      v13 = a2[7] & 0x40;
      if ( (a2[1] & 0x10) != 0 )
      {
        if ( v13 )
          v14 = (unsigned __int8 *)*((_QWORD *)a2 - 1);
        else
          v14 = &a2[-32 * (*((_DWORD *)a2 + 1) & 0x7FFFFFF)];
        v15 = *(unsigned __int8 **)v14;
        v16 = **(_BYTE **)v14;
        if ( v16 == 18 )
        {
          if ( *((void **)v15 + 3) == sub_C33340() )
            v17 = (_BYTE *)*((_QWORD *)v15 + 4);
          else
            v17 = v15 + 24;
          v18 = (v17[20] & 7) == 3;
        }
        else
        {
          v25 = *((_QWORD *)v15 + 1);
          v26 = (unsigned int)*(unsigned __int8 *)(v25 + 8) - 17;
          if ( (unsigned int)v26 > 1 || v16 > 0x15u )
            return 0;
          v27 = (void **)sub_AD7630((__int64)v15, 0, v26);
          v28 = v27;
          if ( !v27 || *(_BYTE *)v27 != 18 )
          {
            if ( *(_BYTE *)(v25 + 8) == 17 )
            {
              v35 = *(_DWORD *)(v25 + 32);
              if ( v35 )
              {
                v47 = 0;
                v36 = 0;
                while ( 1 )
                {
                  v37 = (void **)sub_AD69F0(v15, v36);
                  v38 = v37;
                  if ( !v37 )
                    break;
                  v39 = *(_BYTE *)v37;
                  v48 = v38;
                  if ( v39 != 13 )
                  {
                    if ( v39 != 18 )
                      return 0;
                    v40 = v38[3] == sub_C33340() ? v48[4] : v48 + 3;
                    if ( (v40[20] & 7) != 3 )
                      return 0;
                    v47 = 1;
                  }
                  if ( v35 == ++v36 )
                    goto LABEL_86;
                }
              }
            }
            return 0;
          }
          if ( v27[3] == sub_C33340() )
            v29 = (void **)v28[4];
          else
            v29 = v28 + 3;
          v18 = (*((_BYTE *)v29 + 20) & 7) == 3;
        }
        if ( !v18 )
          return 0;
LABEL_34:
        v13 = a2[7] & 0x40;
        goto LABEL_35;
      }
      if ( v13 )
        v20 = (unsigned __int8 *)*((_QWORD *)a2 - 1);
      else
        v20 = &a2[-32 * (*((_DWORD *)a2 + 1) & 0x7FFFFFF)];
      v21 = *(unsigned __int8 **)v20;
      v22 = **(_BYTE **)v20;
      if ( v22 != 18 )
      {
        v30 = *((_QWORD *)v21 + 1);
        v31 = (unsigned int)*(unsigned __int8 *)(v30 + 8) - 17;
        if ( (unsigned int)v31 > 1 || v22 > 0x15u )
          return 0;
        v32 = (void **)sub_AD7630((__int64)v21, 0, v31);
        v33 = v32;
        if ( v32 && *(_BYTE *)v32 == 18 )
        {
          if ( v32[3] == sub_C33340() )
          {
            v34 = (void **)v33[4];
            if ( (*((_BYTE *)v34 + 20) & 7) != 3 )
              return 0;
          }
          else
          {
            if ( (*((_BYTE *)v33 + 44) & 7) != 3 )
              return 0;
            v34 = v33 + 3;
          }
          if ( (*((_BYTE *)v34 + 20) & 8) == 0 )
            return 0;
        }
        else
        {
          if ( *(_BYTE *)(v30 + 8) != 17 )
            return 0;
          v41 = *(_DWORD *)(v30 + 32);
          if ( !v41 )
            return 0;
          v47 = 0;
          for ( i = 0; i != v41; ++i )
          {
            v43 = (void **)sub_AD69F0(v21, i);
            v44 = v43;
            if ( !v43 )
              return 0;
            v45 = *(_BYTE *)v43;
            v49 = v44;
            if ( v45 != 13 )
            {
              if ( v45 != 18 )
                return 0;
              if ( v44[3] == sub_C33340() )
              {
                v46 = (void **)v49[4];
                if ( (*((_BYTE *)v46 + 20) & 7) != 3 )
                  return 0;
              }
              else
              {
                if ( (*((_BYTE *)v49 + 44) & 7) != 3 )
                  return 0;
                v46 = v49 + 3;
              }
              if ( (*((_BYTE *)v46 + 20) & 8) == 0 )
                return 0;
              v47 = 1;
            }
          }
LABEL_86:
          if ( !v47 )
            return 0;
        }
        goto LABEL_34;
      }
      if ( *((void **)v21 + 3) == sub_C33340() )
      {
        v24 = (_BYTE *)*((_QWORD *)v21 + 4);
        if ( (v24[20] & 7) != 3 )
          return 0;
      }
      else
      {
        v23 = v21[44];
        v24 = v21 + 24;
        if ( (v23 & 7) != 3 )
          return 0;
      }
      if ( (v24[20] & 8) == 0 )
        return 0;
LABEL_35:
      if ( v13 )
        v19 = (unsigned __int8 *)*((_QWORD *)a2 - 1);
      else
        v19 = &a2[-32 * (*((_DWORD *)a2 + 1) & 0x7FFFFFF)];
      v12 = *((_QWORD *)v19 + 4);
      if ( !v12 )
        return 0;
LABEL_25:
      **a1 = v12;
      return 1;
    default:
      return 0;
  }
}
