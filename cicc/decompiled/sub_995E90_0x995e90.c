// Function: sub_995E90
// Address: 0x995e90
//
__int64 __fastcall sub_995E90(_QWORD **a1, unsigned __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  int v5; // eax
  __int64 v6; // rcx
  _QWORD **v7; // r13
  unsigned __int8 *v8; // rbx
  __int64 v10; // r12
  int v11; // edx
  unsigned __int8 v12; // al
  __int64 *v13; // rax
  unsigned __int8 *v14; // rbx
  __int64 v15; // rax
  char v16; // r12
  unsigned __int8 *v17; // rdx
  unsigned __int8 *v18; // r12
  unsigned __int8 v19; // al
  _QWORD *v20; // r12
  bool v21; // al
  unsigned __int8 *v22; // rbx
  unsigned __int8 *v23; // rdx
  unsigned __int8 *v24; // r14
  unsigned __int8 v25; // al
  unsigned __int8 v26; // al
  _BYTE *v27; // r14
  __int64 v28; // r15
  __int64 v29; // rax
  __int64 v30; // rdx
  __int64 v31; // rcx
  __int64 v32; // r8
  __int64 v33; // r14
  __int64 v34; // r14
  __int64 v35; // r15
  __int64 v36; // rax
  __int64 v37; // rdx
  __int64 v38; // rcx
  __int64 v39; // r8
  __int64 v40; // r12
  __int64 v41; // rax
  int v42; // r14d
  unsigned int v43; // r15d
  _BYTE *v44; // rax
  __int64 v45; // rcx
  __int64 v46; // r8
  _BYTE *v47; // rdx
  char v48; // al
  _BYTE *v49; // rdx
  int v50; // r12d
  unsigned int i; // r15d
  _BYTE *v52; // rax
  __int64 v53; // rcx
  __int64 v54; // r8
  _BYTE *v55; // rdx
  char v56; // al
  _BYTE *v57; // rdx
  char v58; // [rsp-41h] [rbp-41h]
  _BYTE *v59; // [rsp-40h] [rbp-40h]
  _BYTE *v60; // [rsp-40h] [rbp-40h]

  v5 = *(unsigned __int8 *)a2;
  if ( (unsigned __int8)v5 <= 0x1Cu )
    return 0;
  v6 = (unsigned int)(v5 - 29);
  v7 = a1;
  v8 = (unsigned __int8 *)a2;
  switch ( *(_BYTE *)a2 )
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
      v10 = *(_QWORD *)(a2 + 8);
      v11 = *(unsigned __int8 *)(v10 + 8);
      a2 = (unsigned int)(v11 - 17);
      v12 = *(_BYTE *)(v10 + 8);
      if ( (unsigned int)a2 <= 1 )
        v12 = *(_BYTE *)(**(_QWORD **)(v10 + 16) + 8LL);
      if ( v12 <= 3u || v12 == 5 || (v12 & 0xFD) == 4 )
        goto LABEL_4;
      if ( (_BYTE)v11 == 15 )
      {
        if ( (*(_BYTE *)(v10 + 9) & 4) == 0 )
          return 0;
        a1 = (_QWORD **)*((_QWORD *)v8 + 1);
        if ( !(unsigned __int8)sub_BCB420(a1) )
          return 0;
        v13 = *(__int64 **)(v10 + 16);
        v10 = *v13;
        v11 = *(unsigned __int8 *)(*v13 + 8);
        a2 = (unsigned int)(v11 - 17);
      }
      else if ( (_BYTE)v11 == 16 )
      {
        do
        {
          v10 = *(_QWORD *)(v10 + 24);
          LOBYTE(v11) = *(_BYTE *)(v10 + 8);
        }
        while ( (_BYTE)v11 == 16 );
        a2 = (unsigned int)(unsigned __int8)v11 - 17;
      }
      if ( (unsigned int)a2 <= 1 )
        LOBYTE(v11) = *(_BYTE *)(**(_QWORD **)(v10 + 16) + 8LL);
      if ( (unsigned __int8)v11 > 3u && (_BYTE)v11 != 5 && (v11 & 0xFD) != 4 )
        return 0;
      v5 = *v8;
      if ( (unsigned __int8)v5 > 0x1Cu )
      {
LABEL_3:
        v6 = (unsigned int)(v5 - 29);
LABEL_4:
        if ( (_DWORD)v6 == 12 )
        {
LABEL_22:
          if ( (v8[7] & 0x40) != 0 )
            v14 = (unsigned __int8 *)*((_QWORD *)v8 - 1);
          else
            v14 = &v8[-32 * (*((_DWORD *)v8 + 1) & 0x7FFFFFF)];
          v15 = *(_QWORD *)v14;
          if ( *(_QWORD *)v14 )
            goto LABEL_25;
          return 0;
        }
      }
      else
      {
        v6 = *((unsigned __int16 *)v8 + 1);
        if ( (_DWORD)v6 == 12 )
          goto LABEL_22;
      }
      if ( (_DWORD)v6 != 16 )
        return 0;
      v16 = v8[7] & 0x40;
      if ( (v8[1] & 0x10) != 0 )
      {
        if ( v16 )
          v17 = (unsigned __int8 *)*((_QWORD *)v8 - 1);
        else
          v17 = &v8[-32 * (*((_DWORD *)v8 + 1) & 0x7FFFFFF)];
        v18 = *(unsigned __int8 **)v17;
        v19 = **(_BYTE **)v17;
        if ( v19 == 18 )
        {
          if ( *((_QWORD *)v18 + 3) == sub_C33340(a1, a2, v17, v6, a5) )
            v20 = (_QWORD *)*((_QWORD *)v18 + 4);
          else
            v20 = v18 + 24;
          v21 = (*((_BYTE *)v20 + 20) & 7) == 3;
        }
        else
        {
          v28 = *((_QWORD *)v18 + 1);
          if ( (unsigned int)*(unsigned __int8 *)(v28 + 8) - 17 > 1 || v19 > 0x15u )
            return 0;
          v29 = sub_AD7630(v18, 0);
          v33 = v29;
          if ( !v29 || *(_BYTE *)v29 != 18 )
          {
            if ( *(_BYTE *)(v28 + 8) == 17 )
            {
              v42 = *(_DWORD *)(v28 + 32);
              if ( v42 )
              {
                v58 = 0;
                v43 = 0;
                while ( 1 )
                {
                  v44 = (_BYTE *)sub_AD69F0(v18, v43);
                  v47 = v44;
                  if ( !v44 )
                    break;
                  v48 = *v44;
                  v59 = v47;
                  if ( v48 != 13 )
                  {
                    if ( v48 != 18 )
                      return 0;
                    v49 = *((_QWORD *)v47 + 3) == sub_C33340(v18, v43, v47, v45, v46)
                        ? (_BYTE *)*((_QWORD *)v59 + 4)
                        : v59 + 24;
                    if ( (v49[20] & 7) != 3 )
                      return 0;
                    v58 = 1;
                  }
                  if ( v42 == ++v43 )
                    goto LABEL_86;
                }
              }
            }
            return 0;
          }
          if ( *(_QWORD *)(v29 + 24) == sub_C33340(v18, 0, v30, v31, v32) )
            v34 = *(_QWORD *)(v33 + 32);
          else
            v34 = v33 + 24;
          v21 = (*(_BYTE *)(v34 + 20) & 7) == 3;
        }
        if ( !v21 )
          return 0;
LABEL_34:
        v16 = v8[7] & 0x40;
        goto LABEL_35;
      }
      if ( v16 )
        v23 = (unsigned __int8 *)*((_QWORD *)v8 - 1);
      else
        v23 = &v8[-32 * (*((_DWORD *)v8 + 1) & 0x7FFFFFF)];
      v24 = *(unsigned __int8 **)v23;
      v25 = **(_BYTE **)v23;
      if ( v25 != 18 )
      {
        v35 = *((_QWORD *)v24 + 1);
        if ( (unsigned int)*(unsigned __int8 *)(v35 + 8) - 17 > 1 || v25 > 0x15u )
          return 0;
        v36 = sub_AD7630(v24, 0);
        v40 = v36;
        if ( v36 && *(_BYTE *)v36 == 18 )
        {
          if ( *(_QWORD *)(v36 + 24) == sub_C33340(v24, 0, v37, v38, v39) )
          {
            v41 = *(_QWORD *)(v40 + 32);
            if ( (*(_BYTE *)(v41 + 20) & 7) != 3 )
              return 0;
          }
          else
          {
            if ( (*(_BYTE *)(v40 + 44) & 7) != 3 )
              return 0;
            v41 = v40 + 24;
          }
          if ( (*(_BYTE *)(v41 + 20) & 8) == 0 )
            return 0;
        }
        else
        {
          if ( *(_BYTE *)(v35 + 8) != 17 )
            return 0;
          v50 = *(_DWORD *)(v35 + 32);
          if ( !v50 )
            return 0;
          v58 = 0;
          for ( i = 0; i != v50; ++i )
          {
            v52 = (_BYTE *)sub_AD69F0(v24, i);
            v55 = v52;
            if ( !v52 )
              return 0;
            v56 = *v52;
            v60 = v55;
            if ( v56 != 13 )
            {
              if ( v56 != 18 )
                return 0;
              if ( *((_QWORD *)v55 + 3) == sub_C33340(v24, i, v55, v53, v54) )
              {
                v57 = (_BYTE *)*((_QWORD *)v60 + 4);
                if ( (v57[20] & 7) != 3 )
                  return 0;
              }
              else
              {
                if ( (v60[44] & 7) != 3 )
                  return 0;
                v57 = v60 + 24;
              }
              if ( (v57[20] & 8) == 0 )
                return 0;
              v58 = 1;
            }
          }
LABEL_86:
          if ( !v58 )
            return 0;
        }
        goto LABEL_34;
      }
      if ( *((_QWORD *)v24 + 3) == sub_C33340(a1, a2, v23, v6, a5) )
      {
        v27 = (_BYTE *)*((_QWORD *)v24 + 4);
        if ( (v27[20] & 7) != 3 )
          return 0;
      }
      else
      {
        v26 = v24[44];
        v27 = v24 + 24;
        if ( (v26 & 7) != 3 )
          return 0;
      }
      if ( (v27[20] & 8) == 0 )
        return 0;
LABEL_35:
      if ( v16 )
        v22 = (unsigned __int8 *)*((_QWORD *)v8 - 1);
      else
        v22 = &v8[-32 * (*((_DWORD *)v8 + 1) & 0x7FFFFFF)];
      v15 = *((_QWORD *)v22 + 4);
      if ( !v15 )
        return 0;
LABEL_25:
      **v7 = v15;
      return 1;
    default:
      return 0;
  }
}
