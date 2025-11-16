// Function: sub_1009310
// Address: 0x1009310
//
bool __fastcall sub_1009310(_QWORD *a1, unsigned __int8 *a2)
{
  __int64 *v2; // rbx
  __int64 *v3; // rbp
  __int64 *v4; // r12
  int v5; // eax
  int v6; // ecx
  char v8; // al
  unsigned __int8 *v9; // rdx
  unsigned __int8 *v10; // r13
  unsigned __int8 v11; // al
  _BYTE *v12; // r13
  char v13; // al
  __int64 v15; // r13
  int v16; // edx
  unsigned int v17; // esi
  unsigned __int8 v18; // al
  __int64 *v19; // rax
  unsigned __int8 *v20; // rbx
  __int64 *v21; // rdx
  int v22; // r14d
  unsigned int i; // r15d
  void **v24; // rax
  void **v25; // rdx
  char v26; // al
  _BYTE *v27; // rdx
  unsigned __int8 *v28; // rbx
  __int64 v29; // r15
  __int64 v30; // rdx
  void **v31; // rax
  void **v32; // r14
  void **v33; // r14
  char v34; // [rsp-51h] [rbp-51h]
  void **v35; // [rsp-50h] [rbp-50h]
  __int64 *v36[8]; // [rsp-40h] [rbp-40h] BYREF

  v5 = *a2;
  if ( (unsigned __int8)v5 <= 0x1Cu )
    return 0;
  v6 = v5 - 29;
  v36[7] = v3;
  v36[3] = v4;
  v36[2] = v2;
  switch ( v5 )
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
      v15 = *((_QWORD *)a2 + 1);
      v16 = *(unsigned __int8 *)(v15 + 8);
      v17 = v16 - 17;
      v18 = *(_BYTE *)(v15 + 8);
      if ( (unsigned int)(v16 - 17) <= 1 )
        v18 = *(_BYTE *)(**(_QWORD **)(v15 + 16) + 8LL);
      if ( v18 <= 3u || v18 == 5 || (v18 & 0xFD) == 4 )
        goto LABEL_4;
      if ( (_BYTE)v16 == 15 )
      {
        if ( (*(_BYTE *)(v15 + 9) & 4) == 0 || !sub_BCB420(*((_QWORD *)a2 + 1)) )
          return 0;
        v19 = *(__int64 **)(v15 + 16);
        v15 = *v19;
        v16 = *(unsigned __int8 *)(*v19 + 8);
        v17 = v16 - 17;
      }
      else if ( (_BYTE)v16 == 16 )
      {
        do
        {
          v15 = *(_QWORD *)(v15 + 24);
          LOBYTE(v16) = *(_BYTE *)(v15 + 8);
        }
        while ( (_BYTE)v16 == 16 );
        v17 = (unsigned __int8)v16 - 17;
      }
      if ( v17 <= 1 )
        LOBYTE(v16) = *(_BYTE *)(**(_QWORD **)(v15 + 16) + 8LL);
      if ( (unsigned __int8)v16 > 3u && (_BYTE)v16 != 5 && (v16 & 0xFD) != 4 )
        return 0;
      v5 = *a2;
      if ( (unsigned __int8)v5 > 0x1Cu )
      {
LABEL_3:
        v6 = v5 - 29;
LABEL_4:
        if ( v6 == 12 )
          goto LABEL_30;
LABEL_5:
        if ( v6 == 16 )
        {
          v8 = a2[7] & 0x40;
          if ( (a2[1] & 0x10) == 0 )
          {
            v36[0] = 0;
            if ( v8 )
              v21 = (__int64 *)*((_QWORD *)a2 - 1);
            else
              v21 = (__int64 *)&a2[-32 * (*((_DWORD *)a2 + 1) & 0x7FFFFFF)];
            v13 = sub_1008640(v36, *v21);
LABEL_13:
            if ( !v13 )
              return 0;
LABEL_52:
            if ( (a2[7] & 0x40) != 0 )
              v28 = (unsigned __int8 *)*((_QWORD *)a2 - 1);
            else
              v28 = &a2[-32 * (*((_DWORD *)a2 + 1) & 0x7FFFFFF)];
            return *((_QWORD *)v28 + 4) == *a1;
          }
          if ( v8 )
            v9 = (unsigned __int8 *)*((_QWORD *)a2 - 1);
          else
            v9 = &a2[-32 * (*((_DWORD *)a2 + 1) & 0x7FFFFFF)];
          v10 = *(unsigned __int8 **)v9;
          v11 = **(_BYTE **)v9;
          if ( v11 == 18 )
          {
            if ( *((void **)v10 + 3) == sub_C33340() )
              v12 = (_BYTE *)*((_QWORD *)v10 + 4);
            else
              v12 = v10 + 24;
            v13 = (v12[20] & 7) == 3;
            goto LABEL_13;
          }
          v29 = *((_QWORD *)v10 + 1);
          v30 = (unsigned int)*(unsigned __int8 *)(v29 + 8) - 17;
          if ( (unsigned int)v30 > 1 || v11 > 0x15u )
            return 0;
          v31 = (void **)sub_AD7630((__int64)v10, 0, v30);
          v32 = v31;
          if ( v31 && *(_BYTE *)v31 == 18 )
          {
            if ( v31[3] == sub_C33340() )
              v33 = (void **)v32[4];
            else
              v33 = v32 + 3;
            v13 = (*((_BYTE *)v33 + 20) & 7) == 3;
            goto LABEL_13;
          }
          if ( *(_BYTE *)(v29 + 8) == 17 )
          {
            v22 = *(_DWORD *)(v29 + 32);
            if ( v22 )
            {
              v34 = 0;
              for ( i = 0; i != v22; ++i )
              {
                v24 = (void **)sub_AD69F0(v10, i);
                v25 = v24;
                if ( !v24 )
                  return 0;
                v26 = *(_BYTE *)v24;
                v35 = v25;
                if ( v26 != 13 )
                {
                  if ( v26 != 18 )
                    return 0;
                  v27 = v25[3] == sub_C33340() ? v35[4] : v35 + 3;
                  if ( (v27[20] & 7) != 3 )
                    return 0;
                  v34 = 1;
                }
              }
              if ( v34 )
                goto LABEL_52;
            }
          }
        }
        return 0;
      }
      v6 = *((unsigned __int16 *)a2 + 1);
      if ( v6 != 12 )
        goto LABEL_5;
LABEL_30:
      if ( (a2[7] & 0x40) != 0 )
        v20 = (unsigned __int8 *)*((_QWORD *)a2 - 1);
      else
        v20 = &a2[-32 * (*((_DWORD *)a2 + 1) & 0x7FFFFFF)];
      return *a1 == *(_QWORD *)v20;
    default:
      return 0;
  }
}
