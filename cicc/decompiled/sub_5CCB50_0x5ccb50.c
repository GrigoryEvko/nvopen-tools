// Function: sub_5CCB50
// Address: 0x5ccb50
//
__int64 __fastcall sub_5CCB50(char *a1, __int64 a2, __int64 a3, char a4)
{
  char v4; // al
  int v8; // ecx
  char v9; // dl
  char v10; // al
  unsigned int v11; // r13d
  char v12; // al
  _BYTE *v13; // rdi
  unsigned int v14; // r14d
  unsigned int v15; // r13d
  char *v16; // rax
  char v17; // al
  char *v19; // rdi
  char v20; // al
  char *v21; // rdx
  char v22; // cl
  unsigned int v23; // r13d
  bool v24; // zf
  bool v25; // r14
  char *v26; // rax
  char v27; // al
  char v28; // dl
  unsigned __int8 v29; // dl
  __int64 v30; // rax
  char v31; // al
  _BYTE *v32; // r13
  _BYTE *v33; // rdx
  char v34; // cl
  unsigned int v35; // r13d
  bool v36; // bl
  char *v37; // rax
  int v38; // eax

  v4 = *a1;
  if ( !*a1 )
    goto LABEL_32;
LABEL_2:
  v8 = 0;
  if ( v4 == 87 )
  {
    v4 = a1[1];
    v8 = 1;
    ++a1;
  }
  switch ( v4 )
  {
    case '0':
      if ( !a4 )
        goto LABEL_36;
      goto LABEL_6;
    case 'E':
      if ( a4 == 2 )
        goto LABEL_36;
      goto LABEL_6;
    case 'T':
    case 'c':
    case 'e':
    case 't':
      if ( a4 != 6 )
        goto LABEL_6;
      v29 = *(_BYTE *)(a2 + 10) - 2;
      if ( v4 != 99 )
      {
        if ( v4 == 101 )
        {
          if ( v29 > 1u || *(_BYTE *)(a3 + 140) != 2 || (*(_BYTE *)(a3 + 161) & 8) == 0 )
            goto LABEL_6;
        }
        else if ( v29 <= 1u )
        {
          goto LABEL_6;
        }
LABEL_101:
        if ( v8 )
        {
LABEL_37:
          sub_5CCAE0(5u, a2);
          goto LABEL_32;
        }
        if ( a1[1] != 58 )
          goto LABEL_32;
        v31 = a1[2];
        v32 = a1 + 2;
        if ( v31 == 124 || !v31 )
          goto LABEL_32;
        while ( 1 )
        {
          v34 = v32[1];
          if ( v34 == 102 )
          {
            a1 = (char *)a3;
            v38 = sub_8D2310(a3);
            v33 = v32 + 2;
            if ( v38 )
            {
              if ( *v32 == 45 )
              {
                v35 = 1858;
                goto LABEL_118;
              }
            }
            else if ( *v32 == 43 )
            {
              v35 = 1859;
LABEL_118:
              v36 = *v33 == 33;
              v37 = sub_5C79F0(a2);
              sub_6849F0((unsigned __int8)(3 * v36 + 5), v35, a2 + 56, v37);
              *(_BYTE *)(a2 + 8) = 0;
LABEL_32:
              v17 = *(_BYTE *)(a2 + 9);
              if ( v17 == 4 || (v11 = 1, v17 == 1) )
              {
                v30 = *(_QWORD *)(a2 + 48);
                v11 = 1;
                if ( v30 )
                {
                  if ( (*(_BYTE *)(v30 + 8) & 8) != 0
                    && (*(_BYTE *)(v30 + 122) & 1) == 0
                    && (!unk_4F077BC || unk_4F077B4) )
                  {
LABEL_98:
                    v11 = 0;
                    sub_5CCAE0(8u, a2);
                  }
                }
              }
              return v11;
            }
          }
          else
          {
            if ( (unsigned __int8)(*(_BYTE *)(a2 + 10) - 2) > 1u || v34 != 100 )
              goto LABEL_126;
            v33 = v32 + 2;
            if ( (*(_BYTE *)(a2 + 11) & 1) != 0 )
            {
              if ( v31 == 45 )
              {
                v35 = 1883;
                goto LABEL_118;
              }
            }
            else if ( v31 == 43 )
            {
              v35 = 1884;
              goto LABEL_118;
            }
          }
          v31 = *v33;
          if ( *v33 == 33 )
          {
            v31 = v32[3];
            ++v33;
          }
          if ( !v31 || v31 == 124 )
            goto LABEL_32;
          v32 = v33;
        }
      }
      if ( v29 <= 1u && (unsigned __int8)(*(_BYTE *)(a3 + 140) - 9) <= 2u )
        goto LABEL_101;
      do
      {
LABEL_6:
        v9 = v4;
        v4 = *++a1;
        if ( v9 == 124 )
          goto LABEL_2;
      }
      while ( v4 );
      if ( (*(_DWORD *)(a2 + 8) & 0xFFFF00) == 0x20300 )
        goto LABEL_121;
      v10 = *(_BYTE *)(a2 + 9);
      if ( (v10 == 2 || (*(_BYTE *)(a2 + 11) & 0x10) != 0) && a4 == 6 && (*(_BYTE *)(a2 + 11) & 2) == 0 )
        goto LABEL_121;
      if ( v10 == 4 || !a3 )
        goto LABEL_98;
      v11 = unk_4F077B4;
      if ( unk_4F077B4 )
      {
LABEL_121:
        v11 = 0;
        sub_5CCAE0(5u, a2);
      }
      else if ( unk_4F077B8 && unk_4F077A8 > 0x9D6Bu )
      {
        sub_5CCAE0(5u, a2);
      }
      else
      {
        sub_5CCAE0(8u, a2);
      }
      return v11;
    case 'd':
      if ( a4 != 8 )
        goto LABEL_6;
      if ( v8 )
        goto LABEL_37;
      if ( a1[1] != 58 )
        goto LABEL_32;
      a1 += 2;
      while ( 2 )
      {
        v12 = *a1;
LABEL_25:
        if ( v12 == 124 || !v12 )
          goto LABEL_32;
        if ( a1[1] != 98 )
          goto LABEL_126;
        if ( (*(_BYTE *)(a3 + 144) & 4) != 0 )
        {
          if ( v12 == 45 )
          {
            v13 = a1 + 2;
            v14 = 1836;
            goto LABEL_31;
          }
        }
        else if ( v12 == 43 )
        {
          v13 = a1 + 2;
          v14 = 1837;
LABEL_31:
          *(_BYTE *)(a2 + 8) = 0;
          v15 = 3 * (*v13 == 33) + 5;
          v16 = sub_5C79F0(a2);
          sub_6849F0(v15, v14, a2 + 56, v16);
          *(_BYTE *)(a2 + 8) = 0;
          goto LABEL_32;
        }
        v12 = a1[2];
        if ( v12 == 33 )
        {
          a1 += 3;
          continue;
        }
        break;
      }
      a1 += 2;
      goto LABEL_25;
    case 'l':
      if ( a4 == 12 )
        goto LABEL_36;
      goto LABEL_6;
    case 'n':
      if ( a4 == 28 )
        goto LABEL_36;
      goto LABEL_6;
    case 'p':
      if ( a4 == 3 )
        goto LABEL_36;
      goto LABEL_6;
    case 'r':
      if ( a4 != 11 )
        goto LABEL_6;
      if ( v8 )
        goto LABEL_37;
      if ( a1[1] != 58 )
        goto LABEL_32;
      a1 += 2;
      while ( 2 )
      {
        v27 = *a1;
LABEL_76:
        if ( v27 == 124 || !v27 )
          goto LABEL_32;
        v28 = a1[1];
        switch ( v28 )
        {
          case 'm':
            a1 += 2;
            if ( (*(_BYTE *)(a3 + 89) & 4) != 0 )
            {
              if ( v27 == 45 )
              {
                v23 = 1838;
LABEL_82:
                v24 = *a1 == 33;
                goto LABEL_59;
              }
            }
            else if ( v27 == 43 )
            {
              v23 = 1839;
              goto LABEL_82;
            }
            goto LABEL_75;
          case 'v':
            a1 += 2;
            if ( (*(_BYTE *)(a3 + 195) & 8) == 0
              || (*(_BYTE *)(a3 + 89) & 4) == 0
              || *(_QWORD *)(a3 + 240)
              || !**(_QWORD **)(*(_QWORD *)(*(_QWORD *)(a3 + 40) + 32LL) + 168LL) )
            {
              if ( (*(_BYTE *)(a3 + 192) & 2) != 0 )
              {
                if ( v27 == 45 )
                {
                  v23 = 1840;
                  goto LABEL_82;
                }
              }
              else if ( v27 == 43 )
              {
                v23 = 1841;
                goto LABEL_82;
              }
            }
LABEL_75:
            v27 = *a1;
            if ( *a1 == 33 )
            {
              ++a1;
              continue;
            }
            goto LABEL_76;
          case 'p':
            a1 += 2;
            if ( (*(_BYTE *)(a3 + 192) & 8) != 0 )
            {
              if ( v27 == 45 )
              {
                v23 = 1842;
                goto LABEL_82;
              }
            }
            else if ( v27 == 43 )
            {
              v23 = 1843;
              goto LABEL_82;
            }
            goto LABEL_75;
          case 'x':
            a1 += 2;
            if ( *(_BYTE *)(a3 + 172) <= 1u )
            {
              if ( v27 == 45 )
              {
                v23 = 1863;
                goto LABEL_82;
              }
            }
            else if ( v27 == 43 && (!unk_4F077B4 || *(_BYTE *)(a2 + 8) != 75) )
            {
              v23 = 1407;
              goto LABEL_82;
            }
            goto LABEL_75;
          case 'i':
            a1 += 2;
            if ( *(char *)(a3 + 192) < 0 )
            {
              if ( v27 == 45 )
              {
                v23 = 1870;
                goto LABEL_82;
              }
            }
            else if ( v27 == 43 )
            {
              v23 = 1871;
              goto LABEL_82;
            }
            goto LABEL_75;
        }
      }
      goto LABEL_126;
    case 's':
      if ( a4 == 21 )
        goto LABEL_36;
      goto LABEL_6;
    case 'u':
      if ( a4 != 29 )
        goto LABEL_6;
LABEL_36:
      if ( v8 )
        goto LABEL_37;
      goto LABEL_32;
    case 'v':
      if ( a4 != 7 )
        goto LABEL_6;
      if ( v8 )
        goto LABEL_37;
      if ( a1[1] != 58 )
        goto LABEL_32;
      v19 = a1 + 2;
      while ( 2 )
      {
        v20 = *v19;
        v21 = v19 + 2;
LABEL_52:
        if ( v20 == 124 || !v20 )
          goto LABEL_32;
        v22 = *(v21 - 1);
        a1 = v21;
        switch ( v22 )
        {
          case 'a':
            if ( *(_BYTE *)(a3 + 136) <= 2u )
            {
              if ( v20 == 43 )
              {
                v23 = 1862;
                goto LABEL_58;
              }
            }
            else if ( v20 == 45 )
            {
              v23 = 1861;
LABEL_58:
              v24 = *v21 == 33;
LABEL_59:
              v25 = v24;
              v26 = sub_5C79F0(a2);
              sub_6849F0(3 * (unsigned int)v25 + 5, v23, a2 + 56, v26);
              *(_BYTE *)(a2 + 8) = 0;
              goto LABEL_32;
            }
            break;
          case 'h':
            if ( *(char *)(a3 + 171) < 0 )
            {
              if ( v20 == 45 )
              {
                v23 = 2468;
                goto LABEL_58;
              }
            }
            else if ( v20 == 43 )
            {
              v23 = 2469;
              goto LABEL_58;
            }
            break;
          case 'l':
            if ( (*(_BYTE *)(a3 + 89) & 1) == 0 || *(_BYTE *)(a3 + 136) == 1 )
            {
              if ( v20 == 43 )
              {
                v23 = 1864;
                goto LABEL_58;
              }
            }
            else if ( v20 == 45 )
            {
              v23 = 1112;
              goto LABEL_58;
            }
            break;
          case 'r':
            if ( *(_BYTE *)(a3 + 136) == 5 )
            {
              if ( v20 == 45 )
              {
                v23 = 1844;
                goto LABEL_58;
              }
            }
            else if ( v20 == 43 )
            {
              v23 = 1845;
              goto LABEL_58;
            }
            break;
          case 'x':
            if ( *(_BYTE *)(a3 + 136) <= 1u )
            {
              if ( v20 == 45 )
              {
                v23 = 1863;
                goto LABEL_58;
              }
            }
            else if ( v20 == 43 )
            {
              v23 = 1407;
              goto LABEL_58;
            }
            break;
          default:
LABEL_126:
            sub_721090(a1);
        }
        v20 = *v21;
        v21 += 2;
        if ( v20 == 33 )
        {
          v19 = a1 + 1;
          continue;
        }
        goto LABEL_52;
      }
    default:
      goto LABEL_126;
  }
}
