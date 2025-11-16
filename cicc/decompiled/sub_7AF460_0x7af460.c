// Function: sub_7AF460
// Address: 0x7af460
//
char *__fastcall sub_7AF460(char *a1, __int64 a2)
{
  char *result; // rax
  char *v3; // r15
  __int64 *v4; // r13
  int v5; // ebx
  int v6; // r14d
  int v7; // r12d
  int v8; // r8d
  __int64 v9; // rdx
  __int64 v10; // rdx
  int v11; // r12d
  _BYTE *v12; // rax
  _BYTE *v13; // rax
  __int64 v14; // rdx
  unsigned __int8 v15; // r9
  __int64 v16; // rax
  char *v17; // rdx
  int v18; // eax
  __int64 v19; // rcx
  _BYTE *v20; // rax
  int v21; // r8d
  _BYTE *v22; // rax
  char v23; // [rsp-3Ah] [rbp-3Ah]
  unsigned __int8 v24; // [rsp-3Ah] [rbp-3Ah]
  char v25; // [rsp-39h] [rbp-39h]
  unsigned __int8 v26; // [rsp-39h] [rbp-39h]

  if ( !dword_4F17FA0 || dword_4F17F7C )
  {
    if ( (_DWORD)a1 )
    {
      v4 = (__int64 *)qword_4F06438;
      v3 = *(char **)(qword_4F06438 + 56LL);
    }
    else
    {
      v3 = (char *)unk_4F06498;
      v4 = (__int64 *)unk_4F06458;
      if ( unk_4F06458 )
      {
        dword_4F17F7C = 1;
        v4 = 0;
      }
    }
    v25 = 0;
    v5 = 0;
    v6 = 0;
    v7 = 10;
    while ( 1 )
    {
      while ( 1 )
      {
        while ( 1 )
        {
          v8 = (unsigned __int8)*v3;
          if ( (_BYTE)v8 != 10 )
            break;
LABEL_18:
          if ( (unsigned int)sub_7AF220((unsigned __int64)v3) )
          {
            a1 = v3;
            v16 = sub_7AF1D0((unsigned __int64)v3);
            *(_BYTE *)(v16 + 48) |= 4u;
            v17 = *(char **)(v16 + 56);
            *(_QWORD *)(v16 + 24) = v4;
            if ( v17 == *(char **)(v16 + 64) )
            {
              v3 += *(_QWORD *)(v16 + 32);
            }
            else
            {
              v4 = (__int64 *)v16;
              v3 = v17;
            }
            v6 = 1;
            if ( (*(_BYTE *)(v16 + 48) & 2) == 0 )
              dword_4F17F7C = 1;
          }
          else
          {
            v11 = dword_4F17F7C;
            v12 = (_BYTE *)qword_4F17F80;
            a2 = (unsigned int)dword_4F17F7C;
            if ( qword_4F17F80 == qword_4F17F88 )
            {
              sub_7ABCA0(v3, (unsigned int)dword_4F17F7C, v10);
              v12 = (_BYTE *)qword_4F17F80;
              a2 = (unsigned int)dword_4F17F7C;
            }
            a1 = v12 + 1;
            *v12 = 10;
            qword_4F17F80 = (__int64)(v12 + 1);
            if ( (_DWORD)a2 )
            {
              v12[1] = 0;
              putc(88, qword_4D04908);
              a1 = qword_4F17F90;
              a2 = (__int64)qword_4D04908;
              fputs(qword_4F17F90, qword_4D04908);
            }
            dword_4F17F7C = v11;
            ++v3;
            v7 = 10;
            qword_4F17F80 = (__int64)qword_4F17F90;
          }
        }
LABEL_9:
        if ( (_BYTE)v8 )
        {
          if ( v5 > 0 )
          {
            v13 = (_BYTE *)qword_4F17F80;
            v14 = qword_4F17F88;
            --v5;
            goto LABEL_28;
          }
          a1 = (char *)dword_4D0432C;
          if ( dword_4D0432C )
          {
            v5 = 0;
            if ( (v8 & 0x80u) != 0 )
            {
              a1 = v3;
              v24 = v8;
              a2 = 0;
              v18 = sub_721AB0(v3, 0, unk_4F064A8 == 0);
              v8 = v24;
              v5 = v18 - 1;
              if ( v18 - 1 > 0 )
              {
                v13 = (_BYTE *)qword_4F17F80;
                v14 = qword_4F17F88;
                goto LABEL_28;
              }
            }
          }
          v13 = (_BYTE *)qword_4F17F80;
          v14 = qword_4F17F88;
          if ( v6 )
          {
            a2 = (__int64)&dword_4F04D98;
            if ( !dword_4F04D98 )
            {
              a1 = (char *)byte_4F04C80;
              a2 = (char)v7 + 128;
              v15 = byte_4F04C80[a2];
              if ( v15 != 1 )
              {
                a2 = byte_4F04C80[(char)v8 + 128];
                if ( (_BYTE)a2 != 1 )
                {
                  if ( v15 == (_BYTE)a2 )
                    goto LABEL_47;
                  a1 = (char *)(v7 & 0xFFFFFFDF);
                  if ( (v7 & 0xDF) == 0x45 && (((_BYTE)v8 - 43) & 0xFD) == 0 )
                    goto LABEL_47;
                  if ( (((_BYTE)v7 - 43) & 0xFD) == 0 && (v25 & 0xDF) == 0x45 )
                  {
                    if ( (unsigned int)(unsigned __int8)v8 - 48 <= 9 )
                      goto LABEL_47;
                    if ( (v7 & 0xDF) == 0x55 || (_BYTE)v7 == 76 )
                    {
LABEL_43:
                      if ( (_BYTE)v8 == 39 || (_BYTE)v8 == 34 )
                        goto LABEL_47;
                    }
                  }
                  else
                  {
                    if ( (_BYTE)v7 == 76 || (v7 & 0xDF) == 0x55 )
                      goto LABEL_43;
                    if ( v25 == 117 && (_BYTE)v7 == 56 )
                    {
                      if ( (_BYTE)v8 != 39 && (_BYTE)v8 != 34 )
                      {
                        v25 = 56;
                        v7 = v8;
LABEL_28:
                        if ( v13 != (_BYTE *)v14 )
                        {
LABEL_29:
                          *v13 = v8;
                          ++v3;
                          v6 = 0;
                          qword_4F17F80 = (__int64)(v13 + 1);
                          continue;
                        }
LABEL_50:
                        v23 = v8;
                        sub_7ABCA0(a1, a2, v14);
                        v13 = (_BYTE *)qword_4F17F80;
                        LOBYTE(v8) = v23;
                        goto LABEL_29;
                      }
LABEL_47:
                      if ( qword_4F17F80 == qword_4F17F88 )
                      {
                        v26 = v8;
                        sub_7ABCA0(a1, a2, qword_4F17F88);
                        v13 = (_BYTE *)qword_4F17F80;
                        v8 = v26;
                        v14 = qword_4F17F88;
                      }
                      a2 = (__int64)(v13 + 1);
                      *v13 = 32;
                      v13 = (_BYTE *)a2;
                      v25 = v7;
                      v7 = v8;
                      qword_4F17F80 = a2;
                      if ( a2 != v14 )
                        goto LABEL_29;
                      goto LABEL_50;
                    }
                  }
                  if ( (_BYTE)v7 == 34 && (_BYTE)a2 == 2 )
                    goto LABEL_47;
                }
              }
            }
          }
          v25 = v7;
          v7 = v8;
          goto LABEL_28;
        }
        result = (char *)(unsigned __int8)v3[1];
        if ( (v3[1] & 0xF6) == 4
          || (v9 = (unsigned int)((_DWORD)result - 7), (unsigned __int8)((_BYTE)result - 7) <= 3u) )
        {
          v3 += 2;
          v6 = 1;
          continue;
        }
        if ( (_BYTE)result != 3 )
          break;
        result = (char *)&qword_4F06438;
        if ( (__int64 *)qword_4F06438 == v4 )
          return result;
        v19 = v4[2];
        if ( !v19 )
        {
          v19 = unk_4F06498;
          if ( unk_4F06478 )
            v19 = unk_4F06470 + unk_4F06498;
        }
        v3 = (char *)(v4[4] + v19);
        if ( (v4[6] & 4) != 0 )
        {
          v4 = (__int64 *)v4[3];
          v6 = 1;
        }
        else
        {
          a1 = (char *)v4;
          v6 = 1;
          v4 = sub_7AF170((__int64)v4);
        }
      }
      switch ( (_BYTE)result )
      {
        case 2:
          v20 = (_BYTE *)qword_4F17F80;
          if ( qword_4F17F80 == qword_4F17F88 )
          {
            sub_7ABCA0(a1, a2, v9);
            v20 = (_BYTE *)qword_4F17F80;
          }
          v21 = dword_4F17F7C;
          *v20 = 10;
          qword_4F17F80 = (__int64)(v20 + 1);
          if ( v21 )
          {
            v20[1] = 0;
            putc(88, qword_4D04908);
            a2 = (__int64)qword_4D04908;
            a1 = qword_4F17F90;
            fputs(qword_4F17F90, qword_4D04908);
          }
          v3 += 2;
          v7 = 10;
          dword_4F17F7C = 0;
          qword_4F17F80 = (__int64)qword_4F17F90;
          break;
        case 1:
          return result;
        case 6:
          v22 = (_BYTE *)qword_4F17F80;
          if ( qword_4F17F80 == qword_4F17F88 )
          {
            sub_7ABCA0(a1, a2, v9);
            v22 = (_BYTE *)qword_4F17F80;
          }
          *v22 = 32;
          v3 += 2;
          v7 = 32;
          qword_4F17F80 = (__int64)(v22 + 1);
          break;
        case 0xB:
          v8 = (unsigned __int8)v3[3];
          v3 += 3;
          if ( (_BYTE)v8 == 10 )
            goto LABEL_18;
          goto LABEL_9;
        default:
          sub_721090();
      }
    }
  }
  qword_4F17F80 = (__int64)qword_4F17F90;
  return qword_4F17F90;
}
