// Function: sub_888610
// Address: 0x888610
//
char *__fastcall sub_888610(char *a1, _DWORD *a2, int *a3, char **a4, int a5)
{
  char *result; // rax
  int v8; // edx
  int v9; // r13d
  unsigned __int64 v10; // r14
  char v11; // cl
  char v12; // si
  char v13; // dl
  unsigned int v14; // r15d
  char *v15; // r14
  char *v16; // rcx
  char *v17; // r10
  bool v18; // cl
  char *v19; // rdi
  bool v20; // dl
  __int64 v21; // rsi
  unsigned __int64 v22; // rax
  unsigned __int64 v23; // rax
  bool v24; // [rsp+Fh] [rbp-51h]
  char *endptr[7]; // [rsp+28h] [rbp-38h] BYREF

  result = a1;
  while ( 1 )
  {
    v8 = (unsigned __int8)*result;
    if ( !(_BYTE)v8 )
      return result;
    v9 = 0;
    if ( (_BYTE)v8 == 83 )
    {
      v8 = (unsigned __int8)result[1];
      v9 = 1;
      ++result;
    }
    if ( (_BYTE)v8 == 103 )
    {
      if ( !HIDWORD(qword_4F077B4) )
        goto LABEL_62;
      v10 = qword_4F077A8;
      if ( (unsigned int)qword_4F077B4 | a5 )
      {
        v11 = result[2];
        goto LABEL_16;
      }
    }
    else if ( (_BYTE)v8 == 76 )
    {
      if ( a5 )
        goto LABEL_66;
      if ( !HIDWORD(qword_4F077B4) || !(_DWORD)qword_4F077B4 )
      {
LABEL_74:
        v11 = result[2];
        v10 = qword_4F077A0;
        goto LABEL_16;
      }
      v10 = qword_4F077A0;
    }
    else
    {
      if ( (unsigned __int8)(v8 - 97) > 0x12u || (v21 = 274689, !_bittest64(&v21, (unsigned int)(v8 - 97))) )
        sub_721090();
      switch ( (_BYTE)v8 )
      {
        case 'm':
          v11 = result[2];
          v10 = qword_4F07788;
          goto LABEL_16;
        case 'i':
          if ( !unk_4D04558 )
          {
LABEL_62:
            v11 = result[2];
            v10 = qword_4F077A8;
            goto LABEL_16;
          }
          v10 = qword_4F077A8;
          if ( !HIDWORD(qword_4F077B4) )
          {
            v11 = result[2];
LABEL_16:
            v13 = v11;
            if ( v11 == 65 || v11 == 88 )
            {
              v13 = result[3];
              v16 = result + 3;
              result += 3;
              if ( v13 != 52 && v13 != 56 )
              {
LABEL_20:
                v14 = 0;
                goto LABEL_21;
              }
              goto LABEL_30;
            }
            result += 2;
            if ( v11 == 52 )
            {
              v16 = result;
LABEL_30:
              v14 = 0;
              goto LABEL_31;
            }
            if ( v11 != 56 )
              goto LABEL_20;
            v16 = result;
            v14 = 0;
            goto LABEL_31;
          }
          break;
        case 'a':
          v10 = qword_4F077A8;
          if ( qword_4F077B4 )
          {
            v11 = result[2];
            goto LABEL_16;
          }
          break;
        case 'n':
          if ( !a5 )
            goto LABEL_74;
LABEL_66:
          v10 = (-(__int64)(unk_4D04600 < 0x337D4u) & 0xFFFFFFFFFFFF3CB0LL) + 110000;
          break;
        default:
          v10 = unk_4F07778;
          break;
      }
    }
    v11 = result[2];
    v12 = result[1];
    v13 = v11;
    if ( v12 != 120 )
    {
      if ( v12 == 99 )
      {
        if ( dword_4F077C4 == 2 )
          goto LABEL_16;
      }
      else if ( v12 != 43 || dword_4F077C4 != 2 )
      {
        goto LABEL_16;
      }
    }
    if ( v11 == 65 )
    {
      v16 = result + 3;
      v13 = result[3];
      v14 = qword_4F06A78 != 0;
      if ( v13 == 52 )
        goto LABEL_68;
    }
    else
    {
      if ( v11 != 88 )
      {
        result += 2;
        if ( v11 != 52 )
        {
          v14 = 1;
          if ( v11 != 56 )
            goto LABEL_21;
          v16 = result;
LABEL_42:
          v14 = 1;
          if ( !(unk_4D045F0 | HIDWORD(qword_4F06A78) | unk_4F06A80) )
          {
            v14 = dword_4D045F4;
            if ( dword_4D045F4 )
            {
              v14 = 1;
              if ( dword_4D0455C )
                v14 = unk_4D04600 > 0x30DA3u;
            }
          }
          goto LABEL_31;
        }
        v16 = result;
        goto LABEL_69;
      }
      v16 = result + 3;
      v13 = result[3];
      v14 = qword_4F06A78 == 0;
      if ( v13 == 52 )
      {
LABEL_68:
        if ( !v14 )
          goto LABEL_31;
LABEL_69:
        v14 = 0;
        if ( !(unk_4D045F0 | HIDWORD(qword_4F06A78) | unk_4F06A80) )
        {
          v14 = 1;
          if ( dword_4D045F4 )
          {
            v14 = dword_4D0455C;
            if ( dword_4D0455C )
              v14 = unk_4D04600 <= 0x30DA3u;
          }
        }
        goto LABEL_31;
      }
    }
    if ( v13 != 56 )
    {
      result = v16;
LABEL_21:
      if ( v13 == 40 )
        goto LABEL_32;
      goto LABEL_22;
    }
    if ( v14 )
      goto LABEL_42;
LABEL_31:
    v13 = v16[1];
    result = v16 + 1;
    if ( v13 == 40 )
    {
LABEL_32:
      v17 = result + 1;
      v18 = 1;
      endptr[0] = result + 1;
      if ( result[1] == 45 )
      {
LABEL_33:
        v19 = v17 + 1;
        v20 = 1;
        endptr[0] = v17 + 1;
        if ( (unsigned __int8)(v17[1] - 48) <= 9u )
        {
          v24 = v18;
          v23 = strtoul(v19, endptr, 10);
          v19 = endptr[0];
          v18 = v24;
          v20 = v23 >= v10;
        }
      }
      else
      {
        v22 = strtoul(result + 1, endptr, 10);
        v19 = endptr[0];
        v18 = v22 <= v10;
        if ( *endptr[0] == 45 )
        {
          v17 = endptr[0];
          goto LABEL_33;
        }
        v20 = v22 >= v10;
      }
      result = v19 + 1;
      LOBYTE(v14) = v20 & v18 & v14;
      v13 = v19[1];
      v14 = (unsigned __int8)v14;
    }
LABEL_22:
    v15 = 0;
    if ( v13 == 91 )
    {
      v15 = result + 1;
      result = strchr(result + 1, 93) + 1;
    }
    if ( v14 )
    {
      *a2 = 1;
      *a4 = v15;
      if ( !*a3 )
      {
        *a3 = v9;
        if ( v9 )
          return result;
      }
    }
  }
}
