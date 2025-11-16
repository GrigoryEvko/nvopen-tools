// Function: sub_28EE9F0
// Address: 0x28ee9f0
//
__int64 __fastcall sub_28EE9F0(unsigned __int8 *a1)
{
  __int64 *v1; // rbp
  __int64 *v2; // r12
  int v3; // eax
  int v4; // ecx
  __int64 result; // rax
  __int64 v6; // rbx
  int v7; // edx
  unsigned int v8; // esi
  unsigned __int8 v9; // al
  __int64 *v10; // rax
  unsigned __int8 **v11; // rax
  unsigned __int8 *v12; // rbx
  unsigned __int8 v13; // al
  _BYTE *v14; // rbx
  __int64 *v16; // rax
  __int64 v17; // r13
  __int64 v18; // rdx
  void **v19; // rax
  void **v20; // r12
  void **v21; // r12
  int v22; // r13d
  char v23; // r14
  unsigned int i; // r15d
  void **v25; // rax
  void **v26; // r12
  char v27; // al
  _BYTE *v28; // r12
  __int64 *v29[8]; // [rsp-40h] [rbp-40h] BYREF

  v3 = *a1;
  if ( (unsigned __int8)v3 <= 0x1Cu )
    return 0;
  v4 = v3 - 29;
  v29[7] = v1;
  v29[3] = v2;
  switch ( v3 )
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
      v6 = *((_QWORD *)a1 + 1);
      v7 = *(unsigned __int8 *)(v6 + 8);
      v8 = v7 - 17;
      v9 = *(_BYTE *)(v6 + 8);
      if ( (unsigned int)(v7 - 17) <= 1 )
        v9 = *(_BYTE *)(**(_QWORD **)(v6 + 16) + 8LL);
      if ( v9 <= 3u || v9 == 5 || (v9 & 0xFD) == 4 )
        goto LABEL_4;
      if ( (_BYTE)v7 == 15 )
      {
        if ( (*(_BYTE *)(v6 + 9) & 4) == 0 || !sub_BCB420(*((_QWORD *)a1 + 1)) )
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
      v3 = *a1;
      if ( (unsigned __int8)v3 > 0x1Cu )
LABEL_3:
        v4 = v3 - 29;
      else
        v4 = *((unsigned __int16 *)a1 + 1);
LABEL_4:
      result = 1;
      if ( v4 == 12 )
        return result;
      if ( v4 != 16 )
        return 0;
      if ( (a1[1] & 0x10) != 0 )
      {
        v11 = (unsigned __int8 **)sub_986520((__int64)a1);
        v12 = *v11;
        v13 = **v11;
        if ( v13 == 18 )
        {
          if ( *((void **)v12 + 3) == sub_C33340() )
            v14 = (_BYTE *)*((_QWORD *)v12 + 4);
          else
            v14 = v12 + 24;
          return (v14[20] & 7) == 3;
        }
        else
        {
          v17 = *((_QWORD *)v12 + 1);
          v18 = (unsigned int)*(unsigned __int8 *)(v17 + 8) - 17;
          if ( (unsigned int)v18 > 1 || v13 > 0x15u )
            return 0;
          v19 = (void **)sub_AD7630((__int64)v12, 0, v18);
          v20 = v19;
          if ( !v19 || *(_BYTE *)v19 != 18 )
          {
            if ( *(_BYTE *)(v17 + 8) == 17 )
            {
              v22 = *(_DWORD *)(v17 + 32);
              if ( v22 )
              {
                v23 = 0;
                for ( i = 0; i != v22; ++i )
                {
                  v25 = (void **)sub_AD69F0(v12, i);
                  v26 = v25;
                  if ( !v25 )
                    return 0;
                  v27 = *(_BYTE *)v25;
                  if ( v27 != 13 )
                  {
                    if ( v27 != 18 )
                      return 0;
                    v28 = v26[3] == sub_C33340() ? v26[4] : v26 + 3;
                    if ( (v28[20] & 7) != 3 )
                      return 0;
                    v23 = 1;
                  }
                }
                if ( v23 )
                  return 1;
              }
            }
            return 0;
          }
          if ( v19[3] == sub_C33340() )
            v21 = (void **)v20[4];
          else
            v21 = v20 + 3;
          return (*((_BYTE *)v21 + 20) & 7) == 3;
        }
      }
      v29[0] = 0;
      v16 = (__int64 *)sub_986520((__int64)a1);
      return sub_1008640(v29, *v16);
    default:
      return 0;
  }
}
