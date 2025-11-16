// Function: sub_11AF4F0
// Address: 0x11af4f0
//
__int64 __fastcall sub_11AF4F0(unsigned __int8 *a1, _DWORD *a2, unsigned __int64 a3, int a4)
{
  unsigned int v4; // r13d
  int v5; // eax
  unsigned __int8 *v6; // r8
  __int64 v8; // r14
  __int64 v10; // rdx
  _DWORD *v12; // rax
  __int64 v13; // rdx
  __int64 v14; // r14
  unsigned __int8 *v15; // rax
  unsigned __int8 *v16; // rsi
  unsigned int v17; // r15d
  unsigned __int8 *v18; // r14
  unsigned __int8 *v19; // r8
  __int64 v20; // rsi
  int v21; // edx
  _DWORD *v22; // rax
  int v23; // eax
  int v24; // [rsp+Ch] [rbp-54h]
  unsigned __int8 *v25; // [rsp+10h] [rbp-50h]
  int v26; // [rsp+18h] [rbp-48h]
  unsigned int v27; // [rsp+18h] [rbp-48h]
  int v28[13]; // [rsp+2Ch] [rbp-34h] BYREF

  v5 = *a1;
  if ( (unsigned __int8)v5 <= 0x15u )
    return 1;
  v6 = a1;
  v8 = (__int64)&a2[a3];
LABEL_3:
  if ( (unsigned __int8)v5 > 0x1Cu )
  {
    v10 = *((_QWORD *)v6 + 2);
    if ( v10 )
    {
      LOBYTE(v4) = a4 == 0 || *(_QWORD *)(v10 + 8) != 0;
      if ( !(_BYTE)v4 )
      {
        switch ( v5 )
        {
          case '*':
          case '+':
          case ',':
          case '-':
          case '.':
          case '/':
          case '2':
          case '5':
          case '6':
          case '7':
          case '8':
          case '9':
          case ':':
          case ';':
          case '?':
          case 'C':
          case 'D':
          case 'E':
          case 'F':
          case 'G':
          case 'H':
          case 'I':
          case 'J':
          case 'K':
          case 'R':
          case 'S':
            goto LABEL_10;
          case '0':
          case '1':
          case '3':
          case '4':
            v26 = a4;
            v28[0] = -1;
            v12 = sub_11AEE30(a2, v8, v28);
            a4 = v26;
            if ( (_DWORD *)v8 != v12 )
              return 0;
LABEL_10:
            v13 = *((_QWORD *)v6 + 1);
            if ( (unsigned int)*(unsigned __int8 *)(v13 + 8) - 17 <= 1 && *(unsigned int *)(v13 + 32) < a3 )
              return 0;
            v14 = 32LL * (*((_DWORD *)v6 + 1) & 0x7FFFFFF);
            v15 = &v6[-v14];
            if ( (v6[7] & 0x40) != 0 )
              v15 = (unsigned __int8 *)*((_QWORD *)v6 - 1);
            v16 = &v15[v14];
            v17 = a4 - 1;
            v18 = v15;
            if ( v16 == v15 )
              return 1;
            while ( 1 )
            {
              v4 = sub_11AF4F0(*(_QWORD *)v18, a2, a3, v17);
              if ( !(_BYTE)v4 )
                break;
              v18 += 32;
              if ( v16 == v18 )
                return 1;
            }
            return v4;
          case '[':
            if ( (v6[7] & 0x40) != 0 )
              v19 = (unsigned __int8 *)*((_QWORD *)v6 - 1);
            else
              v19 = &v6[-32 * (*((_DWORD *)v6 + 1) & 0x7FFFFFF)];
            v20 = *((_QWORD *)v19 + 8);
            if ( *(_BYTE *)v20 != 17 )
              return 0;
            v27 = *(_DWORD *)(v20 + 32);
            if ( v27 > 0x40 )
            {
              v24 = a4;
              v25 = v19;
              v23 = sub_C444A0(v20 + 24);
              v19 = v25;
              a4 = v24;
              v21 = -1;
              if ( v27 - v23 <= 0x40 )
                v21 = **(_DWORD **)(v20 + 24);
            }
            else
            {
              v21 = *(_DWORD *)(v20 + 24);
            }
            if ( a2 == (_DWORD *)v8 )
              goto LABEL_32;
            v22 = a2;
            break;
          default:
            return v4;
        }
        while ( 1 )
        {
          if ( v21 == *v22 )
          {
            if ( (_BYTE)v4 )
              return 0;
            if ( (_DWORD *)v8 == v22 + 1 )
            {
LABEL_32:
              v6 = *(unsigned __int8 **)v19;
              --a4;
              v5 = *v6;
              if ( (unsigned __int8)v5 <= 0x15u )
                return 1;
              goto LABEL_3;
            }
            if ( v21 == v22[1] )
              return 0;
            ++v22;
            v4 = 1;
          }
          if ( ++v22 == (_DWORD *)v8 )
            goto LABEL_32;
        }
      }
    }
  }
  return 0;
}
