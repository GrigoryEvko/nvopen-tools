// Function: sub_1024900
// Address: 0x1024900
//
__int64 __fastcall sub_1024900(
        __int64 a1,
        __int64 a2,
        _BYTE *a3,
        __int64 a4,
        unsigned int a5,
        __int64 a6,
        char a7,
        __int64 a8)
{
  bool v11; // al
  bool v12; // zf
  __int64 v13; // rax
  bool v14; // al
  __int64 v15; // rax
  int v16; // edx
  __int64 v17; // rax
  char v18; // al
  bool v19; // al
  bool v20; // al
  __int64 v21; // rax
  bool v22; // al
  __int64 v23; // rax
  __int64 v24; // [rsp+8h] [rbp-28h]

  switch ( *(_BYTE *)a4 )
  {
    case '*':
    case ',':
      *(_QWORD *)(a1 + 8) = a4;
      *(_DWORD *)(a1 + 16) = 0;
      *(_QWORD *)(a1 + 24) = 0;
      *(_BYTE *)a1 = a5 == 1;
      return a1;
    case '+':
    case '-':
      v14 = sub_B451B0(a4);
      *(_QWORD *)(a1 + 8) = a4;
      *(_DWORD *)(a1 + 16) = 0;
      v12 = !v14;
      v15 = 0;
      if ( v12 )
        v15 = a4;
      *(_BYTE *)a1 = a5 == 10;
      *(_QWORD *)(a1 + 24) = v15;
      return a1;
    case '.':
      *(_QWORD *)(a1 + 8) = a4;
      *(_DWORD *)(a1 + 16) = 0;
      *(_QWORD *)(a1 + 24) = 0;
      *(_BYTE *)a1 = a5 == 2;
      return a1;
    case '/':
    case '2':
      v11 = sub_B451B0(a4);
      *(_QWORD *)(a1 + 8) = a4;
      *(_DWORD *)(a1 + 16) = 0;
      v12 = !v11;
      v13 = 0;
      if ( v12 )
        v13 = a4;
      *(_BYTE *)a1 = a5 == 11;
      *(_QWORD *)(a1 + 24) = v13;
      return a1;
    case '9':
      *(_QWORD *)(a1 + 8) = a4;
      *(_DWORD *)(a1 + 16) = 0;
      *(_QWORD *)(a1 + 24) = 0;
      *(_BYTE *)a1 = a5 == 4;
      return a1;
    case ':':
      *(_QWORD *)(a1 + 8) = a4;
      *(_DWORD *)(a1 + 16) = 0;
      *(_QWORD *)(a1 + 24) = 0;
      *(_BYTE *)a1 = a5 == 3;
      return a1;
    case ';':
      *(_QWORD *)(a1 + 8) = a4;
      *(_DWORD *)(a1 + 16) = 0;
      *(_QWORD *)(a1 + 24) = 0;
      *(_BYTE *)a1 = a5 == 5;
      return a1;
    case 'R':
    case 'S':
    case 'U':
      goto LABEL_4;
    case 'T':
      v16 = *(_DWORD *)(a6 + 16);
      v17 = *(_QWORD *)(a6 + 24);
      *(_BYTE *)a1 = 1;
      *(_QWORD *)(a1 + 8) = a4;
      *(_DWORD *)(a1 + 16) = v16;
      *(_QWORD *)(a1 + 24) = v17;
      return a1;
    case 'V':
      if ( a5 - 10 <= 1 || a5 - 1 <= 1 )
      {
        sub_1022600(a1, a5, (_BYTE *)a4);
        return a1;
      }
      if ( a5 - 19 <= 1 && a8 )
      {
        sub_1022080(a1, a2, (__int64)a3, a4, a8);
        return a1;
      }
LABEL_4:
      if ( a5 - 17 <= 1 )
      {
        sub_1021F00(a1, a2, a3, a4, a6);
        return a1;
      }
      if ( a5 - 6 <= 3 )
        goto LABEL_6;
      if ( (a7 & 0xA) != 0xA )
      {
        v24 = a6;
        v18 = sub_920620(a4);
        a6 = v24;
        if ( !v18 || (v19 = sub_B451C0(a4), a6 = v24, !v19) || (v20 = sub_B451E0(a4), a6 = v24, !v20) )
        {
          if ( *(_BYTE *)a4 != 85 )
            goto LABEL_2;
          v21 = *(_QWORD *)(a4 - 32);
          if ( !v21 )
            goto LABEL_2;
          if ( (*(_BYTE *)v21 || *(_QWORD *)(v21 + 24) != *(_QWORD *)(a4 + 80) || *(_DWORD *)(v21 + 36) != 246)
            && (*(_BYTE *)v21 || *(_QWORD *)(v21 + 24) != *(_QWORD *)(a4 + 80) || *(_DWORD *)(v21 + 36) != 235) )
          {
            goto LABEL_36;
          }
          if ( a5 - 12 > 3 )
            goto LABEL_37;
LABEL_6:
          sub_1024090(a1, a4, a5, a6);
          return a1;
        }
      }
      if ( a5 - 12 <= 3 )
        goto LABEL_6;
      if ( *(_BYTE *)a4 != 85 )
        goto LABEL_2;
      v21 = *(_QWORD *)(a4 - 32);
LABEL_36:
      if ( !v21 )
      {
LABEL_2:
        *(_BYTE *)a1 = 0;
        *(_QWORD *)(a1 + 8) = a4;
        *(_DWORD *)(a1 + 16) = 0;
        *(_QWORD *)(a1 + 24) = 0;
        return a1;
      }
LABEL_37:
      if ( *(_BYTE *)v21
        || *(_QWORD *)(v21 + 24) != *(_QWORD *)(a4 + 80)
        || (*(_BYTE *)(v21 + 33) & 0x20) == 0
        || *(_DWORD *)(v21 + 36) != 174 )
      {
        goto LABEL_2;
      }
      v22 = sub_B451B0(a4);
      *(_QWORD *)(a1 + 8) = a4;
      *(_DWORD *)(a1 + 16) = 0;
      v12 = !v22;
      v23 = 0;
      if ( v12 )
        v23 = a4;
      *(_BYTE *)a1 = a5 == 16;
      *(_QWORD *)(a1 + 24) = v23;
      return a1;
    default:
      goto LABEL_2;
  }
}
