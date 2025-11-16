// Function: sub_1D14650
// Address: 0x1d14650
//
__int64 __fastcall sub_1D14650(__int64 a1, int a2, __int64 a3, __int64 a4)
{
  __int64 v4; // r13
  int v7; // eax
  unsigned int v8; // eax
  unsigned int v9; // eax
  unsigned __int64 v10; // rdx
  unsigned __int64 v11; // rdx
  unsigned int v12; // r15d
  bool v13; // al
  unsigned int v14; // r15d
  bool v15; // al
  unsigned int v16; // r15d
  bool v17; // al
  int v18; // eax
  unsigned int v19; // eax
  unsigned int v20; // r15d
  bool v21; // al
  unsigned __int64 v22; // rdx
  unsigned __int64 v23; // rdx
  unsigned __int64 v24; // [rsp+0h] [rbp-30h] BYREF
  unsigned int v25; // [rsp+8h] [rbp-28h]

  v4 = a3;
  switch ( a2 )
  {
    case '4':
      v25 = *(_DWORD *)(a3 + 8);
      if ( v25 > 0x40 )
        sub_16A4FD0((__int64)&v24, (const void **)a3);
      else
        v24 = *(_QWORD *)a3;
      sub_16A7200((__int64)&v24, (__int64 *)a4);
      goto LABEL_7;
    case '5':
      v25 = *(_DWORD *)(a3 + 8);
      if ( v25 > 0x40 )
        sub_16A4FD0((__int64)&v24, (const void **)a3);
      else
        v24 = *(_QWORD *)a3;
      sub_16A7590((__int64)&v24, (__int64 *)a4);
      goto LABEL_7;
    case '6':
      sub_16A7B50((__int64)&v24, a3, (__int64 *)a4);
      goto LABEL_7;
    case '7':
      v16 = *(_DWORD *)(a4 + 8);
      if ( v16 <= 0x40 )
        v17 = *(_QWORD *)a4 == 0;
      else
        v17 = v16 == (unsigned int)sub_16A57B0(a4);
      if ( v17 )
        goto LABEL_2;
      sub_16A9F90((__int64)&v24, v4, a4);
      goto LABEL_7;
    case '8':
      v12 = *(_DWORD *)(a4 + 8);
      if ( v12 <= 0x40 )
        v13 = *(_QWORD *)a4 == 0;
      else
        v13 = v12 == (unsigned int)sub_16A57B0(a4);
      if ( v13 )
        goto LABEL_2;
      sub_16A9D70((__int64)&v24, v4, a4);
      goto LABEL_7;
    case '9':
      v20 = *(_DWORD *)(a4 + 8);
      if ( v20 <= 0x40 )
        v21 = *(_QWORD *)a4 == 0;
      else
        v21 = v20 == (unsigned int)sub_16A57B0(a4);
      if ( v21 )
        goto LABEL_2;
      sub_16AB4D0((__int64)&v24, v4, a4);
      goto LABEL_7;
    case ':':
      v14 = *(_DWORD *)(a4 + 8);
      if ( v14 <= 0x40 )
        v15 = *(_QWORD *)a4 == 0;
      else
        v15 = v14 == (unsigned int)sub_16A57B0(a4);
      if ( v15 )
        goto LABEL_2;
      sub_16AB0A0((__int64)&v24, v4, a4);
      goto LABEL_7;
    case 'r':
      v18 = sub_16AEA10(a3, a4);
      goto LABEL_47;
    case 's':
      v7 = sub_16AEA10(a3, a4);
      goto LABEL_18;
    case 't':
      v18 = sub_16A9900(a3, (unsigned __int64 *)a4);
LABEL_47:
      if ( v18 > 0 )
        v4 = a4;
      v19 = *(_DWORD *)(v4 + 8);
      *(_DWORD *)(a1 + 8) = v19;
      if ( v19 > 0x40 )
        goto LABEL_50;
      goto LABEL_21;
    case 'u':
      v7 = sub_16A9900(a3, (unsigned __int64 *)a4);
LABEL_18:
      if ( v7 < 0 )
        v4 = a4;
      v8 = *(_DWORD *)(v4 + 8);
      *(_DWORD *)(a1 + 8) = v8;
      if ( v8 > 0x40 )
LABEL_50:
        sub_16A4FD0(a1, (const void **)v4);
      else
LABEL_21:
        *(_QWORD *)a1 = *(_QWORD *)v4;
      goto LABEL_8;
    case 'v':
      v9 = *(_DWORD *)(a3 + 8);
      v25 = v9;
      if ( v9 <= 0x40 )
      {
        v23 = *(_QWORD *)a3;
LABEL_60:
        v11 = *(_QWORD *)a4 & v23;
        goto LABEL_25;
      }
      sub_16A4FD0((__int64)&v24, (const void **)a3);
      v9 = v25;
      if ( v25 <= 0x40 )
      {
        v23 = v24;
        goto LABEL_60;
      }
      sub_16A8890((__int64 *)&v24, (__int64 *)a4);
      v9 = v25;
      v11 = v24;
      goto LABEL_25;
    case 'w':
      v9 = *(_DWORD *)(a3 + 8);
      v25 = v9;
      if ( v9 <= 0x40 )
      {
        v10 = *(_QWORD *)a3;
LABEL_24:
        v11 = *(_QWORD *)a4 | v10;
        goto LABEL_25;
      }
      sub_16A4FD0((__int64)&v24, (const void **)a3);
      v9 = v25;
      if ( v25 <= 0x40 )
      {
        v10 = v24;
        goto LABEL_24;
      }
      sub_16A89F0((__int64 *)&v24, (__int64 *)a4);
      v9 = v25;
      v11 = v24;
      goto LABEL_25;
    case 'x':
      v9 = *(_DWORD *)(a3 + 8);
      v25 = v9;
      if ( v9 <= 0x40 )
      {
        v22 = *(_QWORD *)a3;
LABEL_57:
        v11 = *(_QWORD *)a4 ^ v22;
        goto LABEL_25;
      }
      sub_16A4FD0((__int64)&v24, (const void **)a3);
      v9 = v25;
      if ( v25 <= 0x40 )
      {
        v22 = v24;
        goto LABEL_57;
      }
      sub_16A8F00((__int64 *)&v24, (__int64 *)a4);
      v9 = v25;
      v11 = v24;
LABEL_25:
      *(_DWORD *)(a1 + 8) = v9;
      *(_QWORD *)a1 = v11;
      *(_BYTE *)(a1 + 16) = 1;
      return a1;
    case 'z':
      v25 = *(_DWORD *)(a3 + 8);
      if ( v25 > 0x40 )
        sub_16A4FD0((__int64)&v24, (const void **)a3);
      else
        v24 = *(_QWORD *)a3;
      sub_16A7E20((__int64)&v24, a4);
      goto LABEL_7;
    case '{':
      v25 = *(_DWORD *)(a3 + 8);
      if ( v25 > 0x40 )
        sub_16A4FD0((__int64)&v24, (const void **)a3);
      else
        v24 = *(_QWORD *)a3;
      sub_16A6020((__int64)&v24, a4);
      goto LABEL_7;
    case '|':
      v25 = *(_DWORD *)(a3 + 8);
      if ( v25 > 0x40 )
        sub_16A4FD0((__int64)&v24, (const void **)a3);
      else
        v24 = *(_QWORD *)a3;
      sub_16A81B0((__int64)&v24, a4);
      goto LABEL_7;
    case '}':
      sub_16AB470((__int64)&v24, a3, a4);
      goto LABEL_7;
    case '~':
      sub_16AB4A0((__int64)&v24, a3, a4);
LABEL_7:
      *(_DWORD *)(a1 + 8) = v25;
      *(_QWORD *)a1 = v24;
LABEL_8:
      *(_BYTE *)(a1 + 16) = 1;
      return a1;
    default:
LABEL_2:
      *(_DWORD *)(a1 + 8) = 1;
      *(_QWORD *)a1 = 0;
      *(_BYTE *)(a1 + 16) = 0;
      return a1;
  }
}
