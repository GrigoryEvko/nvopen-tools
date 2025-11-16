// Function: sub_1F596C0
// Address: 0x1f596c0
//
__int64 __fastcall sub_1F596C0(__int64 a1, char *a2)
{
  char v3; // di
  unsigned int v4; // eax
  unsigned __int64 v5; // rcx
  __int8 *v6; // r13
  __int64 v7; // r15
  const char *v8; // rax
  __m128i *v9; // rax
  __int64 v10; // rcx
  unsigned __int64 v12; // rax
  char *v13; // rdi
  char *v14; // rdi
  char v15; // al
  __int64 v16; // rdx
  unsigned __int64 v17; // rcx
  __int8 *v18; // r13
  size_t v19; // rbx
  _QWORD *v20; // rax
  __m128i *v21; // rax
  unsigned __int64 v22; // rax
  unsigned __int64 v23; // rdi
  unsigned __int64 v24; // rcx
  __m128i *v25; // rax
  __int64 v26; // rcx
  __m128i *v27; // rdx
  unsigned __int64 v28; // rax
  _QWORD *v29; // rdi
  char *v30; // [rsp+18h] [rbp-A8h] BYREF
  char v31[8]; // [rsp+20h] [rbp-A0h] BYREF
  __int64 v32; // [rsp+28h] [rbp-98h]
  _QWORD v33[2]; // [rsp+30h] [rbp-90h] BYREF
  _QWORD v34[2]; // [rsp+40h] [rbp-80h] BYREF
  __m128i *v35; // [rsp+50h] [rbp-70h] BYREF
  __int64 v36; // [rsp+58h] [rbp-68h]
  __m128i v37; // [rsp+60h] [rbp-60h] BYREF
  char *v38; // [rsp+70h] [rbp-50h] BYREF
  __int64 v39; // [rsp+78h] [rbp-48h]
  _QWORD v40[8]; // [rsp+80h] [rbp-40h] BYREF

  v3 = *a2;
  switch ( *a2 )
  {
    case 1:
      *(_QWORD *)(a1 + 8) = 2;
      *(_QWORD *)a1 = a1 + 16;
      strcpy((char *)(a1 + 16), "ch");
      return a1;
    case 2:
      *(_QWORD *)a1 = a1 + 16;
      strcpy((char *)(a1 + 16), "i1");
      *(_QWORD *)(a1 + 8) = 2;
      return a1;
    case 3:
      *(_QWORD *)(a1 + 8) = 2;
      *(_QWORD *)a1 = a1 + 16;
      strcpy((char *)(a1 + 16), "i8");
      return a1;
    case 4:
      *(_BYTE *)(a1 + 18) = 54;
      *(_QWORD *)a1 = a1 + 16;
      *(_WORD *)(a1 + 16) = 12649;
      *(_QWORD *)(a1 + 8) = 3;
      *(_BYTE *)(a1 + 19) = 0;
      return a1;
    case 5:
      *(_BYTE *)(a1 + 18) = 50;
      *(_QWORD *)a1 = a1 + 16;
      *(_WORD *)(a1 + 16) = 13161;
      *(_QWORD *)(a1 + 8) = 3;
      *(_BYTE *)(a1 + 19) = 0;
      return a1;
    case 6:
      *(_BYTE *)(a1 + 18) = 52;
      *(_QWORD *)a1 = a1 + 16;
      *(_WORD *)(a1 + 16) = 13929;
      *(_QWORD *)(a1 + 8) = 3;
      *(_BYTE *)(a1 + 19) = 0;
      return a1;
    case 7:
      *(_QWORD *)a1 = a1 + 16;
      strcpy((char *)(a1 + 16), "i128");
      *(_QWORD *)(a1 + 8) = 4;
      return a1;
    case 8:
      *(_BYTE *)(a1 + 18) = 54;
      *(_QWORD *)a1 = a1 + 16;
      *(_WORD *)(a1 + 16) = 12646;
      *(_QWORD *)(a1 + 8) = 3;
      *(_BYTE *)(a1 + 19) = 0;
      return a1;
    case 9:
      *(_BYTE *)(a1 + 18) = 50;
      *(_QWORD *)a1 = a1 + 16;
      *(_WORD *)(a1 + 16) = 13158;
      *(_QWORD *)(a1 + 8) = 3;
      *(_BYTE *)(a1 + 19) = 0;
      return a1;
    case 10:
      *(_BYTE *)(a1 + 18) = 52;
      *(_QWORD *)a1 = a1 + 16;
      *(_WORD *)(a1 + 16) = 13926;
      *(_QWORD *)(a1 + 8) = 3;
      *(_BYTE *)(a1 + 19) = 0;
      return a1;
    case 11:
      *(_BYTE *)(a1 + 18) = 48;
      *(_QWORD *)a1 = a1 + 16;
      *(_WORD *)(a1 + 16) = 14438;
      *(_QWORD *)(a1 + 8) = 3;
      *(_BYTE *)(a1 + 19) = 0;
      return a1;
    case 12:
      *(_QWORD *)a1 = a1 + 16;
      strcpy((char *)(a1 + 16), "f128");
      *(_QWORD *)(a1 + 8) = 4;
      return a1;
    case 13:
      *(_DWORD *)(a1 + 16) = 1717792880;
      *(_QWORD *)a1 = a1 + 16;
      *(_WORD *)(a1 + 20) = 12849;
      *(_BYTE *)(a1 + 22) = 56;
      *(_QWORD *)(a1 + 8) = 7;
      *(_BYTE *)(a1 + 23) = 0;
      return a1;
    case 14:
      *(_QWORD *)a1 = a1 + 16;
      strcpy((char *)(a1 + 16), "v1i1");
      *(_QWORD *)(a1 + 8) = 4;
      return a1;
    case 15:
      *(_QWORD *)a1 = a1 + 16;
      strcpy((char *)(a1 + 16), "v2i1");
      *(_QWORD *)(a1 + 8) = 4;
      return a1;
    case 16:
      *(_QWORD *)a1 = a1 + 16;
      strcpy((char *)(a1 + 16), "v4i1");
      *(_QWORD *)(a1 + 8) = 4;
      return a1;
    case 17:
      *(_QWORD *)a1 = a1 + 16;
      strcpy((char *)(a1 + 16), "v8i1");
      *(_QWORD *)(a1 + 8) = 4;
      return a1;
    case 18:
      *(_BYTE *)(a1 + 20) = 49;
      *(_QWORD *)a1 = a1 + 16;
      *(_DWORD *)(a1 + 16) = 1765159286;
      *(_QWORD *)(a1 + 8) = 5;
      *(_BYTE *)(a1 + 21) = 0;
      return a1;
    case 19:
      *(_BYTE *)(a1 + 20) = 49;
      *(_QWORD *)a1 = a1 + 16;
      *(_DWORD *)(a1 + 16) = 1764897654;
      *(_QWORD *)(a1 + 8) = 5;
      *(_BYTE *)(a1 + 21) = 0;
      return a1;
    case 20:
      *(_BYTE *)(a1 + 20) = 49;
      *(_QWORD *)a1 = a1 + 16;
      *(_DWORD *)(a1 + 16) = 1765029494;
      *(_QWORD *)(a1 + 8) = 5;
      *(_BYTE *)(a1 + 21) = 0;
      return a1;
    case 21:
      *(_QWORD *)a1 = a1 + 16;
      strcpy((char *)(a1 + 16), "v128i1");
      *(_QWORD *)(a1 + 8) = 6;
      return a1;
    case 22:
      *(_QWORD *)a1 = a1 + 16;
      strcpy((char *)(a1 + 16), "v512i1");
      *(_QWORD *)(a1 + 8) = 6;
      return a1;
    case 23:
      *(_BYTE *)(a1 + 22) = 49;
      *(_QWORD *)a1 = a1 + 16;
      *(_DWORD *)(a1 + 16) = 842019190;
      *(_WORD *)(a1 + 20) = 26932;
      *(_QWORD *)(a1 + 8) = 7;
      *(_BYTE *)(a1 + 23) = 0;
      return a1;
    case 24:
      *(_QWORD *)a1 = a1 + 16;
      strcpy((char *)(a1 + 16), "v1i8");
      *(_QWORD *)(a1 + 8) = 4;
      return a1;
    case 25:
      *(_QWORD *)a1 = a1 + 16;
      strcpy((char *)(a1 + 16), "v2i8");
      *(_QWORD *)(a1 + 8) = 4;
      return a1;
    case 26:
      *(_QWORD *)a1 = a1 + 16;
      strcpy((char *)(a1 + 16), "v4i8");
      *(_QWORD *)(a1 + 8) = 4;
      return a1;
    case 27:
      *(_QWORD *)a1 = a1 + 16;
      strcpy((char *)(a1 + 16), "v8i8");
      *(_QWORD *)(a1 + 8) = 4;
      return a1;
    case 28:
      *(_BYTE *)(a1 + 20) = 56;
      *(_QWORD *)a1 = a1 + 16;
      *(_DWORD *)(a1 + 16) = 1765159286;
      *(_QWORD *)(a1 + 8) = 5;
      *(_BYTE *)(a1 + 21) = 0;
      return a1;
    case 29:
      *(_BYTE *)(a1 + 20) = 56;
      *(_QWORD *)a1 = a1 + 16;
      *(_DWORD *)(a1 + 16) = 1764897654;
      *(_QWORD *)(a1 + 8) = 5;
      *(_BYTE *)(a1 + 21) = 0;
      return a1;
    case 30:
      *(_BYTE *)(a1 + 20) = 56;
      *(_QWORD *)a1 = a1 + 16;
      *(_DWORD *)(a1 + 16) = 1765029494;
      *(_QWORD *)(a1 + 8) = 5;
      *(_BYTE *)(a1 + 21) = 0;
      return a1;
    case 31:
      *(_QWORD *)a1 = a1 + 16;
      strcpy((char *)(a1 + 16), "v128i8");
      *(_QWORD *)(a1 + 8) = 6;
      return a1;
    case 32:
      *(_QWORD *)a1 = a1 + 16;
      strcpy((char *)(a1 + 16), "v256i8");
      *(_QWORD *)(a1 + 8) = 6;
      return a1;
    case 33:
      *(_BYTE *)(a1 + 20) = 54;
      *(_QWORD *)a1 = a1 + 16;
      *(_DWORD *)(a1 + 16) = 828977526;
      *(_QWORD *)(a1 + 8) = 5;
      *(_BYTE *)(a1 + 21) = 0;
      return a1;
    case 34:
      *(_BYTE *)(a1 + 20) = 54;
      *(_QWORD *)a1 = a1 + 16;
      *(_DWORD *)(a1 + 16) = 828977782;
      *(_QWORD *)(a1 + 8) = 5;
      *(_BYTE *)(a1 + 21) = 0;
      return a1;
    case 35:
      *(_BYTE *)(a1 + 20) = 54;
      *(_QWORD *)a1 = a1 + 16;
      *(_DWORD *)(a1 + 16) = 828978294;
      *(_QWORD *)(a1 + 8) = 5;
      *(_BYTE *)(a1 + 21) = 0;
      return a1;
    case 36:
      *(_BYTE *)(a1 + 20) = 54;
      *(_QWORD *)a1 = a1 + 16;
      *(_DWORD *)(a1 + 16) = 828979318;
      *(_QWORD *)(a1 + 8) = 5;
      *(_BYTE *)(a1 + 21) = 0;
      return a1;
    case 37:
      strcpy((char *)(a1 + 16), "v16i16");
      *(_QWORD *)a1 = a1 + 16;
      *(_QWORD *)(a1 + 8) = 6;
      return a1;
    case 38:
      strcpy((char *)(a1 + 16), "v32i16");
      *(_QWORD *)a1 = a1 + 16;
      *(_QWORD *)(a1 + 8) = 6;
      return a1;
    case 39:
      strcpy((char *)(a1 + 16), "v64i16");
      *(_QWORD *)a1 = a1 + 16;
      *(_QWORD *)(a1 + 8) = 6;
      return a1;
    case 40:
      *(_DWORD *)(a1 + 16) = 942813558;
      *(_QWORD *)a1 = a1 + 16;
      *(_WORD *)(a1 + 20) = 12649;
      *(_BYTE *)(a1 + 22) = 54;
      *(_QWORD *)(a1 + 8) = 7;
      *(_BYTE *)(a1 + 23) = 0;
      return a1;
    case 41:
      *(_BYTE *)(a1 + 20) = 50;
      *(_QWORD *)a1 = a1 + 16;
      *(_DWORD *)(a1 + 16) = 862531958;
      *(_QWORD *)(a1 + 8) = 5;
      *(_BYTE *)(a1 + 21) = 0;
      return a1;
    case 42:
      *(_BYTE *)(a1 + 20) = 50;
      *(_QWORD *)a1 = a1 + 16;
      *(_DWORD *)(a1 + 16) = 862532214;
      *(_QWORD *)(a1 + 8) = 5;
      *(_BYTE *)(a1 + 21) = 0;
      return a1;
    case 43:
      *(_BYTE *)(a1 + 20) = 50;
      *(_QWORD *)a1 = a1 + 16;
      *(_DWORD *)(a1 + 16) = 862532726;
      *(_QWORD *)(a1 + 8) = 5;
      *(_BYTE *)(a1 + 21) = 0;
      return a1;
    case 44:
      *(_BYTE *)(a1 + 20) = 50;
      *(_QWORD *)a1 = a1 + 16;
      *(_DWORD *)(a1 + 16) = 862533750;
      *(_QWORD *)(a1 + 8) = 5;
      *(_BYTE *)(a1 + 21) = 0;
      return a1;
    case 45:
      strcpy((char *)(a1 + 16), "v16i32");
      *(_QWORD *)a1 = a1 + 16;
      *(_QWORD *)(a1 + 8) = 6;
      return a1;
    case 46:
      strcpy((char *)(a1 + 16), "v32i32");
      *(_QWORD *)a1 = a1 + 16;
      *(_QWORD *)(a1 + 8) = 6;
      return a1;
    case 47:
      strcpy((char *)(a1 + 16), "v64i32");
      *(_QWORD *)a1 = a1 + 16;
      *(_QWORD *)(a1 + 8) = 6;
      return a1;
    case 48:
      *(_DWORD *)(a1 + 16) = 942813558;
      *(_QWORD *)a1 = a1 + 16;
      *(_WORD *)(a1 + 20) = 13161;
      *(_BYTE *)(a1 + 22) = 50;
      *(_QWORD *)(a1 + 8) = 7;
      *(_BYTE *)(a1 + 23) = 0;
      return a1;
    case 49:
      *(_BYTE *)(a1 + 20) = 52;
      *(_QWORD *)a1 = a1 + 16;
      *(_DWORD *)(a1 + 16) = 912863606;
      *(_QWORD *)(a1 + 8) = 5;
      *(_BYTE *)(a1 + 21) = 0;
      return a1;
    case 50:
      *(_BYTE *)(a1 + 20) = 52;
      *(_QWORD *)a1 = a1 + 16;
      *(_DWORD *)(a1 + 16) = 912863862;
      *(_QWORD *)(a1 + 8) = 5;
      *(_BYTE *)(a1 + 21) = 0;
      return a1;
    case 51:
      *(_BYTE *)(a1 + 20) = 52;
      *(_QWORD *)a1 = a1 + 16;
      *(_DWORD *)(a1 + 16) = 912864374;
      *(_QWORD *)(a1 + 8) = 5;
      *(_BYTE *)(a1 + 21) = 0;
      return a1;
    case 52:
      *(_BYTE *)(a1 + 20) = 52;
      *(_QWORD *)a1 = a1 + 16;
      *(_DWORD *)(a1 + 16) = 912865398;
      *(_QWORD *)(a1 + 8) = 5;
      *(_BYTE *)(a1 + 21) = 0;
      return a1;
    case 53:
      strcpy((char *)(a1 + 16), "v16i64");
      *(_QWORD *)a1 = a1 + 16;
      *(_QWORD *)(a1 + 8) = 6;
      return a1;
    case 54:
      strcpy((char *)(a1 + 16), "v32i64");
      *(_QWORD *)a1 = a1 + 16;
      *(_QWORD *)(a1 + 8) = 6;
      return a1;
    case 55:
      strcpy((char *)(a1 + 16), "v1i128");
      *(_QWORD *)a1 = a1 + 16;
      *(_QWORD *)(a1 + 8) = 6;
      return a1;
    case 86:
      *(_BYTE *)(a1 + 20) = 54;
      *(_QWORD *)a1 = a1 + 16;
      *(_DWORD *)(a1 + 16) = 828781174;
      *(_QWORD *)(a1 + 8) = 5;
      *(_BYTE *)(a1 + 21) = 0;
      return a1;
    case 87:
      *(_BYTE *)(a1 + 20) = 54;
      *(_QWORD *)a1 = a1 + 16;
      *(_DWORD *)(a1 + 16) = 828781686;
      *(_QWORD *)(a1 + 8) = 5;
      *(_BYTE *)(a1 + 21) = 0;
      return a1;
    case 88:
      *(_BYTE *)(a1 + 20) = 54;
      *(_QWORD *)a1 = a1 + 16;
      *(_DWORD *)(a1 + 16) = 828782710;
      *(_QWORD *)(a1 + 8) = 5;
      *(_BYTE *)(a1 + 21) = 0;
      return a1;
    case 89:
      *(_BYTE *)(a1 + 20) = 50;
      *(_QWORD *)a1 = a1 + 16;
      *(_DWORD *)(a1 + 16) = 862335350;
      *(_QWORD *)(a1 + 8) = 5;
      *(_BYTE *)(a1 + 21) = 0;
      return a1;
    case 90:
      *(_BYTE *)(a1 + 20) = 50;
      *(_QWORD *)a1 = a1 + 16;
      *(_DWORD *)(a1 + 16) = 862335606;
      *(_QWORD *)(a1 + 8) = 5;
      *(_BYTE *)(a1 + 21) = 0;
      return a1;
    case 91:
      *(_BYTE *)(a1 + 20) = 50;
      *(_QWORD *)a1 = a1 + 16;
      *(_DWORD *)(a1 + 16) = 862336118;
      *(_QWORD *)(a1 + 8) = 5;
      *(_BYTE *)(a1 + 21) = 0;
      return a1;
    case 92:
      *(_BYTE *)(a1 + 20) = 50;
      *(_QWORD *)a1 = a1 + 16;
      *(_DWORD *)(a1 + 16) = 862337142;
      *(_QWORD *)(a1 + 8) = 5;
      *(_BYTE *)(a1 + 21) = 0;
      return a1;
    case 93:
      strcpy((char *)(a1 + 16), "v16f32");
      *(_QWORD *)a1 = a1 + 16;
      *(_QWORD *)(a1 + 8) = 6;
      return a1;
    case 94:
      *(_BYTE *)(a1 + 20) = 52;
      *(_QWORD *)a1 = a1 + 16;
      *(_DWORD *)(a1 + 16) = 912666998;
      *(_QWORD *)(a1 + 8) = 5;
      *(_BYTE *)(a1 + 21) = 0;
      return a1;
    case 95:
      *(_BYTE *)(a1 + 20) = 52;
      *(_QWORD *)a1 = a1 + 16;
      *(_DWORD *)(a1 + 16) = 912667254;
      *(_QWORD *)(a1 + 8) = 5;
      *(_BYTE *)(a1 + 21) = 0;
      return a1;
    case 96:
      *(_BYTE *)(a1 + 20) = 52;
      *(_QWORD *)a1 = a1 + 16;
      *(_DWORD *)(a1 + 16) = 912667766;
      *(_QWORD *)(a1 + 8) = 5;
      *(_BYTE *)(a1 + 21) = 0;
      return a1;
    case 97:
      *(_BYTE *)(a1 + 20) = 52;
      *(_QWORD *)a1 = a1 + 16;
      *(_DWORD *)(a1 + 16) = 912668790;
      *(_QWORD *)(a1 + 8) = 5;
      *(_BYTE *)(a1 + 21) = 0;
      return a1;
    case 110:
      strcpy((char *)(a1 + 16), "x86mmx");
      *(_QWORD *)a1 = a1 + 16;
      *(_QWORD *)(a1 + 8) = 6;
      return a1;
    case 111:
      *(_QWORD *)a1 = a1 + 16;
      strcpy((char *)(a1 + 16), "glue");
      *(_QWORD *)(a1 + 8) = 4;
      return a1;
    case 112:
      strcpy((char *)(a1 + 16), "isVoid");
      *(_QWORD *)a1 = a1 + 16;
      *(_QWORD *)(a1 + 8) = 6;
      return a1;
    case 113:
      *(_BYTE *)(a1 + 22) = 100;
      *(_QWORD *)a1 = a1 + 16;
      *(_DWORD *)(a1 + 16) = 2037673557;
      *(_WORD *)(a1 + 20) = 25968;
      *(_QWORD *)(a1 + 8) = 7;
      *(_BYTE *)(a1 + 23) = 0;
      return a1;
    case 114:
      *(_QWORD *)a1 = a1 + 16;
      strcpy((char *)(a1 + 16), "ExceptRef");
      *(_QWORD *)(a1 + 8) = 9;
      return a1;
    case -7:
      *(_QWORD *)a1 = a1 + 16;
      strcpy((char *)(a1 + 16), "Metadata");
      *(_QWORD *)(a1 + 8) = 8;
      return a1;
    default:
      if ( v3 )
      {
        if ( (unsigned __int8)(v3 - 14) > 0x5Fu )
        {
          v4 = sub_1F58BF0(v3);
          goto LABEL_5;
        }
        switch ( v3 )
        {
          case 24:
          case 25:
          case 26:
          case 27:
          case 28:
          case 29:
          case 30:
          case 31:
          case 32:
          case 62:
          case 63:
          case 64:
          case 65:
          case 66:
          case 67:
            v15 = 3;
            break;
          case 33:
          case 34:
          case 35:
          case 36:
          case 37:
          case 38:
          case 39:
          case 40:
          case 68:
          case 69:
          case 70:
          case 71:
          case 72:
          case 73:
            v15 = 4;
            break;
          case 41:
          case 42:
          case 43:
          case 44:
          case 45:
          case 46:
          case 47:
          case 48:
          case 74:
          case 75:
          case 76:
          case 77:
          case 78:
          case 79:
            v15 = 5;
            break;
          case 49:
          case 50:
          case 51:
          case 52:
          case 53:
          case 54:
          case 80:
          case 81:
          case 82:
          case 83:
          case 84:
          case 85:
            v15 = 6;
            break;
          case 55:
            v15 = 7;
            break;
          case 86:
          case 87:
          case 88:
          case 98:
          case 99:
          case 100:
            v15 = 8;
            break;
          case 89:
          case 90:
          case 91:
          case 92:
          case 93:
          case 101:
          case 102:
          case 103:
          case 104:
          case 105:
            v15 = 9;
            break;
          case 94:
          case 95:
          case 96:
          case 97:
          case 106:
          case 107:
          case 108:
          case 109:
            v15 = 10;
            break;
          default:
            v15 = 2;
            break;
        }
        v16 = 0;
      }
      else
      {
        if ( !sub_1F58D20((__int64)a2) )
        {
          v4 = sub_1F58D40((__int64)a2);
LABEL_5:
          v5 = v4;
          if ( !v4 )
          {
            v37.m128i_i8[4] = 48;
            v6 = &v37.m128i_i8[4];
            v38 = (char *)v40;
LABEL_7:
            v7 = 1;
            LOBYTE(v40[0]) = *v6;
            v8 = (const char *)v40;
LABEL_8:
            v39 = v7;
            v8[v7] = 0;
            v9 = (__m128i *)sub_2241130(&v38, 0, 0, "i", 1);
            *(_QWORD *)a1 = a1 + 16;
            if ( (__m128i *)v9->m128i_i64[0] == &v9[1] )
            {
              *(__m128i *)(a1 + 16) = _mm_loadu_si128(v9 + 1);
            }
            else
            {
              *(_QWORD *)a1 = v9->m128i_i64[0];
              *(_QWORD *)(a1 + 16) = v9[1].m128i_i64[0];
            }
            v10 = v9->m128i_i64[1];
            v9->m128i_i64[0] = (__int64)v9[1].m128i_i64;
            v9->m128i_i64[1] = 0;
            *(_QWORD *)(a1 + 8) = v10;
            v9[1].m128i_i8[0] = 0;
            goto LABEL_11;
          }
          v6 = &v37.m128i_i8[5];
          do
          {
            *--v6 = v5 % 0xA + 48;
            v12 = v5;
            v5 /= 0xAu;
          }
          while ( v12 > 9 );
          v13 = (char *)(&v37.m128i_u8[5] - (unsigned __int8 *)v6);
          v38 = (char *)v40;
          v7 = &v37.m128i_u8[5] - (unsigned __int8 *)v6;
          v33[0] = &v37.m128i_u8[5] - (unsigned __int8 *)v6;
          if ( (unsigned __int64)(&v37.m128i_u8[5] - (unsigned __int8 *)v6) <= 0xF )
          {
            if ( v13 == (char *)1 )
              goto LABEL_7;
            if ( !v13 )
            {
              v8 = (const char *)v40;
              goto LABEL_8;
            }
            v14 = (char *)v40;
          }
          else
          {
            v38 = (char *)sub_22409D0(&v38, v33, 0);
            v14 = v38;
            v40[0] = v33[0];
          }
          memcpy(v14, v6, &v37.m128i_u8[5] - (unsigned __int8 *)v6);
          v7 = v33[0];
          v8 = v38;
          goto LABEL_8;
        }
        v15 = sub_1F596B0((__int64)a2);
      }
      v31[0] = v15;
      v32 = v16;
      sub_1F596C0(&v38, v31);
      if ( *a2 )
        v17 = word_42F4D80[(unsigned __int8)(*a2 - 14)];
      else
        v17 = (unsigned int)sub_1F58D30((__int64)a2);
      if ( !v17 )
      {
        v37.m128i_i8[4] = 48;
        v18 = &v37.m128i_i8[4];
        v33[0] = v34;
LABEL_101:
        v19 = 1;
        LOBYTE(v34[0]) = *v18;
        v20 = v34;
        goto LABEL_102;
      }
      v18 = &v37.m128i_i8[5];
      do
      {
        *--v18 = v17 % 0xA + 48;
        v28 = v17;
        v17 /= 0xAu;
      }
      while ( v28 > 9 );
      v19 = &v37.m128i_u8[5] - (unsigned __int8 *)v18;
      v33[0] = v34;
      v30 = (char *)(&v37.m128i_u8[5] - (unsigned __int8 *)v18);
      if ( (unsigned __int64)(&v37.m128i_u8[5] - (unsigned __int8 *)v18) > 0xF )
      {
        v33[0] = sub_22409D0(v33, &v30, 0);
        v29 = (_QWORD *)v33[0];
        v34[0] = v30;
LABEL_128:
        memcpy(v29, v18, v19);
        v19 = (size_t)v30;
        v20 = (_QWORD *)v33[0];
        goto LABEL_102;
      }
      if ( v19 == 1 )
        goto LABEL_101;
      if ( v19 )
      {
        v29 = v34;
        goto LABEL_128;
      }
      v20 = v34;
LABEL_102:
      v33[1] = v19;
      *((_BYTE *)v20 + v19) = 0;
      v21 = (__m128i *)sub_2241130(v33, 0, 0, "v", 1);
      v35 = &v37;
      if ( (__m128i *)v21->m128i_i64[0] == &v21[1] )
      {
        v37 = _mm_loadu_si128(v21 + 1);
      }
      else
      {
        v35 = (__m128i *)v21->m128i_i64[0];
        v37.m128i_i64[0] = v21[1].m128i_i64[0];
      }
      v36 = v21->m128i_i64[1];
      v21->m128i_i64[0] = (__int64)v21[1].m128i_i64;
      v21->m128i_i64[1] = 0;
      v21[1].m128i_i8[0] = 0;
      v22 = 15;
      v23 = 15;
      if ( v35 != &v37 )
        v23 = v37.m128i_i64[0];
      v24 = v36 + v39;
      if ( v36 + v39 <= v23 )
        goto LABEL_110;
      if ( v38 != (char *)v40 )
        v22 = v40[0];
      if ( v24 <= v22 )
      {
        v25 = (__m128i *)sub_2241400(&v38, 0, 0, v35, v36);
        *(_QWORD *)a1 = a1 + 16;
        v26 = v25->m128i_i64[0];
        v27 = v25 + 1;
        if ( (__m128i *)v25->m128i_i64[0] != &v25[1] )
          goto LABEL_111;
      }
      else
      {
LABEL_110:
        v25 = (__m128i *)sub_2241490(&v35, v38, v39, v24, v36);
        *(_QWORD *)a1 = a1 + 16;
        v26 = v25->m128i_i64[0];
        v27 = v25 + 1;
        if ( (__m128i *)v25->m128i_i64[0] != &v25[1] )
        {
LABEL_111:
          *(_QWORD *)a1 = v26;
          *(_QWORD *)(a1 + 16) = v25[1].m128i_i64[0];
          goto LABEL_112;
        }
      }
      *(__m128i *)(a1 + 16) = _mm_loadu_si128(v25 + 1);
LABEL_112:
      *(_QWORD *)(a1 + 8) = v25->m128i_i64[1];
      v25->m128i_i64[0] = (__int64)v27;
      v25->m128i_i64[1] = 0;
      v25[1].m128i_i8[0] = 0;
      if ( v35 != &v37 )
        j_j___libc_free_0(v35, v37.m128i_i64[0] + 1);
      if ( (_QWORD *)v33[0] != v34 )
        j_j___libc_free_0(v33[0], v34[0] + 1LL);
LABEL_11:
      if ( v38 != (char *)v40 )
        j_j___libc_free_0(v38, v40[0] + 1LL);
      return a1;
  }
}
