// Function: sub_2035AA0
// Address: 0x2035aa0
//
__int64 __fastcall sub_2035AA0(__int64 a1, __int64 a2)
{
  __int64 v2; // rbx
  __int64 v3; // r8
  __int64 v5; // rsi
  _QWORD *v6; // r13
  __int64 v7; // r9
  __int16 v8; // r14
  int v9; // eax
  __int64 v10; // r8
  char v11; // al
  __int64 v12; // rax
  __int64 v13; // rdx
  __int64 v14; // rax
  __int64 v15; // rdx
  __int64 v16; // r14
  __int64 v18; // r14
  unsigned int v19; // ebx
  __int64 v20; // rax
  __int64 v21; // rdx
  __int64 v22; // [rsp+8h] [rbp-98h]
  __int64 v23; // [rsp+18h] [rbp-88h]
  __int64 v24; // [rsp+20h] [rbp-80h]
  __int64 v25; // [rsp+20h] [rbp-80h]
  unsigned int v26; // [rsp+20h] [rbp-80h]
  int v27; // [rsp+28h] [rbp-78h]
  __int64 v28; // [rsp+28h] [rbp-78h]
  __int64 v29; // [rsp+30h] [rbp-70h] BYREF
  int v30; // [rsp+38h] [rbp-68h]
  _BYTE v31[8]; // [rsp+40h] [rbp-60h] BYREF
  __int64 v32; // [rsp+48h] [rbp-58h]
  __m128i v33; // [rsp+50h] [rbp-50h] BYREF
  __int64 v34; // [rsp+60h] [rbp-40h]

  v3 = a1;
  v5 = *(_QWORD *)(a2 + 72);
  v29 = v5;
  if ( v5 )
  {
    sub_1623A60((__int64)&v29, v5, 2);
    v3 = a1;
  }
  v6 = *(_QWORD **)(v3 + 8);
  v7 = *(_QWORD *)(a2 + 104);
  v30 = *(_DWORD *)(a2 + 64);
  if ( (*(_BYTE *)(a2 + 27) & 4) != 0 )
  {
    v24 = v3;
    v33 = _mm_loadu_si128((const __m128i *)(v7 + 40));
    v34 = *(_QWORD *)(v7 + 56);
    v8 = *(_WORD *)(v7 + 32);
    v9 = sub_1E34390(v7);
    v10 = v24;
    v27 = v9;
    v11 = *(_BYTE *)(a2 + 88);
    v32 = *(_QWORD *)(a2 + 96);
    v31[0] = v11;
    if ( v11 )
    {
      switch ( v11 )
      {
        case 14:
        case 15:
        case 16:
        case 17:
        case 18:
        case 19:
        case 20:
        case 21:
        case 22:
        case 23:
        case 56:
        case 57:
        case 58:
        case 59:
        case 60:
        case 61:
          LOBYTE(v12) = 2;
          v13 = 0;
          break;
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
          LOBYTE(v12) = 3;
          v13 = 0;
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
          LOBYTE(v12) = 4;
          v13 = 0;
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
          LOBYTE(v12) = 5;
          v13 = 0;
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
          LOBYTE(v12) = 6;
          v13 = 0;
          break;
        case 55:
          LOBYTE(v12) = 7;
          v13 = 0;
          break;
        case 86:
        case 87:
        case 88:
        case 98:
        case 99:
        case 100:
          LOBYTE(v12) = 8;
          v13 = 0;
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
          LOBYTE(v12) = 9;
          v13 = 0;
          break;
        case 94:
        case 95:
        case 96:
        case 97:
        case 106:
        case 107:
        case 108:
        case 109:
          LOBYTE(v12) = 10;
          v13 = 0;
          break;
      }
    }
    else
    {
      LOBYTE(v12) = sub_1F596B0((__int64)v31);
      v10 = v24;
      v2 = v12;
    }
    LOBYTE(v2) = v12;
    v22 = v13;
    v25 = *(_QWORD *)(a2 + 32);
    v23 = *(_QWORD *)(a2 + 104);
    v14 = sub_2032580(v10, *(_QWORD *)(v25 + 40), *(_QWORD *)(v25 + 48));
    v16 = sub_1D2C750(
            v6,
            **(_QWORD **)(a2 + 32),
            *(_QWORD *)(*(_QWORD *)(a2 + 32) + 8LL),
            (__int64)&v29,
            v14,
            v15,
            *(_QWORD *)(v25 + 80),
            *(_QWORD *)(v25 + 88),
            *(_OWORD *)v23,
            *(_QWORD *)(v23 + 16),
            v2,
            v22,
            v27,
            v8,
            (__int64)&v33);
  }
  else
  {
    v18 = *(_QWORD *)(a2 + 32);
    v28 = v7;
    v33 = _mm_loadu_si128((const __m128i *)(v7 + 40));
    v34 = *(_QWORD *)(v7 + 56);
    v26 = *(unsigned __int16 *)(v7 + 32);
    v19 = 1 << *(_WORD *)(v7 + 34);
    v20 = sub_2032580(v3, *(_QWORD *)(v18 + 40), *(_QWORD *)(v18 + 48));
    v16 = sub_1D2BF40(
            v6,
            **(_QWORD **)(a2 + 32),
            *(_QWORD *)(*(_QWORD *)(a2 + 32) + 8LL),
            (__int64)&v29,
            v20,
            v21,
            *(_QWORD *)(v18 + 80),
            *(_QWORD *)(v18 + 88),
            *(_OWORD *)v28,
            *(_QWORD *)(v28 + 16),
            v19 >> 1,
            v26,
            (__int64)&v33);
  }
  if ( v29 )
    sub_161E7C0((__int64)&v29, v29);
  return v16;
}
