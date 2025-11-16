// Function: sub_38B1900
// Address: 0x38b1900
//
__int64 __fastcall sub_38B1900(__int64 a1, __int64 *a2, __int64 a3, __int64 *a4, double a5, __m128i a6, double a7)
{
  __int64 v8; // r13
  int v9; // ebx
  unsigned __int64 v10; // rsi
  const char *v11; // rax
  __int64 result; // rax
  int v14; // eax
  _QWORD *v15; // rax
  _QWORD *v16; // rbx
  int v17; // ecx
  char v18; // r8
  char v19; // r8
  int v20; // eax
  int v21; // eax
  int v22; // [rsp+Ch] [rbp-64h]
  int v24; // [rsp+18h] [rbp-58h]
  const char *v25; // [rsp+20h] [rbp-50h] BYREF
  char v26; // [rsp+30h] [rbp-40h]
  char v27; // [rsp+31h] [rbp-3Fh]

  v8 = a1 + 8;
  v9 = *(_DWORD *)(a1 + 64);
  v10 = *(_QWORD *)(a1 + 56);
  if ( v9 )
  {
    v22 = *(_DWORD *)(a1 + 104);
    v14 = sub_3887100(a1 + 8);
    *(_DWORD *)(a1 + 64) = v14;
    switch ( v9 )
    {
      case 57:
        v17 = 1;
        return (unsigned __int8)sub_38B09F0(a1, a2, a4, v17, a5, a6, a7);
      case 58:
        v17 = 2;
        return (unsigned __int8)sub_38B09F0(a1, a2, a4, v17, a5, a6, a7);
      case 59:
        v17 = 3;
        return (unsigned __int8)sub_38B09F0(a1, a2, a4, v17, a5, a6, a7);
      case 233:
      case 235:
      case 237:
      case 245:
        if ( v14 == 83 )
        {
          v20 = sub_3887100(v8);
          *(_DWORD *)(a1 + 64) = v20;
          if ( v20 != 84 )
          {
            if ( !(unsigned __int8)sub_38AEF70(a1, a2, a4, v22, 1, a5, *(double *)a6.m128i_i64, a7) )
            {
              sub_15F2310(*a2, 1);
              return 0;
            }
            return 1;
          }
        }
        else
        {
          if ( v14 != 84 )
            return (unsigned __int8)sub_38AEF70(a1, a2, a4, v22, 1, a5, *(double *)a6.m128i_i64, a7) != 0;
          v21 = sub_3887100(v8);
          *(_DWORD *)(a1 + 64) = v21;
          if ( v21 != 83 )
          {
            if ( !(unsigned __int8)sub_38AEF70(a1, a2, a4, v22, 1, a5, *(double *)a6.m128i_i64, a7) )
              goto LABEL_86;
            return 1;
          }
        }
        *(_DWORD *)(a1 + 64) = sub_3887100(v8);
        if ( !(unsigned __int8)sub_38AEF70(a1, a2, a4, v22, 1, a5, *(double *)a6.m128i_i64, a7) )
        {
          sub_15F2310(*a2, 1);
LABEL_86:
          sub_15F2330(*a2, 1);
          return 0;
        }
        return 1;
      case 234:
      case 236:
      case 238:
      case 241:
      case 244:
        v24 = 0;
        while ( 2 )
        {
          switch ( v14 )
          {
            case 'K':
              v24 |= 2u;
              *(_DWORD *)(a1 + 64) = sub_3887100(v8);
              goto LABEL_11;
            case 'L':
              v24 |= 4u;
              *(_DWORD *)(a1 + 64) = sub_3887100(v8);
              goto LABEL_11;
            case 'M':
              v24 |= 8u;
              *(_DWORD *)(a1 + 64) = sub_3887100(v8);
              goto LABEL_11;
            case 'N':
              v24 |= 0x10u;
              *(_DWORD *)(a1 + 64) = sub_3887100(v8);
              goto LABEL_11;
            case 'O':
              v24 |= 0x20u;
              *(_DWORD *)(a1 + 64) = sub_3887100(v8);
              goto LABEL_11;
            case 'P':
              v24 |= 1u;
              *(_DWORD *)(a1 + 64) = sub_3887100(v8);
              goto LABEL_11;
            case 'Q':
              v24 |= 0x40u;
              *(_DWORD *)(a1 + 64) = sub_3887100(v8);
              goto LABEL_11;
            case 'R':
              v24 = -1;
              *(_DWORD *)(a1 + 64) = sub_3887100(v8);
LABEL_11:
              v14 = *(_DWORD *)(a1 + 64);
              continue;
            default:
              v19 = sub_38AEF70(a1, a2, a4, v22, 2, a5, *(double *)a6.m128i_i64, a7);
              result = 1;
              if ( !v19 )
                goto LABEL_74;
              return result;
          }
        }
      case 239:
      case 240:
      case 246:
      case 247:
        if ( v14 != 85 )
          return (unsigned __int8)sub_38AEF70(a1, a2, a4, v22, 1, a5, *(double *)a6.m128i_i64, a7);
        *(_DWORD *)(a1 + 64) = sub_3887100(v8);
        if ( (unsigned __int8)sub_38AEF70(a1, a2, a4, v22, 1, a5, *(double *)a6.m128i_i64, a7) )
          return 1;
        sub_15F2350(*a2, 1);
        return 0;
      case 242:
      case 243:
        return (unsigned __int8)sub_38AEF70(a1, a2, a4, v22, 1, a5, *(double *)a6.m128i_i64, a7);
      case 248:
      case 249:
      case 250:
        return (unsigned __int8)sub_38AF0C0(a1, a2, a4, v22, a5, *(double *)a6.m128i_i64, a7);
      case 251:
        return (unsigned __int8)sub_38AF1C0(a1, a2, a4, v22, a5, *(double *)a6.m128i_i64, a7);
      case 252:
        v24 = 0;
        while ( 2 )
        {
          switch ( v14 )
          {
            case 'K':
              v24 |= 2u;
              *(_DWORD *)(a1 + 64) = sub_3887100(v8);
              goto LABEL_65;
            case 'L':
              v24 |= 4u;
              *(_DWORD *)(a1 + 64) = sub_3887100(v8);
              goto LABEL_65;
            case 'M':
              v24 |= 8u;
              *(_DWORD *)(a1 + 64) = sub_3887100(v8);
              goto LABEL_65;
            case 'N':
              v24 |= 0x10u;
              *(_DWORD *)(a1 + 64) = sub_3887100(v8);
              goto LABEL_65;
            case 'O':
              v24 |= 0x20u;
              *(_DWORD *)(a1 + 64) = sub_3887100(v8);
              goto LABEL_65;
            case 'P':
              v24 |= 1u;
              *(_DWORD *)(a1 + 64) = sub_3887100(v8);
              goto LABEL_65;
            case 'Q':
              v24 |= 0x40u;
              *(_DWORD *)(a1 + 64) = sub_3887100(v8);
              goto LABEL_65;
            case 'R':
              v24 = -1;
              *(_DWORD *)(a1 + 64) = sub_3887100(v8);
LABEL_65:
              v14 = *(_DWORD *)(a1 + 64);
              continue;
            default:
              v18 = sub_38AF1C0(a1, a2, a4, v22, a5, *(double *)a6.m128i_i64, a7);
              result = 1;
              if ( !v18 )
              {
LABEL_74:
                result = 0;
                if ( v24 )
                {
                  sub_15F2440(*a2, v24);
                  return 0;
                }
              }
              return result;
          }
        }
      case 253:
        return sub_38A1B50((__int64 **)a1, a2, a4, a5, *(double *)a6.m128i_i64, a7);
      case 254:
        v17 = 0;
        return (unsigned __int8)sub_38B09F0(a1, a2, a4, v17, a5, a6, a7);
      case 255:
      case 256:
      case 257:
      case 258:
      case 259:
      case 260:
      case 261:
      case 262:
      case 263:
      case 264:
      case 265:
      case 266:
      case 267:
        return (unsigned __int8)sub_38ABD10(a1, a2, a4, v22, a5, *(double *)a6.m128i_i64, a7);
      case 268:
        return (unsigned __int8)sub_38AE200(a1, a2, a4, a5, *(double *)a6.m128i_i64, a7);
      case 269:
        return (unsigned __int8)sub_38AE490(a1, a2, a4, a5, *(double *)a6.m128i_i64, a7);
      case 270:
        return (unsigned __int8)sub_38AC1E0(a1, a2, a4, a5, *(double *)a6.m128i_i64, a7);
      case 275:
        return (unsigned __int8)sub_38A16D0(a1, a2, a5, *(double *)a6.m128i_i64, a7, a3, a4);
      case 276:
        return (unsigned __int8)sub_38AB370((__int64 **)a1, a2, a4, a5, *(double *)a6.m128i_i64, a7);
      case 277:
        return (unsigned __int8)sub_38AB500(a1, a2, a4, a5, *(double *)a6.m128i_i64, a7);
      case 278:
        return (unsigned __int8)sub_38AB950(a1, a2, a4, a5, *(double *)a6.m128i_i64, a7);
      case 279:
        return (unsigned __int8)sub_38AFC80(a1, a2, a4, a5, a6, a7);
      case 280:
        return (unsigned __int8)sub_38ABC90((__int64 **)a1, a2, a4, a5, *(double *)a6.m128i_i64, a7);
      case 281:
        v15 = sub_1648A60(56, 0);
        v16 = v15;
        if ( v15 )
          sub_15F82A0((__int64)v15, *(_QWORD *)a1, 0);
        *a2 = (__int64)v16;
        return 0;
      case 282:
        return (unsigned __int8)sub_38AE9B0((__int64 **)a1, a2, a4, a5, *(double *)a6.m128i_i64, a7);
      case 283:
        return (unsigned __int8)sub_38AEF10(a1, a2, a4, a5, *(double *)a6.m128i_i64, a7);
      case 284:
        return (unsigned __int8)sub_38AEAF0((__int64 **)a1, a2, a4, a5, *(double *)a6.m128i_i64, a7);
      case 285:
        return (unsigned __int8)sub_38AF940((__int64 **)a1, a2, a4, a5, *(double *)a6.m128i_i64, a7);
      case 286:
        return (unsigned __int8)sub_38AFAD0((__int64 **)a1, a2, a4, a5, *(double *)a6.m128i_i64, a7);
      case 287:
        return sub_38AC420(a1, a2, a4, a5, *(double *)a6.m128i_i64, a7);
      case 288:
        return sub_38AC790(a1, a2, a4, a5, *(double *)a6.m128i_i64, a7);
      case 289:
        return sub_38ACA10(a1, a2, a4, a5, *(double *)a6.m128i_i64, a7);
      case 290:
        return sub_388EAE0((__int64 *)a1, a2);
      case 291:
        return sub_38ACC60(a1, a2, a4, a5, *(double *)a6.m128i_i64, a7);
      case 292:
        return sub_38ACF80(a1, a2, a4, a5, *(double *)a6.m128i_i64, a7);
      case 293:
        return sub_38AD2A0(a1, a2, a4, a5, *(double *)a6.m128i_i64, a7);
      case 294:
        return (unsigned __int8)sub_38AE640(a1, a2, a4, a5, *(double *)a6.m128i_i64, a7);
      case 295:
        return (unsigned __int8)sub_38AE740(a1, a2, a4, a5, *(double *)a6.m128i_i64, a7);
      case 296:
        return (unsigned __int8)sub_38AE880(a1, a2, a4, a5, *(double *)a6.m128i_i64, a7);
      case 297:
        return sub_38ADA40(a1, a2, a4, a5, *(double *)a6.m128i_i64, a7);
      case 298:
        return sub_38ADC60(a1, a2, a4, a5, *(double *)a6.m128i_i64, a7);
      default:
        v27 = 1;
        v11 = "expected instruction opcode";
        goto LABEL_3;
    }
  }
  v27 = 1;
  v11 = "found end of file when expecting more instructions";
LABEL_3:
  v25 = v11;
  v26 = 3;
  return (unsigned __int8)sub_38814C0(v8, v10, (__int64)&v25);
}
