// Function: sub_380ED90
// Address: 0x380ed90
//
__int64 __fastcall sub_380ED90(__int64 a1, unsigned __int64 a2, unsigned int a3, __m128i a4)
{
  __int64 v4; // rbx
  __int64 result; // rax
  __int64 v6; // rdx
  __int64 v7; // rcx
  __int64 v8; // r8
  int v9; // eax
  unsigned __int64 v10; // rcx
  unsigned int v11; // edx
  __int64 v12; // r8
  unsigned int v13; // edx
  unsigned int v14; // edx
  unsigned int v15; // edx
  unsigned __int8 *v16; // rax
  __int64 v17; // rdx
  unsigned int v18; // edx
  unsigned int v19; // edx
  unsigned int v20; // edx
  unsigned int v21; // edx
  unsigned int v22; // edx
  unsigned int v23; // edx
  unsigned int v24; // edx
  unsigned int v25; // edx
  unsigned int v26; // edx
  unsigned int v27; // edx
  unsigned int v28; // edx
  unsigned int v29; // edx
  unsigned int v30; // edx
  unsigned int v31; // edx
  unsigned int v32; // edx

  v4 = a3;
  result = sub_3761870(
             (_QWORD *)a1,
             a2,
             *(_WORD *)(*(_QWORD *)(a2 + 48) + 16LL * a3),
             *(_QWORD *)(*(_QWORD *)(a2 + 48) + 16LL * a3 + 8),
             1);
  if ( !(_BYTE)result )
  {
    v9 = *(_DWORD *)(a2 + 24);
    if ( v9 <= 381 )
    {
      if ( v9 > 204 )
      {
        switch ( v9 )
        {
          case 205:
            result = sub_380E320(a1, a2);
            v10 = result;
            v12 = v31;
            goto LABEL_11;
          case 207:
            result = (__int64)sub_380E420(a1, a2);
            v10 = result;
            v12 = v32;
            goto LABEL_11;
          case 220:
          case 221:
            result = (__int64)sub_3804DF0((__int64 *)a1, a2, a4);
            v10 = result;
            v12 = v24;
            goto LABEL_11;
          case 230:
            result = (__int64)sub_3803D50((__int64 *)a1, a2, a4);
            v10 = result;
            v12 = v25;
            goto LABEL_11;
          case 234:
            result = (__int64)sub_38036B0((__int64 *)a1, a2, a4);
            v10 = result;
            v12 = v26;
            goto LABEL_11;
          case 244:
          case 245:
          case 246:
          case 247:
          case 248:
          case 249:
          case 250:
          case 251:
          case 252:
          case 253:
          case 254:
          case 255:
          case 256:
          case 262:
          case 263:
          case 264:
          case 265:
          case 266:
          case 267:
          case 268:
          case 269:
          case 270:
          case 271:
          case 272:
          case 273:
          case 274:
            goto LABEL_16;
          case 257:
          case 260:
          case 279:
          case 280:
          case 281:
          case 282:
          case 283:
          case 284:
          case 285:
          case 286:
            goto LABEL_17;
          case 258:
          case 259:
            result = (__int64)sub_380DEC0((__int64 *)a1, a2);
            v10 = result;
            v12 = v23;
            goto LABEL_11;
          case 261:
            result = (__int64)sub_380E010((__int64 *)a1, a2);
            v10 = result;
            v12 = v27;
            goto LABEL_11;
          case 287:
          case 288:
          case 289:
            result = sub_380E180((__int64 *)a1, a2);
            v10 = result;
            v12 = v19;
            goto LABEL_11;
          case 298:
            result = (__int64)sub_3804540((__int64 *)a1, a2);
            v10 = result;
            v12 = v28;
            goto LABEL_11;
          case 338:
            result = (__int64)sub_3804970((__int64 *)a1, a2);
            v10 = result;
            v12 = v29;
            goto LABEL_11;
          case 342:
            result = (__int64)sub_3805050((__int64 *)a1, a2, a4);
            v10 = result;
            v12 = v30;
            goto LABEL_11;
          case 374:
          case 375:
            v16 = sub_346AFF0(a4, *(_QWORD *)a1, a2, *(_QWORD **)(a1 + 8), v7, v8);
            return sub_3760E70(a1, a2, 0, (unsigned __int64)v16, v17);
          case 376:
          case 377:
          case 378:
          case 379:
          case 380:
          case 381:
            v16 = sub_346A7D0(*(_QWORD *)a1, a2, *(_QWORD **)(a1 + 8), a4);
            return sub_3760E70(a1, a2, 0, (unsigned __int64)v16, v17);
          default:
            goto LABEL_15;
        }
      }
      if ( v9 <= 158 )
      {
        if ( v9 > 95 )
        {
          switch ( v9 )
          {
            case 96:
            case 97:
            case 98:
            case 99:
            case 100:
LABEL_17:
              result = (__int64)sub_380DBE0((__int64 *)a1, a2, a4);
              v10 = result;
              v12 = v15;
              goto LABEL_11;
            case 145:
              result = (__int64)sub_38040F0((__int64 *)a1, a2);
              v10 = result;
              v12 = v22;
              goto LABEL_11;
            case 150:
            case 151:
              result = sub_380DD40((__int64 *)a1, a2);
              v10 = result;
              v12 = v18;
              goto LABEL_11;
            case 152:
              result = (__int64)sub_380D970((__int64 *)a1, a2);
              v10 = result;
              v12 = v20;
              goto LABEL_11;
            case 154:
LABEL_16:
              result = (__int64)sub_380DAC0((__int64 *)a1, a2, a4);
              v10 = result;
              v12 = v14;
              goto LABEL_11;
            case 158:
              result = (__int64)sub_380E540(a1, a2, a4, v6, v7);
              v10 = result;
              v12 = v21;
              goto LABEL_11;
            default:
              goto LABEL_15;
          }
        }
        if ( v9 == 12 )
        {
          result = (__int64)sub_38039F0((__int64 *)a1, a2, a4);
          v10 = result;
          v12 = v13;
LABEL_11:
          if ( v10 )
            return sub_375F650(a1, a2, v4, v10, v12);
          return result;
        }
        if ( v9 == 51 )
        {
          result = (__int64)sub_3804F80((__int64 *)a1, a2);
          v10 = result;
          v12 = v11;
          goto LABEL_11;
        }
      }
    }
LABEL_15:
    sub_C64ED0("Do not know how to promote this operator's result!", 1u);
  }
  return result;
}
