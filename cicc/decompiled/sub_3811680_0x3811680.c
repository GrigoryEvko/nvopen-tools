// Function: sub_3811680
// Address: 0x3811680
//
__int64 __fastcall sub_3811680(__int64 a1, unsigned __int64 a2, unsigned int a3, __m128i a4)
{
  __int64 v5; // rbx
  __int64 result; // rax
  __int64 v7; // rcx
  __int64 v8; // r8
  int v9; // eax
  unsigned __int64 v10; // rcx
  unsigned int v11; // edx
  __int64 v12; // r8
  unsigned __int8 *v13; // rax
  __int64 v14; // rdx
  unsigned int v15; // edx
  unsigned int v16; // edx
  unsigned int v17; // edx
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
  _QWORD *v32; // rdi
  unsigned int v33; // edx
  unsigned int v34; // r14d
  __int64 v35; // [rsp+8h] [rbp-158h]
  __int64 v36; // [rsp+120h] [rbp-40h] BYREF
  int v37; // [rsp+128h] [rbp-38h]

  v5 = a3;
  result = sub_3761870(
             (_QWORD *)a1,
             a2,
             *(_WORD *)(*(_QWORD *)(a2 + 48) + 16LL * a3),
             *(_QWORD *)(*(_QWORD *)(a2 + 48) + 16LL * a3 + 8),
             1);
  if ( !(_BYTE)result )
  {
    v9 = *(_DWORD *)(a2 + 24);
    if ( v9 > 298 )
    {
      switch ( v9 )
      {
        case 335:
          result = (__int64)sub_3805370(a1, a2, a4);
          v10 = result;
          v12 = v18;
          goto LABEL_7;
        case 338:
          result = (__int64)sub_3805750(a1, a2);
          v10 = result;
          v12 = v17;
          goto LABEL_7;
        case 342:
          result = (__int64)sub_3805050((__int64 *)a1, a2, a4);
          v10 = result;
          v12 = v19;
          goto LABEL_7;
        case 374:
        case 375:
          v13 = sub_346AFF0(a4, *(_QWORD *)a1, a2, *(_QWORD **)(a1 + 8), v7, v8);
          return sub_3760E70(a1, a2, 0, (unsigned __int64)v13, v14);
        case 376:
        case 377:
        case 378:
        case 379:
        case 380:
        case 381:
          v13 = sub_346A7D0(*(_QWORD *)a1, a2, *(_QWORD **)(a1 + 8), a4);
          return sub_3760E70(a1, a2, 0, (unsigned __int64)v13, v14);
        default:
          goto LABEL_11;
      }
    }
    if ( v9 > 50 )
    {
      switch ( v9 )
      {
        case 51:
          v32 = *(_QWORD **)(a1 + 8);
          v36 = 0;
          v37 = 0;
          result = (__int64)sub_33F17F0(v32, 51, (__int64)&v36, 6u, 0);
          v34 = v33;
          if ( v36 )
          {
            v35 = result;
            sub_B91220((__int64)&v36, v36);
            result = v35;
          }
          v10 = result;
          v12 = v34;
          goto LABEL_7;
        case 52:
        case 154:
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
          result = (__int64)sub_3810F90((__int64 *)a1, a2, a4);
          v10 = result;
          v12 = v15;
          goto LABEL_7;
        case 96:
        case 97:
        case 98:
        case 99:
        case 100:
        case 257:
        case 260:
        case 279:
        case 280:
        case 283:
        case 284:
        case 285:
        case 286:
          result = (__int64)sub_3811390((__int64 *)a1, a2, a4);
          v10 = result;
          v12 = v16;
          goto LABEL_7;
        case 143:
        case 144:
        case 220:
        case 221:
          result = (__int64)sub_3805880((__int64 *)a1, a2, a4);
          v10 = result;
          v12 = v20;
          goto LABEL_7;
        case 145:
        case 230:
          result = (__int64)sub_380A560((__int64 *)a1, a2);
          v10 = result;
          v12 = v24;
          goto LABEL_7;
        case 150:
        case 151:
          result = (__int64)sub_380FD60((__int64 *)a1, a2, a4);
          v10 = result;
          v12 = v23;
          goto LABEL_7;
        case 152:
          result = (__int64)sub_380F530((__int64 *)a1, a2, a4);
          v10 = result;
          v12 = v31;
          goto LABEL_7;
        case 158:
          result = (__int64)sub_38054F0(a1, a2, a4);
          v10 = result;
          v12 = v25;
          goto LABEL_7;
        case 205:
          result = sub_3810CE0(a1, a2);
          v10 = result;
          v12 = v26;
          goto LABEL_7;
        case 207:
          result = (__int64)sub_3810E70(a1, a2);
          v10 = result;
          v12 = v27;
          goto LABEL_7;
        case 234:
          result = (__int64)sub_375A6A0(a1, **(_QWORD **)(a2 + 40), *(_QWORD *)(*(_QWORD *)(a2 + 40) + 8LL), a4);
          v10 = result;
          v12 = v28;
          goto LABEL_7;
        case 258:
        case 259:
          result = (__int64)sub_38100F0((__int64 *)a1, a2);
          v10 = result;
          v12 = v22;
          goto LABEL_7;
        case 261:
          result = (__int64)sub_3810510((__int64 *)a1, a2, a4);
          v10 = result;
          v12 = v29;
          goto LABEL_7;
        case 287:
        case 288:
        case 289:
          result = sub_38107B0((__int64 *)a1, a2, a4);
          v10 = result;
          v12 = v21;
          goto LABEL_7;
        case 298:
          result = (__int64)sub_3805620(a1, a2);
          v10 = result;
          v12 = v30;
          goto LABEL_7;
        default:
          goto LABEL_11;
      }
    }
    if ( v9 != 12 )
LABEL_11:
      sub_C64ED0("Do not know how to soft promote this operator's result!", 1u);
    result = (__int64)sub_3805420(a1, a2, a4);
    v10 = result;
    v12 = v11;
LABEL_7:
    if ( v10 )
      return sub_375F970(a1, a2, v5, v10, v12);
  }
  return result;
}
