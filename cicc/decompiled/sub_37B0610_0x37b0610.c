// Function: sub_37B0610
// Address: 0x37b0610
//
__int64 __fastcall sub_37B0610(unsigned __int64 *a1, unsigned __int64 a2, unsigned int a3, __m128i a4)
{
  unsigned __int16 *v4; // rax
  __int64 result; // rax
  __int64 v6; // rdx
  __int64 v7; // rcx
  __int64 v8; // r8
  __int64 v9; // r9
  __int64 v10; // rsi
  int v11; // eax
  unsigned int v12; // edx
  unsigned int v13; // edx
  unsigned int v14; // edx
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
  unsigned int v32; // edx
  unsigned int v33; // edx
  unsigned int v34; // edx
  unsigned int v35; // edx
  unsigned int v36; // edx
  unsigned int v37; // edx
  unsigned int v38; // edx
  unsigned int v39; // edx
  unsigned int v40; // edx
  unsigned int v41; // edx
  unsigned int v42; // edx
  unsigned int v43; // edx
  unsigned __int64 v44; // rax
  __int64 v45; // rdx
  unsigned int v46; // edx
  unsigned int v47; // edx
  unsigned int v48; // edx
  unsigned int v49; // edx
  unsigned int v50; // edx
  unsigned int v51; // edx
  unsigned int v52; // edx
  unsigned int v53; // edx
  unsigned int v54; // [rsp+284h] [rbp-4Ch] BYREF
  unsigned __int64 v55; // [rsp+288h] [rbp-48h] BYREF
  unsigned __int64 v56; // [rsp+290h] [rbp-40h] BYREF
  __int64 v57; // [rsp+298h] [rbp-38h]
  unsigned __int64 *v58[5]; // [rsp+2A0h] [rbp-30h] BYREF

  v4 = (unsigned __int16 *)(*(_QWORD *)(a2 + 48) + 16LL * a3);
  v54 = a3;
  v55 = a2;
  result = sub_3762770((__int64)a1, a2, *v4);
  if ( !(_BYTE)result )
  {
    v10 = v55;
    v58[1] = a1;
    v58[0] = &v55;
    v58[2] = &v56;
    v58[3] = (unsigned __int64 *)&v54;
    v11 = *(_DWORD *)(v55 + 24);
    v56 = 0;
    LODWORD(v57) = 0;
    switch ( v11 )
    {
      case 4:
        v56 = (unsigned __int64)sub_3790EE0((__int64 *)a1, v55, a4);
        result = v49;
        LODWORD(v57) = v49;
        break;
      case 51:
        v56 = (unsigned __int64)sub_378F060((__int64 *)a1, v55);
        result = v48;
        LODWORD(v57) = v48;
        break;
      case 52:
      case 154:
      case 189:
      case 197:
      case 198:
      case 199:
      case 200:
      case 201:
      case 203:
      case 204:
      case 244:
      case 245:
      case 335:
      case 412:
      case 413:
      case 414:
      case 415:
      case 416:
      case 417:
      case 418:
      case 419:
      case 433:
      case 434:
      case 435:
      case 443:
      case 444:
      case 445:
      case 446:
      case 447:
      case 448:
      case 449:
        goto LABEL_11;
      case 55:
        v44 = sub_3761980((__int64)a1, v55, v54);
        v56 = sub_379AB60((__int64)a1, v44, v45);
        result = v46;
        LODWORD(v57) = v46;
        break;
      case 56:
      case 57:
      case 58:
      case 82:
      case 83:
      case 84:
      case 85:
      case 86:
      case 87:
      case 172:
      case 173:
      case 174:
      case 175:
      case 176:
      case 177:
      case 178:
      case 179:
      case 180:
      case 181:
      case 182:
      case 183:
      case 186:
      case 187:
      case 188:
      case 190:
      case 191:
      case 192:
      case 193:
      case 194:
      case 279:
      case 280:
      case 281:
      case 282:
      case 283:
      case 284:
      case 285:
      case 286:
      case 395:
      case 396:
      case 397:
      case 398:
      case 399:
      case 400:
      case 401:
      case 402:
      case 403:
      case 404:
      case 405:
      case 406:
      case 407:
      case 408:
      case 409:
      case 410:
      case 411:
      case 424:
      case 425:
      case 426:
      case 427:
      case 428:
      case 429:
      case 430:
      case 431:
      case 432:
      case 438:
      case 439:
      case 440:
      case 441:
      case 442:
        v56 = (unsigned __int64)sub_379BBB0((__int64 *)a1, v55, a4);
        result = v12;
        LODWORD(v57) = v12;
        break;
      case 59:
      case 60:
      case 61:
      case 62:
      case 96:
      case 97:
      case 98:
      case 99:
        goto LABEL_15;
      case 76:
      case 77:
      case 78:
      case 79:
      case 80:
      case 81:
        v56 = (unsigned __int64)sub_379E370((__int64)a1, v55, v54, a4);
        result = v17;
        LODWORD(v57) = v17;
        break;
      case 88:
      case 89:
      case 90:
      case 91:
        v56 = sub_379C210((__int64 *)a1, v55);
        result = v21;
        LODWORD(v57) = v21;
        break;
      case 100:
      case 257:
      case 260:
        result = sub_378DF80(v58, a4);
        if ( (_BYTE)result )
          break;
        v10 = v55;
LABEL_15:
        v56 = (unsigned __int64)sub_379C350((__int64 *)a1, v10);
        result = v16;
        LODWORD(v57) = v16;
        break;
      case 101:
      case 102:
      case 103:
      case 104:
      case 105:
      case 106:
      case 107:
      case 108:
      case 109:
      case 110:
      case 111:
      case 112:
      case 113:
      case 114:
      case 115:
      case 116:
      case 117:
      case 118:
      case 119:
      case 120:
      case 121:
      case 122:
      case 123:
      case 124:
      case 125:
      case 126:
      case 127:
      case 128:
      case 129:
      case 130:
      case 131:
      case 132:
      case 133:
      case 134:
      case 135:
      case 136:
      case 137:
      case 138:
      case 139:
      case 140:
      case 141:
      case 142:
      case 143:
      case 144:
      case 145:
      case 146:
      case 147:
      case 148:
        v56 = (unsigned __int64)sub_379D060((__int64 *)a1, v55, v6, v7, v8, v9, a4);
        result = v13;
        LODWORD(v57) = v13;
        break;
      case 149:
      case 155:
        v56 = (unsigned __int64)sub_379FB10((__int64 *)a1, v55, a4);
        result = v28;
        LODWORD(v57) = v28;
        break;
      case 150:
      case 195:
      case 196:
      case 422:
      case 423:
      case 436:
        v56 = sub_379B900((__int64 *)a1, v55);
        result = v18;
        LODWORD(v57) = v18;
        break;
      case 152:
        v56 = (unsigned __int64)sub_379CED0((__int64 *)a1, v55, a4);
        result = v35;
        LODWORD(v57) = v35;
        break;
      case 156:
        v56 = (unsigned __int64)sub_378E9D0((__int64 *)a1, v55);
        result = v34;
        LODWORD(v57) = v34;
        break;
      case 157:
        v56 = sub_37A2410((__int64)a1, v55);
        result = v51;
        LODWORD(v57) = v51;
        break;
      case 159:
        v56 = sub_37A0A30((__int64 *)a1, v55, a4);
        result = v50;
        LODWORD(v57) = v50;
        break;
      case 160:
        v56 = sub_37A1660((__int64 *)a1, v55);
        result = v31;
        LODWORD(v57) = v31;
        break;
      case 161:
        v56 = (unsigned __int64)sub_37A17C0((__int64 *)a1, v55, v6, v7, v8);
        result = v30;
        LODWORD(v57) = v30;
        break;
      case 164:
        v56 = (unsigned __int64)sub_37A4450((__int64 *)a1, v55, a4, v6, v7, v8);
        result = v33;
        LODWORD(v57) = v33;
        break;
      case 165:
        v56 = (unsigned __int64)sub_37A3FD0((__int64 *)a1, v55, a4);
        result = v32;
        LODWORD(v57) = v32;
        break;
      case 167:
      case 168:
      case 170:
      case 492:
        v56 = sub_378EED0((__int64 *)a1, v55, a4);
        result = v23;
        LODWORD(v57) = v23;
        break;
      case 171:
        v56 = sub_3791030((__int64 *)a1, v55, a4);
        result = v52;
        LODWORD(v57) = v52;
        break;
      case 184:
      case 185:
        v56 = (unsigned __int64)sub_379BE10((__int64 *)a1, v55);
        result = v29;
        LODWORD(v57) = v29;
        break;
      case 205:
      case 206:
      case 488:
      case 489:
        v56 = (unsigned __int64)sub_37A38B0((__int64 *)a1, v55);
        result = v22;
        LODWORD(v57) = v22;
        break;
      case 207:
        v56 = (unsigned __int64)sub_37A3EB0((__int64)a1, v55);
        result = v47;
        LODWORD(v57) = v47;
        break;
      case 208:
      case 463:
        v56 = (unsigned __int64)sub_37A4C40((__int64 *)a1, v55);
        result = v27;
        LODWORD(v57) = v27;
        break;
      case 213:
      case 214:
      case 215:
      case 216:
      case 220:
      case 221:
      case 226:
      case 227:
      case 230:
      case 233:
      case 452:
      case 453:
      case 454:
      case 455:
      case 456:
      case 457:
      case 458:
      case 459:
      case 460:
        v56 = sub_37AF6D0((__int64 *)a1, v55);
        result = v15;
        LODWORD(v57) = v15;
        break;
      case 222:
        v56 = (unsigned __int64)sub_37A0290((__int64 *)a1, v55);
        result = v53;
        LODWORD(v57) = v53;
        break;
      case 223:
      case 224:
      case 225:
        v56 = (unsigned __int64)sub_379F210((__int64 *)a1, v55);
        result = v24;
        LODWORD(v57) = v24;
        break;
      case 228:
      case 229:
        v56 = (unsigned __int64)sub_379EB70((__int64 *)a1, v55);
        result = v25;
        LODWORD(v57) = v25;
        break;
      case 234:
        v56 = (unsigned __int64)sub_37AE4B0((__int64 *)a1, v55, a4);
        result = v37;
        LODWORD(v57) = v37;
        break;
      case 235:
        v56 = (unsigned __int64)sub_37A0900((__int64 *)a1, v55);
        result = v36;
        LODWORD(v57) = v36;
        break;
      case 246:
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
        result = sub_378DF80(v58, a4);
        if ( (_BYTE)result )
          break;
        v10 = v55;
LABEL_11:
        v56 = (unsigned __int64)sub_37A0050((__int64 *)a1, v10);
        result = v14;
        LODWORD(v57) = v14;
        break;
      case 258:
      case 259:
        result = sub_378DF80(v58, a4);
        if ( !(_BYTE)result )
        {
          v56 = (unsigned __int64)sub_379FD50((__int64 *)a1, v55, a4);
          result = v26;
          LODWORD(v57) = v26;
        }
        break;
      case 261:
      case 287:
      case 288:
      case 289:
        result = sub_378DF80(v58, a4);
        if ( !(_BYTE)result )
        {
          v56 = (unsigned __int64)sub_37A0500((__int64 *)a1, v55, v54, a4);
          result = v20;
          LODWORD(v57) = v20;
        }
        break;
      case 275:
      case 276:
      case 277:
      case 278:
      case 450:
      case 451:
        v56 = sub_379EE60((__int64 *)a1, v55);
        result = v19;
        LODWORD(v57) = v19;
        break;
      case 298:
        v56 = (unsigned __int64)sub_3793DA0((__int64)a1, v55, a4);
        result = v43;
        LODWORD(v57) = v43;
        break;
      case 362:
        v56 = (unsigned __int64)sub_37A28B0((__int64 *)a1, v55);
        result = v42;
        LODWORD(v57) = v42;
        break;
      case 364:
        v56 = (unsigned __int64)sub_37A2C30((__int64 *)a1, v55, a4);
        result = v41;
        LODWORD(v57) = v41;
        break;
      case 468:
        v56 = (unsigned __int64)sub_37A24F0((__int64 *)a1, v55);
        result = v40;
        LODWORD(v57) = v40;
        break;
      case 469:
        v56 = (unsigned __int64)sub_37A26F0((__int64 **)a1, v55);
        result = v39;
        LODWORD(v57) = v39;
        break;
      case 470:
        v56 = (unsigned __int64)sub_37A3390((__int64 *)a1, v55);
        result = v38;
        LODWORD(v57) = v38;
        break;
      default:
        sub_C64ED0("Do not know how to widen the result of this operator!", 1u);
    }
    if ( v56 )
      return sub_3760B50((__int64)a1, v55, v54, v56, v57);
  }
  return result;
}
