// Function: sub_378D7C0
// Address: 0x378d7c0
//
__int64 __fastcall sub_378D7C0(__int64 a1, unsigned __int64 a2, unsigned int a3, __m128i a4)
{
  unsigned int *v6; // rax
  __int64 v7; // rax
  unsigned int v8; // eax
  __int64 v9; // rcx
  __int64 v10; // r8
  __int64 v11; // r9
  __int64 v12; // rdx
  unsigned int v13; // r13d
  unsigned __int64 v15; // rcx
  unsigned int v16; // edx
  __int64 v17; // r8
  unsigned int v18; // edx
  unsigned int v19; // edx
  __int64 v20; // rax
  unsigned __int16 v21; // cx
  __int64 v22; // rsi
  __int64 v23; // rax
  __int64 v24; // rax
  unsigned __int16 v25; // dx
  __int64 v26; // rax
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
  unsigned int v44; // edx
  unsigned int v45; // edx
  unsigned int v46; // edx
  unsigned int v47; // edx
  unsigned int v48; // edx
  unsigned int v49; // edx
  unsigned int v50; // edx
  unsigned int v51; // edx
  unsigned int v52; // edx
  __int64 v53; // rdx
  __int64 v54; // rax
  __int64 v55; // rdx
  unsigned int v56; // edx
  unsigned __int16 v57; // [rsp+1E0h] [rbp-60h] BYREF
  __int64 v58; // [rsp+1E8h] [rbp-58h]
  unsigned __int16 v59; // [rsp+1F0h] [rbp-50h] BYREF
  __int64 v60; // [rsp+1F8h] [rbp-48h]
  unsigned __int64 v61; // [rsp+200h] [rbp-40h]
  __int64 v62; // [rsp+208h] [rbp-38h]
  unsigned __int64 v63; // [rsp+210h] [rbp-30h]
  __int64 v64; // [rsp+218h] [rbp-28h]

  v6 = (unsigned int *)(*(_QWORD *)(a2 + 40) + 40LL * a3);
  v7 = *(_QWORD *)(*(_QWORD *)v6 + 48LL) + 16LL * v6[2];
  v8 = sub_3761870((_QWORD *)a1, a2, *(_WORD *)v7, *(_QWORD *)(v7 + 8), 0);
  if ( (_BYTE)v8 )
    return 0;
  v12 = *(unsigned int *)(a2 + 24);
  v13 = v8;
  switch ( (int)v12 )
  {
    case 141:
    case 142:
    case 146:
    case 213:
    case 214:
    case 215:
    case 226:
    case 227:
    case 233:
    case 269:
    case 275:
    case 276:
    case 277:
    case 278:
    case 452:
    case 453:
      v15 = (unsigned __int64)sub_37856A0((__int64 *)a1, a2, a4);
      v17 = v18;
      break;
    case 143:
    case 144:
    case 220:
    case 221:
    case 454:
    case 455:
      v20 = *(_QWORD *)(a2 + 48);
      v21 = *(_WORD *)v20;
      v22 = *(_QWORD *)(v20 + 8);
      v57 = *(_WORD *)v20;
      v58 = v22;
      if ( (int)v12 > 239 )
      {
        v23 = (unsigned int)(v12 - 242) < 2 ? 0x28 : 0;
      }
      else
      {
        v23 = 40;
        if ( (int)v12 <= 237 )
          v23 = (unsigned int)(v12 - 143) < 6 ? 0x28 : 0;
      }
      v24 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a2 + 40) + v23) + 48LL)
          + 16LL * *(unsigned int *)(*(_QWORD *)(a2 + 40) + v23 + 8);
      v25 = *(_WORD *)v24;
      v26 = *(_QWORD *)(v24 + 8);
      if ( v21 == v25 && (v25 || v22 == v26)
        || (v59 = v25,
            v60 = v26,
            v63 = sub_2D5B750(&v59),
            v64 = v53,
            v54 = sub_2D5B750(&v57),
            v62 = v55,
            v61 = v54,
            (_BYTE)v55)
        && !(_BYTE)v64
        || v61 >= v63 )
      {
        v15 = (unsigned __int64)sub_37856A0((__int64 *)a1, a2, a4);
        v17 = v27;
      }
      else
      {
        v15 = (unsigned __int64)sub_378A4B0((__int64 *)a1, a2, a4);
        v17 = v56;
      }
      break;
    case 145:
    case 230:
    case 456:
      v15 = (unsigned __int64)sub_378BDE0((__int64 *)a1, a2, a4);
      v17 = v29;
      break;
    case 147:
    case 148:
    case 208:
    case 463:
      v15 = (unsigned __int64)sub_378B3B0((__int64 *)a1, a2, a4);
      v17 = v28;
      break;
    case 152:
    case 259:
      v15 = (unsigned __int64)sub_378C4F0((__int64 *)a1, a2);
      v17 = v42;
      break;
    case 158:
      v15 = (unsigned __int64)sub_3786920((__int64 *)a1, a2);
      v17 = v43;
      break;
    case 159:
      v15 = (unsigned __int64)sub_378A160(a1, a2, a4, v12, v9, v10, v11);
      v17 = v44;
      break;
    case 160:
      v15 = sub_3786140(a1, a2);
      v17 = v45;
      break;
    case 161:
      v15 = (unsigned __int64)sub_3786360(a1, a2, a4);
      v17 = v48;
      break;
    case 171:
      v15 = (unsigned __int64)sub_3785060((__int64 *)a1, a2);
      v17 = v49;
      break;
    case 184:
    case 185:
      v15 = (unsigned __int64)sub_378C8E0(a1, a2);
      v17 = v37;
      break;
    case 206:
      v15 = (unsigned __int64)sub_3784B90(a1, a2);
      v17 = v50;
      break;
    case 216:
    case 458:
      v15 = (unsigned __int64)sub_378A4B0((__int64 *)a1, a2, a4);
      v17 = v31;
      break;
    case 223:
    case 224:
    case 225:
      v15 = (unsigned __int64)sub_37874C0((__int64 *)a1, a2, a4);
      v17 = v30;
      break;
    case 228:
    case 229:
      v15 = (unsigned __int64)sub_378CBD0(a1, a2);
      v17 = v38;
      break;
    case 234:
      v15 = (unsigned __int64)sub_3785EA0((__int64 *)a1, a2, a4);
      v17 = v51;
      break;
    case 299:
      v15 = (unsigned __int64)sub_3789BF0((__int64 *)a1, a2);
      v17 = v52;
      break;
    case 363:
      v15 = (unsigned __int64)sub_3788820(a1, a2, a3);
      v17 = v47;
      break;
    case 364:
    case 470:
      v15 = sub_3787590((__int64 *)a1, a2, a4);
      v17 = v34;
      break;
    case 365:
    case 467:
      v15 = (unsigned __int64)sub_3789130((__int64 *)a1, a2, a3, a4);
      v17 = v33;
      break;
    case 368:
      v15 = (unsigned __int64)sub_3785D90(a1, a2);
      v17 = v46;
      break;
    case 374:
    case 375:
      v15 = (unsigned __int64)sub_37852A0(a1, a2, a4);
      v17 = v32;
      break;
    case 376:
    case 377:
    case 378:
    case 379:
    case 380:
    case 381:
    case 382:
    case 383:
    case 384:
    case 385:
    case 386:
    case 387:
    case 388:
    case 389:
    case 390:
      v15 = (unsigned __int64)sub_3785130(a1, a2, a3, a4);
      v17 = v19;
      break;
    case 391:
    case 392:
      v15 = sub_346F960(a4, *(_QWORD *)a1, a2, *(_QWORD **)(a1 + 8), v9, v10);
      v17 = v35;
      break;
    case 420:
    case 421:
      v15 = sub_378CEA0((__int64 *)a1, a2, a4);
      v17 = v36;
      break;
    case 465:
      v15 = (unsigned __int64)sub_3787680(a1, a2, a3);
      v17 = v40;
      break;
    case 466:
      v15 = (unsigned __int64)sub_3787FB0(a1, a2, a3);
      v17 = v41;
      break;
    case 471:
    case 472:
    case 473:
    case 474:
    case 475:
    case 476:
    case 477:
    case 478:
    case 479:
    case 480:
    case 481:
    case 482:
    case 483:
    case 484:
    case 485:
    case 486:
    case 487:
      v15 = (unsigned __int64)sub_3785410(a1, a2, a3, a4);
      v17 = v16;
      break;
    case 497:
      v15 = (unsigned __int64)sub_378D280((unsigned __int64 *)a1, a2, a4);
      v17 = v39;
      break;
    default:
      sub_C64ED0("Do not know how to split this operator's operand!\n", 1u);
  }
  if ( !v15 )
  {
    return 0;
  }
  else if ( a2 == v15 )
  {
    return 1;
  }
  else
  {
    sub_3760E70(a1, a2, 0, v15, v17);
  }
  return v13;
}
