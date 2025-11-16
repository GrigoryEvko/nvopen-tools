// Function: sub_36EFA70
// Address: 0x36efa70
//
__int64 __fastcall sub_36EFA70(__int64 a1, __int64 a2, __m128i a3)
{
  __int64 v3; // rdx
  __int64 v4; // rcx
  __int64 v5; // r8
  __int64 v6; // r9
  __int64 v7; // rax
  bool v8; // cc
  _QWORD *v9; // rax
  __int64 result; // rax
  int v11; // eax
  int v12; // eax
  int v13; // eax
  int v14; // ecx
  int v15; // ecx
  int v16; // ecx
  int v17; // ecx
  int v18; // ecx
  int v19; // ecx
  int v20; // ecx
  int v21; // ecx

  v3 = sub_2E79000(*(__int64 **)(*(_QWORD *)(a1 + 64) + 40LL));
  v7 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a2 + 40) + 40LL) + 96LL);
  v8 = *(_DWORD *)(v7 + 32) <= 0x40u;
  v9 = *(_QWORD **)(v7 + 24);
  if ( !v8 )
    v9 = (_QWORD *)*v9;
  if ( (unsigned int)v9 <= 0x20E8 )
  {
    if ( (unsigned int)v9 > 0x2057 )
    {
      switch ( (int)v9 )
      {
        case 8280:
          goto LABEL_38;
        case 8315:
          sub_36E5F70(a1, a2, v3, v4, v5, v6);
          return 1;
        case 8316:
          sub_36EB980(a1, a2);
          return 1;
        case 8318:
          sub_36E62C0(a1, a2);
          return 1;
        case 8320:
          sub_36E5D40(a1, a2);
          return 1;
        case 8322:
          sub_36EBCF0(a1, a2);
          return 1;
        case 8324:
        case 8325:
        case 8326:
        case 8327:
        case 8328:
        case 8329:
        case 8330:
        case 8331:
          sub_36EC510(a1, a2);
          return 1;
        case 8332:
        case 8333:
        case 8334:
          sub_36EF280(a1, a2, 1u);
          return 1;
        case 8341:
        case 8342:
        case 8343:
        case 8344:
        case 8345:
          sub_36EF280(a1, a2, 0);
          return 1;
        case 8347:
        case 8348:
        case 8349:
          v16 = 1;
          goto LABEL_47;
        case 8350:
        case 8351:
        case 8352:
        case 8353:
        case 8354:
          v16 = 0;
LABEL_47:
          sub_36EF6F0(a1, a2, 0, v16, a3);
          return 1;
        case 8355:
        case 8356:
        case 8357:
          v15 = 1;
          goto LABEL_45;
        case 8358:
        case 8359:
        case 8360:
        case 8361:
        case 8362:
          v15 = 0;
LABEL_45:
          sub_36EF6F0(a1, a2, 5u, v15, a3);
          return 1;
        case 8363:
        case 8364:
        case 8365:
          v14 = 1;
          goto LABEL_43;
        case 8366:
        case 8367:
        case 8368:
        case 8369:
        case 8370:
          v14 = 0;
LABEL_43:
          sub_36EF6F0(a1, a2, 4u, v14, a3);
          return 1;
        case 8371:
        case 8372:
        case 8373:
          v21 = 1;
          goto LABEL_60;
        case 8374:
        case 8375:
        case 8376:
        case 8377:
        case 8378:
          v21 = 0;
LABEL_60:
          sub_36EF6F0(a1, a2, 3u, v21, a3);
          return 1;
        case 8379:
        case 8380:
        case 8381:
          v20 = 1;
          goto LABEL_58;
        case 8382:
        case 8383:
        case 8384:
        case 8385:
        case 8386:
          v20 = 0;
LABEL_58:
          sub_36EF6F0(a1, a2, 2u, v20, a3);
          return 1;
        case 8387:
        case 8388:
        case 8389:
          v19 = 1;
          goto LABEL_56;
        case 8390:
        case 8391:
        case 8392:
        case 8393:
        case 8394:
          v19 = 0;
LABEL_56:
          sub_36EF6F0(a1, a2, 1u, v19, a3);
          return 1;
        case 8395:
        case 8396:
        case 8397:
          v18 = 1;
          goto LABEL_54;
        case 8398:
        case 8399:
        case 8400:
        case 8401:
        case 8402:
          v18 = 0;
LABEL_54:
          sub_36EF6F0(a1, a2, 6u, v18, a3);
          return 1;
        case 8403:
        case 8404:
        case 8405:
          v17 = 1;
          goto LABEL_52;
        case 8406:
        case 8407:
        case 8408:
        case 8409:
        case 8410:
          v17 = 0;
LABEL_52:
          sub_36EF6F0(a1, a2, 7u, v17, a3);
          result = 1;
          break;
        case 8411:
        case 8412:
        case 8413:
          sub_36EEFA0(a1, a2, 1);
          result = 1;
          break;
        case 8414:
        case 8415:
        case 8416:
        case 8417:
        case 8418:
          sub_36EEFA0(a1, a2, 0);
          result = 1;
          break;
        case 8420:
        case 8421:
        case 8422:
        case 8423:
        case 8424:
          sub_36EBF00(a1, a2);
          result = 1;
          break;
        default:
          return 0;
      }
      return result;
    }
    return 0;
  }
  if ( (unsigned int)v9 > 0x2402 )
  {
    if ( (unsigned int)v9 <= 0x27DB )
    {
      if ( (unsigned int)v9 > 0x27D3 )
      {
        sub_36DAD70(a1, a2, 1, a3);
        return 1;
      }
      if ( (unsigned int)v9 <= 0x240A )
      {
        if ( (unsigned int)v9 > 0x2405 )
          sub_36EDCF0(a1, a2, 0);
        else
          sub_36EDCF0(a1, a2, 3u);
        return 1;
      }
      if ( (unsigned int)v9 <= 0x24A6 )
      {
        if ( (unsigned int)v9 > 0x24A4 )
        {
          sub_36DB420(a1, a2, a3);
          return 1;
        }
        return 0;
      }
      if ( (unsigned int)((_DWORD)v9 - 10183) > 0xC )
        return 0;
      goto LABEL_33;
    }
    if ( (unsigned int)v9 > 0x2856 )
    {
      if ( (unsigned int)v9 <= 0x285E )
      {
        sub_36E64A0(a1, a2, 1, a3);
        return 1;
      }
      if ( (unsigned int)((_DWORD)v9 - 10335) > 0xF )
        return 0;
    }
    else if ( (unsigned int)v9 <= 0x2849 )
    {
      if ( (unsigned int)v9 > 0x27EB )
      {
        if ( (unsigned int)((_DWORD)v9 - 10299) <= 9 )
        {
          sub_36E9630(a1, a2, a3);
          return 1;
        }
        return 0;
      }
LABEL_33:
      sub_36DAD70(a1, a2, 0, a3);
      return 1;
    }
    sub_36E64A0(a1, a2, 0, a3);
    return 1;
  }
  if ( (unsigned int)v9 > 0x23FF )
  {
    sub_36EDCF0(a1, a2, 2u);
    return 1;
  }
  if ( (_DWORD)v9 == 9057 )
  {
    sub_36EB890(a1, a2);
    return 1;
  }
  if ( (unsigned int)v9 > 0x2361 )
  {
    if ( (_DWORD)v9 == 9145 )
    {
      sub_36E6D40(a1, a2, a3);
      return 1;
    }
    if ( (unsigned int)v9 <= 0x23B9 )
    {
      if ( (_DWORD)v9 == 9058 )
      {
        sub_36EB6A0(a1, a2, a3, v3, v4, v5, v6);
        return 1;
      }
    }
    else if ( (unsigned int)((_DWORD)v9 - 9213) <= 2 )
    {
      sub_36EDCF0(a1, a2, 1u);
      return 1;
    }
    return 0;
  }
  switch ( (int)v9 )
  {
    case 8837:
    case 8851:
    case 8861:
      v13 = sub_36D7360((unsigned int)v9, a2, v3);
      sub_36E91F0(a1, 0, v13, a2, a3);
      result = 1;
      break;
    case 8838:
    case 8852:
    case 8862:
      v12 = sub_36D7360((unsigned int)v9, a2, v3);
      sub_36E91F0(a1, 1, v12, a2, a3);
      result = 1;
      break;
    case 8890:
    case 8898:
    case 8906:
    case 8914:
LABEL_38:
      v11 = sub_36D7360((unsigned int)v9, a2, v3);
      sub_36E8110(a1, v11, a2, a3);
      result = 1;
      break;
    default:
      return 0;
  }
  return result;
}
