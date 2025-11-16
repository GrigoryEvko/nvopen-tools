// Function: sub_36F0030
// Address: 0x36f0030
//
void __fastcall sub_36F0030(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, size_t a6, __m128i a7)
{
  int v7; // eax
  char v8; // dl
  char v9; // dl
  __int64 v10; // rax

  v7 = *(_DWORD *)(a2 + 24);
  if ( v7 < 0 )
  {
    *(_DWORD *)(a2 + 36) = -1;
    return;
  }
  if ( v7 > 578 )
    goto LABEL_17;
  if ( v7 > 517 )
  {
    switch ( v7 )
    {
      case 518:
        v10 = *(_QWORD *)(*(_QWORD *)(a2 + 40) + 40LL);
        if ( *(_DWORD *)(v10 + 24) == 501 )
          *(_QWORD *)(a1 + 968) = **(_QWORD **)(v10 + 40);
        goto LABEL_17;
      case 533:
        sub_36DC680(a1, a2, a7);
        return;
      case 534:
        sub_36DC890(a1, a2, a7);
        return;
      case 539:
        sub_36DCAA0(a1, a2);
        return;
      case 548:
      case 549:
      case 550:
        if ( !(unsigned __int8)sub_36E2470(a1, a2, a7) )
          goto LABEL_17;
        return;
      case 551:
      case 552:
        if ( !(unsigned __int8)sub_36E0E00(a1, a2, a7) )
          goto LABEL_17;
        return;
      case 553:
      case 554:
      case 555:
        if ( !(unsigned __int8)sub_36E30C0(a1, a2) )
          goto LABEL_17;
        return;
      case 556:
      case 557:
      case 558:
        v9 = 0;
        goto LABEL_35;
      case 559:
      case 560:
      case 561:
        v8 = 0;
        goto LABEL_33;
      case 562:
      case 563:
      case 564:
        v9 = 1;
LABEL_35:
        sub_36E4430(a1, a2, v9, a7);
        break;
      case 565:
      case 566:
      case 567:
        v8 = 1;
LABEL_33:
        sub_36E4A00(a1, a2, v8, a7);
        break;
      case 568:
      case 569:
      case 570:
        if ( !(unsigned __int8)sub_36DDC60(a1, a2, a7) )
          goto LABEL_17;
        break;
      case 571:
      case 572:
      case 573:
      case 574:
      case 575:
        if ( !(unsigned __int8)sub_36DE500(a1, a2, a7) )
          goto LABEL_17;
        break;
      case 576:
      case 577:
      case 578:
        if ( !(unsigned __int8)sub_36DE090(a1, a2, a7) )
          goto LABEL_17;
        break;
      default:
        goto LABEL_17;
    }
  }
  else
  {
    if ( v7 == 186 )
      goto LABEL_66;
    if ( v7 > 186 )
    {
      if ( v7 > 339 )
        goto LABEL_17;
      if ( v7 > 297 )
      {
        switch ( v7 )
        {
          case 298:
          case 338:
            if ( !(unsigned __int8)sub_36E1F40(a1, a2, a7) )
              goto LABEL_17;
            break;
          case 299:
          case 339:
            if ( !(unsigned __int8)sub_36E2B10(a1, a2, a7) )
              goto LABEL_17;
            break;
          case 315:
            *(_BYTE *)(a1 + 961) = 0;
            if ( !(unsigned __int8)sub_36E3A70((__int64 *)a1, a2) )
              goto LABEL_17;
            break;
          case 316:
            *(_BYTE *)(a1 + 961) = 1;
            *(_QWORD *)(a1 + 968) = 0;
            goto LABEL_17;
          case 337:
            if ( !(unsigned __int8)sub_36E3960(a1, a2) )
              goto LABEL_17;
            break;
          default:
            goto LABEL_17;
        }
        return;
      }
      if ( v7 > 192 )
      {
        if ( v7 == 235 )
        {
          sub_36DD770(a1, a2, a7, a3, a4, a5, a6);
          return;
        }
        goto LABEL_17;
      }
      if ( v7 <= 190 )
      {
LABEL_17:
        sub_3425710((__int64 *)a1, a2, byte_4501140, 114026, a5, a6, a7);
        return;
      }
LABEL_66:
      if ( *(_DWORD *)(*(_QWORD *)(a1 + 1136) + 344LL) <= 0x45u && (unsigned __int8)sub_36D7E80(a1, a2, a7) )
        return;
      goto LABEL_17;
    }
    if ( v7 > 98 )
    {
      if ( v7 == 158 && (unsigned __int8)sub_36DCBA0(a1, a2) )
        return;
      goto LABEL_17;
    }
    if ( v7 <= 45 )
      goto LABEL_17;
    switch ( v7 )
    {
      case '.':
        if ( !(unsigned __int8)sub_36E5B70((__int64 *)a1, a2, a7, a3, a4, a5, a6) )
          goto LABEL_17;
        break;
      case '/':
        if ( !(unsigned __int8)sub_36EAFA0(a1, a2, a7) )
          goto LABEL_17;
        break;
      case '0':
        if ( !(unsigned __int8)sub_36EFA70(a1, a2, a7) )
          goto LABEL_17;
        break;
      case '1':
        if ( *(_WORD *)(*(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a2 + 40) + 40LL) + 48LL)
                      + 16LL * *(unsigned int *)(*(_QWORD *)(a2 + 40) + 48LL)) != 9 )
          goto LABEL_17;
        sub_36E0720(a1, a2);
        break;
      case '2':
        if ( *(_WORD *)(*(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a2 + 40) + 40LL) + 48LL)
                      + 16LL * *(unsigned int *)(*(_QWORD *)(a2 + 40) + 48LL)) != 9 )
          goto LABEL_17;
        sub_36E0AB0(a1, a2);
        break;
      case '`':
      case 'a':
      case 'b':
        if ( !(unsigned __int8)sub_36DF2D0(a1, a2) )
          goto LABEL_17;
        break;
      default:
        goto LABEL_17;
    }
  }
}
