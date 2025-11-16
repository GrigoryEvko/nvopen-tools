// Function: sub_1237990
// Address: 0x1237990
//
__int64 __fastcall sub_1237990(__int64 a1, __int64 *a2, __int64 a3, __int64 *a4)
{
  __int64 v4; // r15
  unsigned __int64 v5; // rbx
  const char *v6; // rax
  unsigned int v7; // r12d
  int v10; // eax
  int v11; // ecx
  _QWORD *v12; // rax
  _QWORD *v13; // rbx
  int v14; // eax
  int v15; // eax
  int v16; // eax
  int v17; // eax
  int v19; // [rsp+8h] [rbp-68h]
  int v20; // [rsp+Ch] [rbp-64h]
  int v21; // [rsp+Ch] [rbp-64h]
  const char *v22; // [rsp+10h] [rbp-60h] BYREF
  char v23; // [rsp+30h] [rbp-40h]
  char v24; // [rsp+31h] [rbp-3Fh]

  v4 = a1 + 176;
  v5 = *(_QWORD *)(a1 + 232);
  if ( !*(_DWORD *)(a1 + 240) )
  {
    v24 = 1;
    v6 = "found end of file when expecting more instructions";
LABEL_3:
    v22 = v6;
    v23 = 3;
    sub_11FD800(v4, v5, (__int64)&v22, 1);
    return 1;
  }
  v20 = *(_DWORD *)(a1 + 240);
  v19 = *(_DWORD *)(a1 + 280);
  v10 = sub_1205200(a1 + 176);
  *(_DWORD *)(a1 + 240) = v10;
  switch ( v20 )
  {
    case 60:
      v11 = 1;
      return (unsigned __int8)sub_1236B50(a1, a2, a4, v11);
    case 61:
      v11 = 2;
      return (unsigned __int8)sub_1236B50(a1, a2, a4, v11);
    case 62:
      v11 = 3;
      return (unsigned __int8)sub_1236B50(a1, a2, a4, v11);
    case 332:
      v21 = 0;
      while ( 2 )
      {
        switch ( v10 )
        {
          case 'M':
            v21 |= 2u;
            *(_DWORD *)(a1 + 240) = sub_1205200(v4);
            goto LABEL_99;
          case 'N':
            v21 |= 4u;
            *(_DWORD *)(a1 + 240) = sub_1205200(v4);
            goto LABEL_99;
          case 'O':
            v21 |= 8u;
            *(_DWORD *)(a1 + 240) = sub_1205200(v4);
            goto LABEL_99;
          case 'P':
            v21 |= 0x10u;
            *(_DWORD *)(a1 + 240) = sub_1205200(v4);
            goto LABEL_99;
          case 'Q':
            v21 |= 0x20u;
            *(_DWORD *)(a1 + 240) = sub_1205200(v4);
            goto LABEL_99;
          case 'R':
            v21 |= 1u;
            *(_DWORD *)(a1 + 240) = sub_1205200(v4);
            goto LABEL_99;
          case 'S':
            v21 |= 0x40u;
            *(_DWORD *)(a1 + 240) = sub_1205200(v4);
            goto LABEL_99;
          case 'T':
            v21 = -1;
            *(_DWORD *)(a1 + 240) = sub_1205200(v4);
LABEL_99:
            v10 = *(_DWORD *)(a1 + 240);
            continue;
          default:
            v7 = 1;
            if ( !(unsigned __int8)sub_1230940(a1, a2, a4, v19, 1) )
              goto LABEL_134;
            return v7;
        }
      }
    case 333:
    case 335:
    case 337:
    case 345:
      if ( v10 == 85 )
      {
        v15 = sub_1205200(v4);
        *(_DWORD *)(a1 + 240) = v15;
        if ( v15 != 86 )
        {
          if ( (unsigned __int8)sub_12347A0(a1, a2, a4, v19, 0) )
            return 1;
          sub_B447F0((unsigned __int8 *)*a2, 1);
          return 0;
        }
        goto LABEL_161;
      }
      if ( v10 != 86 )
        return (unsigned __int8)sub_12347A0(a1, a2, a4, v19, 0) != 0;
      v17 = sub_1205200(v4);
      *(_DWORD *)(a1 + 240) = v17;
      if ( v17 == 85 )
      {
LABEL_161:
        *(_DWORD *)(a1 + 240) = sub_1205200(v4);
        if ( (unsigned __int8)sub_12347A0(a1, a2, a4, v19, 0) )
          return 1;
        sub_B447F0((unsigned __int8 *)*a2, 1);
        goto LABEL_163;
      }
      if ( !(unsigned __int8)sub_12347A0(a1, a2, a4, v19, 0) )
      {
LABEL_163:
        sub_B44850((unsigned __int8 *)*a2, 1);
        return 0;
      }
      return 1;
    case 334:
    case 336:
    case 338:
    case 341:
    case 344:
      v21 = 0;
      while ( 2 )
      {
        switch ( v10 )
        {
          case 'M':
            v21 |= 2u;
            *(_DWORD *)(a1 + 240) = sub_1205200(v4);
            goto LABEL_12;
          case 'N':
            v21 |= 4u;
            *(_DWORD *)(a1 + 240) = sub_1205200(v4);
            goto LABEL_12;
          case 'O':
            v21 |= 8u;
            *(_DWORD *)(a1 + 240) = sub_1205200(v4);
            goto LABEL_12;
          case 'P':
            v21 |= 0x10u;
            *(_DWORD *)(a1 + 240) = sub_1205200(v4);
            goto LABEL_12;
          case 'Q':
            v21 |= 0x20u;
            *(_DWORD *)(a1 + 240) = sub_1205200(v4);
            goto LABEL_12;
          case 'R':
            v21 |= 1u;
            *(_DWORD *)(a1 + 240) = sub_1205200(v4);
            goto LABEL_12;
          case 'S':
            v21 |= 0x40u;
            *(_DWORD *)(a1 + 240) = sub_1205200(v4);
            goto LABEL_12;
          case 'T':
            v21 = -1;
            *(_DWORD *)(a1 + 240) = sub_1205200(v4);
LABEL_12:
            v10 = *(_DWORD *)(a1 + 240);
            continue;
          default:
            v7 = 1;
            if ( !(unsigned __int8)sub_12347A0(a1, a2, a4, v19, 1) )
              goto LABEL_134;
            return v7;
        }
      }
    case 339:
    case 340:
    case 346:
    case 347:
      if ( v10 != 88 )
        return (unsigned __int8)sub_12347A0(a1, a2, a4, v19, 0);
      *(_DWORD *)(a1 + 240) = sub_1205200(v4);
      if ( (unsigned __int8)sub_12347A0(a1, a2, a4, v19, 0) )
        return 1;
      v7 = 0;
      sub_B448B0(*a2, 1);
      return v7;
    case 342:
    case 343:
      return (unsigned __int8)sub_12347A0(a1, a2, a4, v19, 0);
    case 348:
    case 350:
      return (unsigned __int8)sub_12348F0(a1, a2, a4, v19);
    case 349:
      if ( v10 != 89 )
        return (unsigned __int8)sub_12348F0(a1, a2, a4, v19);
      *(_DWORD *)(a1 + 240) = sub_1205200(v4);
      if ( !(unsigned __int8)sub_12348F0(a1, a2, a4, v19) )
        goto LABEL_68;
      return 1;
    case 351:
      if ( v10 != 92 )
        return (unsigned __int8)sub_1234A00(a1, a2, a4, v19);
      *(_DWORD *)(a1 + 240) = sub_1205200(v4);
      if ( (unsigned __int8)sub_1234A00(a1, a2, a4, v19) )
        return 1;
LABEL_68:
      v7 = 0;
      *(_BYTE *)(*a2 + 1) |= 2u;
      return v7;
    case 352:
      v21 = 0;
      while ( 2 )
      {
        switch ( v10 )
        {
          case 'M':
            v21 |= 2u;
            *(_DWORD *)(a1 + 240) = sub_1205200(v4);
            goto LABEL_110;
          case 'N':
            v21 |= 4u;
            *(_DWORD *)(a1 + 240) = sub_1205200(v4);
            goto LABEL_110;
          case 'O':
            v21 |= 8u;
            *(_DWORD *)(a1 + 240) = sub_1205200(v4);
            goto LABEL_110;
          case 'P':
            v21 |= 0x10u;
            *(_DWORD *)(a1 + 240) = sub_1205200(v4);
            goto LABEL_110;
          case 'Q':
            v21 |= 0x20u;
            *(_DWORD *)(a1 + 240) = sub_1205200(v4);
            goto LABEL_110;
          case 'R':
            v21 |= 1u;
            *(_DWORD *)(a1 + 240) = sub_1205200(v4);
            goto LABEL_110;
          case 'S':
            v21 |= 0x40u;
            *(_DWORD *)(a1 + 240) = sub_1205200(v4);
            goto LABEL_110;
          case 'T':
            v21 = -1;
            *(_DWORD *)(a1 + 240) = sub_1205200(v4);
LABEL_110:
            v10 = *(_DWORD *)(a1 + 240);
            continue;
          default:
            v7 = 1;
            if ( !(unsigned __int8)sub_1234A00(a1, a2, a4, v19) )
              goto LABEL_134;
            return v7;
        }
      }
    case 353:
      v21 = 0;
      while ( 2 )
      {
        switch ( v10 )
        {
          case 'M':
            v21 |= 2u;
            *(_DWORD *)(a1 + 240) = sub_1205200(v4);
            goto LABEL_121;
          case 'N':
            v21 |= 4u;
            *(_DWORD *)(a1 + 240) = sub_1205200(v4);
            goto LABEL_121;
          case 'O':
            v21 |= 8u;
            *(_DWORD *)(a1 + 240) = sub_1205200(v4);
            goto LABEL_121;
          case 'P':
            v21 |= 0x10u;
            *(_DWORD *)(a1 + 240) = sub_1205200(v4);
            goto LABEL_121;
          case 'Q':
            v21 |= 0x20u;
            *(_DWORD *)(a1 + 240) = sub_1205200(v4);
            goto LABEL_121;
          case 'R':
            v21 |= 1u;
            *(_DWORD *)(a1 + 240) = sub_1205200(v4);
            goto LABEL_121;
          case 'S':
            v21 |= 0x40u;
            *(_DWORD *)(a1 + 240) = sub_1205200(v4);
            goto LABEL_121;
          case 'T':
            v21 = -1;
            *(_DWORD *)(a1 + 240) = sub_1205200(v4);
LABEL_121:
            v10 = *(_DWORD *)(a1 + 240);
            continue;
          default:
            v7 = sub_1224DD0((_QWORD **)a1, a2, a4);
            if ( v7 || !v21 )
              return v7;
            if ( (unsigned __int8)sub_920620(*a2) )
              goto LABEL_135;
            v24 = 1;
            v6 = "fast-math-flags specified for phi without floating-point scalar or vector return type";
            break;
        }
        goto LABEL_3;
      }
    case 354:
      v11 = 0;
      return (unsigned __int8)sub_1236B50(a1, a2, a4, v11);
    case 355:
      if ( v10 == 85 )
      {
        v14 = sub_1205200(v4);
        *(_DWORD *)(a1 + 240) = v14;
        if ( v14 != 86 )
        {
          if ( (unsigned __int8)sub_12336C0(a1, a2, a4, v19) )
            return 1;
          *(_BYTE *)(*a2 + 1) |= 2u;
          return 0;
        }
        goto LABEL_157;
      }
      if ( v10 != 86 )
        return (unsigned __int8)sub_12336C0(a1, a2, a4, v19) != 0;
      v16 = sub_1205200(v4);
      *(_DWORD *)(a1 + 240) = v16;
      if ( v16 == 85 )
      {
LABEL_157:
        *(_DWORD *)(a1 + 240) = sub_1205200(v4);
        if ( (unsigned __int8)sub_12336C0(a1, a2, a4, v19) )
          return 1;
        *(_BYTE *)(*a2 + 1) |= 2u;
        goto LABEL_159;
      }
      if ( !(unsigned __int8)sub_12336C0(a1, a2, a4, v19) )
      {
LABEL_159:
        v7 = 0;
        *(_BYTE *)(*a2 + 1) |= 4u;
        return v7;
      }
      return 1;
    case 356:
    case 360:
      if ( v10 != 91 )
        return (unsigned __int8)sub_12336C0(a1, a2, a4, v19);
      *(_DWORD *)(a1 + 240) = sub_1205200(v4);
      if ( (unsigned __int8)sub_12336C0(a1, a2, a4, v19) )
        return 1;
      v7 = 0;
      sub_B448D0(*a2, 1);
      return v7;
    case 357:
    case 361:
    case 362:
    case 363:
    case 364:
    case 365:
    case 366:
    case 367:
      return (unsigned __int8)sub_12336C0(a1, a2, a4, v19);
    case 358:
    case 359:
      v21 = 0;
      while ( 2 )
      {
        switch ( v10 )
        {
          case 'M':
            v21 |= 2u;
            *(_DWORD *)(a1 + 240) = sub_1205200(v4);
            goto LABEL_33;
          case 'N':
            v21 |= 4u;
            *(_DWORD *)(a1 + 240) = sub_1205200(v4);
            goto LABEL_33;
          case 'O':
            v21 |= 8u;
            *(_DWORD *)(a1 + 240) = sub_1205200(v4);
            goto LABEL_33;
          case 'P':
            v21 |= 0x10u;
            *(_DWORD *)(a1 + 240) = sub_1205200(v4);
            goto LABEL_33;
          case 'Q':
            v21 |= 0x20u;
            *(_DWORD *)(a1 + 240) = sub_1205200(v4);
            goto LABEL_33;
          case 'R':
            v21 |= 1u;
            *(_DWORD *)(a1 + 240) = sub_1205200(v4);
            goto LABEL_33;
          case 'S':
            v21 |= 0x40u;
            *(_DWORD *)(a1 + 240) = sub_1205200(v4);
            goto LABEL_33;
          case 'T':
            v21 = -1;
            *(_DWORD *)(a1 + 240) = sub_1205200(v4);
LABEL_33:
            v10 = *(_DWORD *)(a1 + 240);
            continue;
          default:
            v7 = 1;
            if ( (unsigned __int8)sub_12336C0(a1, a2, a4, v19) )
              return v7;
LABEL_134:
            v7 = 0;
            if ( v21 )
              goto LABEL_135;
            return v7;
        }
      }
    case 368:
      v21 = 0;
      while ( 2 )
      {
        switch ( v10 )
        {
          case 'M':
            v21 |= 2u;
            *(_DWORD *)(a1 + 240) = sub_1205200(v4);
            goto LABEL_88;
          case 'N':
            v21 |= 4u;
            *(_DWORD *)(a1 + 240) = sub_1205200(v4);
            goto LABEL_88;
          case 'O':
            v21 |= 8u;
            *(_DWORD *)(a1 + 240) = sub_1205200(v4);
            goto LABEL_88;
          case 'P':
            v21 |= 0x10u;
            *(_DWORD *)(a1 + 240) = sub_1205200(v4);
            goto LABEL_88;
          case 'Q':
            v21 |= 0x20u;
            *(_DWORD *)(a1 + 240) = sub_1205200(v4);
            goto LABEL_88;
          case 'R':
            v21 |= 1u;
            *(_DWORD *)(a1 + 240) = sub_1205200(v4);
            goto LABEL_88;
          case 'S':
            v21 |= 0x40u;
            *(_DWORD *)(a1 + 240) = sub_1205200(v4);
            goto LABEL_88;
          case 'T':
            v21 = -1;
            *(_DWORD *)(a1 + 240) = sub_1205200(v4);
LABEL_88:
            v10 = *(_DWORD *)(a1 + 240);
            continue;
          default:
            v7 = 1;
            if ( (unsigned __int8)sub_1232F30(a1, (unsigned __int8 **)a2, a4) )
              return v7;
            v7 = 0;
            if ( !v21 )
              return v7;
            if ( (unsigned __int8)sub_920620(*a2) )
            {
LABEL_135:
              sub_B45150(*a2, v21);
              return v7;
            }
            v24 = 1;
            v6 = "fast-math-flags specified for select without floating-point scalar or vector return type";
            break;
        }
        goto LABEL_3;
      }
    case 369:
      return (unsigned __int8)sub_1233170(a1, (unsigned __int8 **)a2, a4);
    case 370:
      return (unsigned __int8)sub_1230A50(a1, a2, a4);
    case 375:
      return (unsigned __int8)sub_1234D00(a1, a2, a3, a4);
    case 376:
      return (unsigned __int8)sub_122FF30((__int64 **)a1, a2, a4);
    case 377:
      return (unsigned __int8)sub_12300F0(a1, a2, a4);
    case 378:
      return (unsigned __int8)sub_1230530(a1, a2, a4);
    case 379:
      return (unsigned __int8)sub_12351A0(a1, a2, a4);
    case 380:
      return (unsigned __int8)sub_12308C0((__int64 **)a1, a2, a4);
    case 381:
      v12 = sub_BD2C40(72, unk_3F148B8);
      v13 = v12;
      if ( v12 )
        sub_B4C8A0((__int64)v12, *(_QWORD *)a1, 0, 0);
      *a2 = (__int64)v13;
      return 0;
    case 382:
      return (unsigned __int8)sub_1234170((__int64 **)a1, a2, a4);
    case 383:
      return (unsigned __int8)sub_1234740(a1, a2, a4);
    case 384:
      return (unsigned __int8)sub_12342C0((__int64 **)a1, a2, a4);
    case 385:
      return (unsigned __int8)sub_122FAA0((__int64 **)a1, a2, a4);
    case 386:
      return (unsigned __int8)sub_122FC50((__int64 **)a1, a2, a4);
    case 387:
      return (unsigned __int8)sub_1235DD0(a1, a2, a4);
    case 388:
      return sub_1230CE0(a1, a2, a4);
    case 389:
      return sub_1231270(a1, a2, a4);
    case 390:
      return sub_1231630(a1, a2, a4);
    case 391:
      return sub_1210620((__int64 *)a1, a2);
    case 392:
      return sub_12319F0(a1, a2, a4);
    case 393:
      return sub_1231D90(a1, a2, a4);
    case 394:
      return sub_12324E0(a1, a2, a4);
    case 395:
      return (unsigned __int8)sub_1233300(a1, a2, a4);
    case 396:
      return (unsigned __int8)sub_1233410(a1, a2, a4);
    case 397:
      return (unsigned __int8)sub_1233580(a1, a2, a4);
    case 399:
      return sub_1232D30(a1, a2, a4);
    case 400:
      return sub_1233B90(a1, a2, a4);
    case 405:
      return (unsigned __int8)sub_1230C60((__int64 **)a1, a2, a4);
    default:
      v24 = 1;
      v6 = "expected instruction opcode";
      goto LABEL_3;
  }
}
