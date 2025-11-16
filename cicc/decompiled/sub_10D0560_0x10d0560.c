// Function: sub_10D0560
// Address: 0x10d0560
//
__int64 __fastcall sub_10D0560(__int64 a1, int a2, unsigned __int8 *a3)
{
  unsigned int v3; // r12d
  __int64 v5; // rax
  __int64 v6; // r13
  char v7; // al
  __int64 v8; // rax
  __int64 v9; // rcx
  int v10; // ecx
  __int64 v11; // rcx
  __int64 v12; // rsi
  int v13; // esi
  __int64 v14; // rcx
  __int64 v15; // rax
  __int64 v16; // rax
  __int64 v17; // rax
  unsigned __int64 v18; // rax
  __int64 v19; // rdx
  unsigned __int8 *v20; // [rsp+8h] [rbp-38h]

  if ( a2 + 29 != *a3 )
    return 0;
  v5 = *((_QWORD *)a3 - 8);
  if ( *(_BYTE *)v5 == 93
    && *(_DWORD *)(v5 + 80) == 1
    && **(_DWORD **)(v5 + 72) == 1
    && (v11 = *(_QWORD *)(v5 - 32), *(_BYTE *)v11 == 85)
    && (v12 = *(_QWORD *)(v11 - 32)) != 0
    && !*(_BYTE *)v12
    && *(_QWORD *)(v12 + 24) == *(_QWORD *)(v11 + 80)
    && (*(_BYTE *)(v12 + 33) & 0x20) != 0 )
  {
    v13 = *(_DWORD *)(v12 + 36);
    if ( v13 != 312 )
    {
      switch ( v13 )
      {
        case 333:
        case 339:
        case 360:
        case 369:
        case 372:
          break;
        default:
          goto LABEL_5;
      }
    }
    **(_QWORD **)a1 = v11;
    **(_QWORD **)(a1 + 8) = v11;
    **(_QWORD **)(a1 + 16) = v5;
    v6 = *((_QWORD *)a3 - 4);
    v14 = *(_QWORD *)(v6 + 16);
    v7 = *(_BYTE *)v6;
    if ( v14 && !*(_QWORD *)(v14 + 8) && v7 == 82 )
    {
      v15 = *(_QWORD *)(v6 - 64);
      if ( *(_BYTE *)v15 != 93 )
        return 0;
      v20 = a3;
      if ( *(_DWORD *)(v15 + 80) != 1 || **(_DWORD **)(v15 + 72) || *(_QWORD *)(v15 - 32) != **(_QWORD **)(a1 + 32) )
        return 0;
      v3 = sub_991580(a1 + 40, *(_QWORD *)(v6 - 32));
      if ( (_BYTE)v3 )
      {
        if ( !*(_QWORD *)(a1 + 24) )
          return v3;
        goto LABEL_43;
      }
      a3 = v20;
      v6 = *((_QWORD *)v20 - 4);
      v7 = *(_BYTE *)v6;
    }
  }
  else
  {
LABEL_5:
    v6 = *((_QWORD *)a3 - 4);
    v7 = *(_BYTE *)v6;
  }
  if ( v7 != 93 )
    return 0;
  if ( *(_DWORD *)(v6 + 80) != 1 )
    return 0;
  if ( **(_DWORD **)(v6 + 72) != 1 )
    return 0;
  v8 = *(_QWORD *)(v6 - 32);
  if ( *(_BYTE *)v8 != 85 )
    return 0;
  v9 = *(_QWORD *)(v8 - 32);
  if ( !v9 || *(_BYTE *)v9 || *(_QWORD *)(v9 + 24) != *(_QWORD *)(v8 + 80) || (*(_BYTE *)(v9 + 33) & 0x20) == 0 )
    return 0;
  v10 = *(_DWORD *)(v9 + 36);
  if ( v10 == 312 )
  {
LABEL_34:
    **(_QWORD **)a1 = v8;
    **(_QWORD **)(a1 + 8) = v8;
    **(_QWORD **)(a1 + 16) = v6;
    v6 = *((_QWORD *)a3 - 8);
    v16 = *(_QWORD *)(v6 + 16);
    if ( v16 )
    {
      if ( !*(_QWORD *)(v16 + 8) && *(_BYTE *)v6 == 82 )
      {
        v17 = *(_QWORD *)(v6 - 64);
        if ( *(_BYTE *)v17 == 93
          && *(_DWORD *)(v17 + 80) == 1
          && !**(_DWORD **)(v17 + 72)
          && *(_QWORD *)(v17 - 32) == **(_QWORD **)(a1 + 32) )
        {
          v3 = sub_991580(a1 + 40, *(_QWORD *)(v6 - 32));
          if ( (_BYTE)v3 )
          {
            if ( !*(_QWORD *)(a1 + 24) )
              return v3;
LABEL_43:
            v18 = sub_B53900(v6);
            v19 = *(_QWORD *)(a1 + 24);
            *(_DWORD *)v19 = v18;
            *(_BYTE *)(v19 + 4) = BYTE4(v18);
            return v3;
          }
        }
      }
    }
    return 0;
  }
  v3 = 0;
  switch ( v10 )
  {
    case 333:
    case 339:
    case 360:
    case 369:
    case 372:
      goto LABEL_34;
    case 334:
    case 335:
    case 336:
    case 337:
    case 338:
    case 340:
    case 341:
    case 342:
    case 343:
    case 344:
    case 345:
    case 346:
    case 347:
    case 348:
    case 349:
    case 350:
    case 351:
    case 352:
    case 353:
    case 354:
    case 355:
    case 356:
    case 357:
    case 358:
    case 359:
    case 361:
    case 362:
    case 363:
    case 364:
    case 365:
    case 366:
    case 367:
    case 368:
    case 370:
    case 371:
      return 0;
    default:
      return v3;
  }
  return v3;
}
