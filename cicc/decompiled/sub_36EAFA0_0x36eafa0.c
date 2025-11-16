// Function: sub_36EAFA0
// Address: 0x36eafa0
//
__int64 __fastcall sub_36EAFA0(__int64 a1, __int64 a2, __m128i a3)
{
  __int64 v4; // rcx
  __int64 v5; // r8
  __int64 v6; // r9
  __int64 v7; // rax
  __int64 v8; // rdx
  unsigned int v9; // edi
  __int64 result; // rax
  int v11; // eax
  __int64 v12; // rdx
  int v13; // eax
  int v14; // eax
  int v15; // eax
  int v16; // eax
  int v17; // eax
  int v18; // eax
  int v19; // eax
  int v20; // eax
  int v21; // eax
  int v22; // eax
  int v23; // eax
  int v24; // eax
  int v25; // eax
  int v26; // eax
  int v27; // eax
  int v28; // eax
  int v29; // eax
  int v30; // eax
  int v31; // eax
  int v32; // eax

  v5 = sub_2E79000(*(__int64 **)(*(_QWORD *)(a1 + 64) + 40LL));
  v7 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a2 + 40) + 40LL) + 96LL);
  if ( *(_DWORD *)(v7 + 32) <= 0x40u )
    v8 = *(_QWORD *)(v7 + 24);
  else
    v8 = **(_QWORD **)(v7 + 24);
  v9 = v8;
  if ( (unsigned int)v8 <= 0x236B )
  {
    if ( (unsigned int)v8 <= 0x2273 )
    {
      switch ( (int)v8 )
      {
        case 8276:
          v15 = sub_36D7360(0x2054u, a2, v5);
          sub_36E7B50(a1, 0, 4u, v15, a2);
          result = 1;
          break;
        case 8277:
          v14 = sub_36D7360(0x2055u, a2, v5);
          sub_36E7B50(a1, 1u, 4u, v14, a2);
          result = 1;
          break;
        case 8278:
          v12 = v5;
          v9 = 8278;
LABEL_34:
          v13 = sub_36D7360(v9, a2, v12);
          sub_36E7EA0(a1, v13, a2);
          result = 1;
          break;
        case 8279:
          v11 = sub_36D7360(0x2057u, a2, v5);
          sub_36E8BD0(a1, 4u, v11, a2, a3);
          result = 1;
          break;
        case 8293:
        case 8294:
        case 8295:
        case 8296:
        case 8297:
          sub_36DA9E0(a1, a2, v8, v4, v5, v6);
          result = 1;
          break;
        default:
          return 0;
      }
    }
    else
    {
      switch ( (int)v8 )
      {
        case 8820:
          sub_36DA4B0(a1, a2);
          result = 1;
          break;
        case 8829:
        case 8843:
        case 8853:
          v20 = sub_36D7360(v8, a2, v5);
          sub_36E72A0(a1, 0, v20, a2);
          result = 1;
          break;
        case 8830:
        case 8844:
        case 8854:
          v19 = sub_36D7360(v8, a2, v5);
          sub_36E72A0(a1, 1u, v19, a2);
          result = 1;
          break;
        case 8831:
        case 8832:
        case 8845:
        case 8846:
        case 8855:
        case 8856:
          v18 = sub_36D7360(v8, a2, v5);
          sub_36E7580(a1, v18, a2);
          result = 1;
          break;
        case 8833:
        case 8835:
        case 8847:
        case 8849:
        case 8857:
        case 8859:
          v17 = sub_36D7360(v8, a2, v5);
          sub_36E77C0(a1, 0, v17, a2, a3);
          result = 1;
          break;
        case 8834:
        case 8836:
        case 8848:
        case 8850:
        case 8858:
        case 8860:
          v16 = sub_36D7360(v8, a2, v5);
          sub_36E77C0(a1, 1, v16, a2, a3);
          result = 1;
          break;
        case 8883:
        case 8891:
        case 8899:
          v22 = sub_36D7360(v8, a2, v5);
          sub_36E7B50(a1, 0, 1u, v22, a2);
          result = 1;
          break;
        case 8884:
        case 8892:
        case 8900:
          v21 = sub_36D7360(v8, a2, v5);
          sub_36E7B50(a1, 0, 0, v21, a2);
          result = 1;
          break;
        case 8885:
        case 8893:
        case 8901:
          v26 = sub_36D7360(v8, a2, v5);
          sub_36E7B50(a1, 1u, 1u, v26, a2);
          result = 1;
          break;
        case 8886:
        case 8894:
        case 8902:
          v25 = sub_36D7360(v8, a2, v5);
          sub_36E7B50(a1, 1u, 0, v25, a2);
          result = 1;
          break;
        case 8887:
        case 8895:
        case 8903:
          v12 = v5;
          goto LABEL_34;
        case 8888:
        case 8896:
        case 8904:
          v24 = sub_36D7360(v8, a2, v5);
          sub_36E8630(a1, 1u, v24, a2, a3);
          result = 1;
          break;
        case 8889:
        case 8897:
        case 8905:
          v23 = sub_36D7360(v8, a2, v5);
          sub_36E8630(a1, 0, v23, a2, a3);
          result = 1;
          break;
        case 8907:
          v27 = sub_36D7360(0x22CBu, a2, v5);
          sub_36E7B50(a1, 0, 3u, v27, a2);
          result = 1;
          break;
        case 8908:
          v29 = sub_36D7360(0x22CCu, a2, v5);
          sub_36E7B50(a1, 0, 2u, v29, a2);
          result = 1;
          break;
        case 8909:
          v28 = sub_36D7360(0x22CDu, a2, v5);
          sub_36E7B50(a1, 1u, 3u, v28, a2);
          result = 1;
          break;
        case 8910:
          v32 = sub_36D7360(0x22CEu, a2, v5);
          sub_36E7B50(a1, 1u, 2u, v32, a2);
          result = 1;
          break;
        case 8911:
          v12 = v5;
          v9 = 8911;
          goto LABEL_34;
        case 8912:
          v31 = sub_36D7360(0x22D0u, a2, v5);
          sub_36E8630(a1, 3u, v31, a2, a3);
          result = 1;
          break;
        case 8913:
          v30 = sub_36D7360(0x22D1u, a2, v5);
          sub_36E8630(a1, 2u, v30, a2, a3);
          result = 1;
          break;
        case 8959:
        case 8960:
        case 8961:
          result = sub_36E0E00(a1, a2, a3);
          break;
        case 8985:
          sub_36EA5D0(a1, a2, 1);
          result = 1;
          break;
        case 8986:
          sub_36EA5D0(a1, a2, 2);
          result = 1;
          break;
        case 8987:
          sub_36EA5D0(a1, a2, 4);
          result = 1;
          break;
        case 9056:
        case 9059:
          sub_36EAA50(a1, a2, a3);
          result = 1;
          break;
        case 9067:
          sub_36E6A00(a1, a2, a3);
          result = 1;
          break;
        default:
          return 0;
      }
    }
    return result;
  }
  if ( (unsigned int)v8 > 0x283A )
  {
    if ( (_DWORD)v8 == 10649 )
      return sub_36E3F30(a1, a2);
    if ( (unsigned int)v8 <= 0x2999 )
    {
      if ( (_DWORD)v8 == 10641 )
        return sub_36E3D50(a1, a2);
      if ( (_DWORD)v8 == 10648 )
        return sub_36E3E40(a1, a2);
    }
    else if ( (unsigned int)(v8 - 10654) <= 0x7F )
    {
      sub_36DA6A0(a1, a2, v8, a3);
      return 1;
    }
    return 0;
  }
  if ( (unsigned int)v8 > 0x27EB )
  {
    switch ( (int)v8 )
    {
      case 10233:
      case 10234:
      case 10235:
      case 10236:
      case 10237:
      case 10238:
      case 10239:
      case 10240:
        sub_36DA1D0(a1, a2, 1, a3);
        result = 1;
        break;
      case 10257:
      case 10258:
      case 10259:
      case 10260:
      case 10261:
      case 10262:
      case 10263:
      case 10264:
      case 10265:
      case 10266:
      case 10267:
      case 10268:
      case 10269:
      case 10270:
      case 10271:
      case 10272:
      case 10273:
      case 10274:
      case 10275:
      case 10276:
      case 10277:
        sub_36DB0F0(a1, a2, 1, a3);
        result = 1;
        break;
      case 10278:
      case 10279:
      case 10280:
      case 10281:
      case 10282:
      case 10283:
      case 10284:
      case 10285:
      case 10286:
      case 10287:
      case 10288:
      case 10289:
      case 10290:
      case 10291:
      case 10292:
      case 10293:
      case 10294:
      case 10295:
      case 10296:
      case 10297:
      case 10298:
        sub_36DB0F0(a1, a2, 0, a3);
        result = 1;
        break;
      default:
        sub_36DA1D0(a1, a2, 0, a3);
        result = 1;
        break;
    }
    return result;
  }
  if ( (unsigned int)v8 > 0x27B6 )
  {
    if ( (unsigned int)(v8 - 10167) <= 0xF )
      goto LABEL_21;
    return 0;
  }
  if ( (unsigned int)v8 > 0x27AE )
  {
    sub_36DAB10(a1, a2, 1, a3);
    return 1;
  }
  if ( (_DWORD)v8 == 9376 )
    return sub_36E4040(a1, a2, a3);
  result = 0;
  if ( (unsigned int)(v8 - 10146) <= 0xC )
  {
LABEL_21:
    sub_36DAB10(a1, a2, 0, a3);
    return 1;
  }
  return result;
}
