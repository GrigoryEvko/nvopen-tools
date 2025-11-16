// Function: sub_1AED190
// Address: 0x1aed190
//
void __fastcall sub_1AED190(__int64 a1, __int64 *a2)
{
  _BYTE *v2; // r12
  char *v3; // rax
  __int64 v4; // rdx
  __int64 *v5; // rax
  int v6; // [rsp+4h] [rbp-2Ch] BYREF
  __int64 v7[5]; // [rsp+8h] [rbp-28h] BYREF

  v2 = *(_BYTE **)(a1 - 24);
  if ( !v2[16] && (v2[32] & 0xFu) - 7 > 1 && (v2[23] & 0x20) != 0 )
  {
    v3 = (char *)sub_1649960(*(_QWORD *)(a1 - 24));
    if ( (unsigned __int8)sub_149B630(*a2, v3, v4, &v6) )
    {
      if ( (((int)*(unsigned __int8 *)(*a2 + v6 / 4) >> (2 * (v6 & 3))) & 3) != 0 )
      {
        switch ( v6 )
        {
          case 108:
          case 109:
          case 110:
          case 155:
          case 156:
          case 157:
          case 162:
          case 163:
          case 164:
          case 165:
          case 166:
          case 170:
          case 176:
          case 177:
          case 178:
          case 184:
          case 185:
          case 186:
          case 203:
          case 204:
          case 205:
          case 209:
          case 210:
          case 211:
          case 212:
          case 213:
          case 214:
          case 275:
          case 276:
          case 277:
          case 288:
          case 289:
          case 292:
          case 301:
          case 302:
          case 303:
          case 333:
          case 334:
          case 335:
          case 337:
          case 338:
          case 339:
          case 344:
          case 345:
          case 349:
          case 353:
          case 354:
          case 355:
          case 361:
          case 366:
          case 368:
          case 371:
          case 377:
          case 403:
          case 404:
          case 405:
            if ( !(unsigned __int8)sub_1560180((__int64)(v2 + 112), 36) )
            {
              v7[0] = *(_QWORD *)(a1 + 56);
              v5 = (__int64 *)sub_16498A0(a1);
              *(_QWORD *)(a1 + 56) = sub_1563AB0(v7, v5, -1, 21);
            }
            break;
          default:
            return;
        }
      }
    }
  }
}
