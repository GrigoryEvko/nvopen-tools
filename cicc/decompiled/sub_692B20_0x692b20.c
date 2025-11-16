// Function: sub_692B20
// Address: 0x692b20
//
__int64 __fastcall sub_692B20(int a1)
{
  __int64 result; // rax
  __int64 v2; // rdx

  switch ( (__int16)a1 )
  {
    case 1:
    case 2:
    case 3:
    case 4:
    case 5:
    case 6:
    case 7:
    case 8:
    case 21:
    case 22:
    case 27:
    case 31:
    case 32:
    case 33:
    case 34:
    case 35:
    case 36:
    case 37:
    case 38:
    case 99:
    case 109:
    case 110:
    case 111:
    case 112:
    case 113:
    case 114:
    case 115:
    case 116:
    case 117:
    case 138:
    case 139:
    case 140:
    case 141:
    case 143:
    case 144:
    case 145:
    case 146:
    case 152:
    case 155:
    case 156:
    case 161:
    case 162:
    case 166:
    case 167:
    case 176:
    case 177:
    case 178:
    case 181:
    case 182:
    case 183:
    case 185:
    case 188:
    case 195:
    case 196:
    case 197:
    case 198:
    case 199:
    case 200:
    case 201:
    case 202:
    case 203:
    case 204:
    case 205:
    case 206:
    case 207:
    case 208:
    case 209:
    case 210:
    case 211:
    case 212:
    case 213:
    case 214:
    case 215:
    case 216:
    case 217:
    case 218:
    case 219:
    case 220:
    case 221:
    case 222:
    case 223:
    case 224:
    case 225:
    case 226:
    case 227:
    case 228:
    case 229:
    case 230:
    case 231:
    case 232:
    case 233:
    case 234:
    case 235:
    case 237:
    case 242:
    case 243:
    case 247:
    case 251:
    case 252:
    case 253:
    case 254:
    case 255:
    case 256:
    case 257:
    case 258:
    case 259:
    case 261:
    case 262:
    case 267:
    case 269:
    case 270:
    case 271:
    case 282:
    case 284:
    case 285:
    case 286:
    case 288:
    case 289:
    case 290:
    case 291:
    case 292:
    case 293:
    case 294:
    case 296:
    case 297:
    case 298:
    case 299:
    case 300:
    case 301:
    case 302:
    case 303:
    case 304:
    case 305:
    case 306:
    case 307:
    case 308:
    case 309:
    case 310:
    case 311:
    case 312:
    case 313:
    case 314:
    case 315:
    case 316:
    case 317:
    case 318:
    case 319:
    case 320:
    case 321:
    case 322:
    case 323:
    case 324:
    case 325:
    case 326:
    case 327:
    case 328:
    case 329:
    case 330:
    case 336:
    case 337:
    case 338:
    case 355:
    case 356:
      return 1;
    case 25:
      return unk_4D0448C;
    case 77:
      result = 0;
      if ( dword_4F077C4 == 2 )
        return unk_4F07778 > 202301;
      return result;
    default:
      result = 0;
      if ( dword_4F077C4 != 2 )
        return result;
      if ( (unsigned __int16)(a1 - 80) > 0x30u )
      {
        result = 1;
        if ( (_WORD)a1 == 165 || (_WORD)a1 == 180 )
          return result;
      }
      else
      {
        v2 = 0x1C70006066221LL;
        if ( _bittest64(&v2, (unsigned int)(a1 - 80)) )
          return 1;
      }
      result = 1;
      if ( (unsigned __int16)(a1 - 331) > 4u
        && (_WORD)a1 != 18
        && (!(unk_4D04548 | unk_4D04558) || (unsigned __int16)(a1 - 133) > 3u) )
      {
        if ( (_WORD)a1 == 239 )
          return 1;
        result = 1;
        if ( (unsigned __int16)(a1 - 272) > 8u
          && (!(_DWORD)qword_4F077B4 || (_WORD)a1 != 236 && (unsigned __int16)(a1 - 339) > 0xFu) )
        {
          result = dword_4F077BC;
          if ( dword_4F077BC )
          {
            result = 0;
            if ( qword_4F077A8 <= 0x76BFu )
            {
              if ( (_WORD)a1 == 101 || (_WORD)a1 == 104 )
                return 1;
              return (((_WORD)a1 - 87) & 0xFFBF) == 0;
            }
          }
        }
      }
      return result;
  }
}
