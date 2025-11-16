// Function: sub_1FDD7A0
// Address: 0x1fdd7a0
//
__int64 __fastcall sub_1FDD7A0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  unsigned __int8 v5; // al
  unsigned int v6; // r13d
  unsigned __int64 v7; // r14
  __int64 v8; // rsi
  __int64 v9; // rsi
  __int64 v10; // r13
  __int64 v11; // rax
  _BYTE *v12; // rdx
  __int64 v13; // rsi
  unsigned __int8 *v14; // rsi
  unsigned int v16; // eax
  __int64 v17; // rax
  _BYTE *v18; // rdx
  __int64 v19; // rsi
  unsigned __int8 *v20; // rsi
  __int64 v21; // rax
  unsigned __int64 v22; // rsi
  __int64 v23; // rdx
  __int64 v24; // r13
  __int64 v25; // r13
  __int64 v26; // r15
  __int64 v27; // r13
  __int64 v28; // rdx
  char *v29; // rax
  __int64 v30; // rdx
  __int64 v31; // rax
  __int64 v32; // rdi
  __int64 v33; // rdx
  unsigned __int64 v34; // rsi
  unsigned __int64 v35; // rax
  __int64 v36; // rsi
  _BYTE *v37; // [rsp+0h] [rbp-50h]
  __int64 *v38; // [rsp+8h] [rbp-48h]
  int v39; // [rsp+14h] [rbp-3Ch] BYREF
  _QWORD v40[7]; // [rsp+18h] [rbp-38h] BYREF

  v37 = *(_BYTE **)(a1 + 144);
  v5 = *(_BYTE *)(a2 + 16);
  if ( (unsigned int)v5 - 25 <= 9 )
  {
    v6 = sub_1FDCFE0((_QWORD *)a1, *(_QWORD *)(a2 + 40));
    if ( !(_BYTE)v6 )
    {
      sub_1FD3D60((_QWORD *)a1, v37);
      return v6;
    }
    v5 = *(_BYTE *)(a2 + 16);
  }
  if ( v5 > 0x17u )
  {
    v7 = a2 | 4;
    if ( v5 != 78 )
    {
      if ( v5 != 29 )
        goto LABEL_7;
      v7 = a2 & 0xFFFFFFFFFFFFFFFBLL;
    }
    if ( (v7 & 0xFFFFFFFFFFFFFFF8LL) != 0 && *(char *)((v7 & 0xFFFFFFFFFFFFFFF8LL) + 23) < 0 )
    {
      v21 = sub_1648A40(v7 & 0xFFFFFFFFFFFFFFF8LL);
      v22 = v7 & 0xFFFFFFFFFFFFFFF8LL;
      v24 = v21 + v23;
      if ( *(char *)((v7 & 0xFFFFFFFFFFFFFFF8LL) + 23) >= 0 )
      {
        v25 = v24 >> 4;
      }
      else
      {
        v22 = v7 & 0xFFFFFFFFFFFFFFF8LL;
        v25 = (v24 - sub_1648A40(v7 & 0xFFFFFFFFFFFFFFF8LL)) >> 4;
      }
      if ( (_DWORD)v25 )
      {
        v26 = 0;
        v27 = 16LL * (unsigned int)v25;
        do
        {
          v28 = 0;
          if ( *(char *)(v22 + 23) < 0 )
            v28 = sub_1648A40(v22);
          if ( *(_DWORD *)(*(_QWORD *)(v28 + v26) + 8LL) != 1 )
            return 0;
          v26 += 16;
        }
        while ( v26 != v27 );
      }
    }
  }
LABEL_7:
  if ( a1 + 80 != a2 + 48 )
  {
    v8 = *(_QWORD *)(a1 + 80);
    if ( v8 )
      sub_161E7C0(a1 + 80, v8);
    v9 = *(_QWORD *)(a2 + 48);
    *(_QWORD *)(a1 + 80) = v9;
    if ( v9 )
      sub_1623A60(a1 + 80, v9, 2);
  }
  *(_QWORD *)(a1 + 168) = *(_QWORD *)(*(_QWORD *)(a1 + 40) + 792LL);
  if ( *(_BYTE *)(a2 + 16) == 78 )
  {
    v10 = *(_QWORD *)(a2 - 24);
    if ( !*(_BYTE *)(v10 + 16) )
    {
      if ( (*(_BYTE *)(v10 + 32) & 0xFu) - 7 > 1 && (*(_BYTE *)(v10 + 23) & 0x20) != 0 )
      {
        v38 = *(__int64 **)(a1 + 128);
        v29 = (char *)sub_1649960(*(_QWORD *)(a2 - 24));
        if ( (unsigned __int8)sub_149B630(*v38, v29, v30, &v39) )
        {
          a4 = 2 * (v39 & 3u);
          if ( (((int)*(unsigned __int8 *)(**(_QWORD **)(a1 + 128) + v39 / 4) >> (2 * (v39 & 3))) & 3) != 0 )
          {
            switch ( v39 )
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
                return 0;
              default:
                break;
            }
          }
        }
      }
      if ( *(_DWORD *)(v10 + 36) == 205 )
      {
        if ( sub_15602A0((_QWORD *)(a2 + 56), -1, "trap-func-name", 0xEu) )
          return 0;
        v31 = *(_QWORD *)(a2 - 24);
        if ( !*(_BYTE *)(v31 + 16) )
        {
          v40[0] = *(_QWORD *)(v31 + 112);
          if ( sub_15602A0(v40, -1, "trap-func-name", 0xEu) )
            return 0;
        }
      }
    }
  }
  if ( !*(_BYTE *)(a1 + 136) )
  {
    LOBYTE(v16) = sub_1FD8350((_QWORD *)a1, a2, *(unsigned __int8 *)(a2 + 16) - 24, a4);
    v6 = v16;
    if ( (_BYTE)v16 )
      goto LABEL_30;
    sub_1FD3A30(a1);
    v17 = *(_QWORD *)(a1 + 40);
    v18 = *(_BYTE **)(a1 + 168);
    if ( v18 != *(_BYTE **)(v17 + 792) )
    {
      sub_1FD3B40(a1, *(_BYTE **)(v17 + 792), v18);
      v17 = *(_QWORD *)(a1 + 40);
    }
    *(_QWORD *)(a1 + 168) = *(_QWORD *)(v17 + 792);
  }
  v6 = (*(__int64 (__fastcall **)(__int64, __int64))(*(_QWORD *)a1 + 24LL))(a1, a2);
  if ( !(_BYTE)v6 )
  {
    sub_1FD3A30(a1);
    v11 = *(_QWORD *)(a1 + 40);
    v12 = *(_BYTE **)(a1 + 168);
    if ( v12 != *(_BYTE **)(v11 + 792) )
      sub_1FD3B40(a1, *(_BYTE **)(v11 + 792), v12);
    v13 = *(_QWORD *)(a1 + 80);
    v40[0] = 0;
    if ( v13 )
    {
      sub_161E7C0(a1 + 80, v13);
      v14 = (unsigned __int8 *)v40[0];
      *(_QWORD *)(a1 + 80) = v40[0];
      if ( v14 )
        sub_1623210((__int64)v40, v14, a1 + 80);
    }
    if ( (unsigned int)*(unsigned __int8 *)(a2 + 16) - 25 <= 9 )
    {
      sub_1FD3D60((_QWORD *)a1, v37);
      v32 = *(_QWORD *)(a1 + 40);
      v33 = *(_QWORD *)(v32 + 904);
      v34 = *(unsigned int *)(v32 + 928);
      v35 = (*(_QWORD *)(v32 + 912) - v33) >> 4;
      if ( v34 > v35 )
      {
        sub_1FD4090((const __m128i **)(v32 + 904), v34 - v35);
        return v6;
      }
      if ( v34 < v35 )
      {
        v36 = v33 + 16 * v34;
        if ( *(_QWORD *)(v32 + 912) != v36 )
        {
          *(_QWORD *)(v32 + 912) = v36;
          return v6;
        }
      }
    }
    return 0;
  }
LABEL_30:
  v19 = *(_QWORD *)(a1 + 80);
  v40[0] = 0;
  if ( v19 )
  {
    sub_161E7C0(a1 + 80, v19);
    v20 = (unsigned __int8 *)v40[0];
    *(_QWORD *)(a1 + 80) = v40[0];
    if ( v20 )
      sub_1623210((__int64)v40, v20, a1 + 80);
  }
  return v6;
}
