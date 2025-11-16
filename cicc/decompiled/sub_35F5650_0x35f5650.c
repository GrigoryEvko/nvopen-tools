// Function: sub_35F5650
// Address: 0x35f5650
//
void __fastcall sub_35F5650(__int64 a1, __int64 a2, unsigned int a3, _QWORD *a4, __int64 a5)
{
  unsigned __int64 v8; // r12
  __int64 v9; // r15
  size_t v10; // rax
  __int64 v11; // r8
  __int64 v12; // r9
  void *v13; // rdx
  _QWORD *v14; // rdx
  _WORD *v15; // rdx
  __int64 v16; // rdx
  __int64 v17; // rdx
  __m128i *v18; // rdx
  unsigned __int64 v19; // rax
  __int64 v20; // rdx
  __int64 v21; // rdx
  __int16 v22; // ax
  _QWORD *v23; // rdx
  __int64 v24; // rdx
  __int64 v25; // rdx
  __int64 v26; // rdx
  void *v27; // rdx
  _QWORD *v28; // rdx
  _QWORD *v29; // rdx
  _QWORD *v30; // rdx
  _QWORD *v31; // rdx
  _QWORD *v32; // rdx
  __m128i *v33; // rdx
  __m128i si128; // xmm0
  __m128i *v35; // rdx
  __m128i *v36; // rdx
  __m128i v37; // xmm0
  __m128i *v38; // rdx
  void *v39; // rdx
  __int64 v40; // rdx
  __int64 v41; // rdx
  __int64 v42; // rdx
  __int64 v43; // rdx
  __int64 v44; // rdx
  __int64 v45; // rdx
  __int64 v46; // rdx
  size_t v47; // rdx
  char *v48; // rsi

  if ( !a5 )
    sub_C64ED0("Empty modifier in Load/StoreExtVer2 Instructions.", 1u);
  v8 = sub_CE1180(*(_QWORD *)(*(_QWORD *)(a2 + 16) + 16LL * a3 + 8));
  v9 = *(_QWORD *)(*(_QWORD *)(a2 + 16) + 16LL * (a3 + 1) + 8);
  v10 = strlen((const char *)a5);
  switch ( v10 )
  {
    case 2uLL:
      if ( *(_WORD *)a5 == 29555 )
      {
        switch ( (int)v9 )
        {
          case 0:
            return;
          case 1:
            v20 = a4[4];
            if ( (unsigned __int64)(a4[3] - v20) <= 6 )
            {
              sub_CB6200((__int64)a4, (unsigned __int8 *)".global", 7u);
            }
            else
            {
              *(_DWORD *)v20 = 1869375278;
              *(_WORD *)(v20 + 4) = 24930;
              *(_BYTE *)(v20 + 6) = 108;
              a4[4] += 7LL;
            }
            return;
          case 2:
            goto LABEL_134;
          case 3:
            v18 = (__m128i *)a4[4];
            v19 = a4[3] - (_QWORD)v18;
            if ( (BYTE4(v8) & 0x60) == 0x20 )
            {
              if ( v19 <= 0xF )
              {
                sub_CB6200((__int64)a4, (unsigned __int8 *)".shared::cluster", 0x10u);
              }
              else
              {
                *v18 = _mm_load_si128((const __m128i *)&xmmword_44FE820);
                a4[4] += 16LL;
              }
            }
            else if ( v19 <= 0xB )
            {
              sub_CB6200((__int64)a4, (unsigned __int8 *)".shared::cta", 0xCu);
            }
            else
            {
              qmemcpy(v18, ".shared::cta", 12);
              a4[4] += 12LL;
            }
            return;
          case 4:
            v21 = a4[4];
            if ( (unsigned __int64)(a4[3] - v21) <= 5 )
            {
              sub_CB6200((__int64)a4, (unsigned __int8 *)".const", 6u);
            }
            else
            {
              *(_DWORD *)v21 = 1852793646;
              *(_WORD *)(v21 + 4) = 29811;
              a4[4] += 6LL;
            }
            return;
          case 5:
            v17 = a4[4];
            if ( (unsigned __int64)(a4[3] - v17) <= 5 )
            {
              sub_CB6200((__int64)a4, (unsigned __int8 *)".local", 6u);
            }
            else
            {
              *(_DWORD *)v17 = 1668246574;
              *(_WORD *)(v17 + 4) = 27745;
              a4[4] += 6LL;
            }
            return;
          default:
            if ( (_DWORD)v9 != 101 )
              goto LABEL_134;
            v16 = a4[4];
            if ( (unsigned __int64)(a4[3] - v16) <= 5 )
            {
              sub_CB6200((__int64)a4, (unsigned __int8 *)".param", 6u);
            }
            else
            {
              *(_DWORD *)v16 = 1918988334;
              *(_WORD *)(v16 + 4) = 28001;
              a4[4] += 6LL;
            }
            break;
        }
      }
      break;
    case 9uLL:
      if ( *(_QWORD *)a5 == 0x6564726F5F6D656DLL && *(_BYTE *)(a5 + 8) == 114 )
      {
        switch ( (v8 >> 41) & 0xF )
        {
          case 1uLL:
            v32 = (_QWORD *)a4[4];
            if ( a4[3] - (_QWORD)v32 <= 7u )
            {
              sub_CB6200((__int64)a4, (unsigned __int8 *)".relaxed", 8u);
            }
            else
            {
              *v32 = 0x646578616C65722ELL;
              a4[4] += 8LL;
            }
            goto LABEL_64;
          case 2uLL:
            v31 = (_QWORD *)a4[4];
            if ( a4[3] - (_QWORD)v31 <= 7u )
            {
              sub_CB6200((__int64)a4, ".acquire", 8u);
            }
            else
            {
              *v31 = 0x657269757163612ELL;
              a4[4] += 8LL;
            }
            goto LABEL_64;
          case 3uLL:
            v30 = (_QWORD *)a4[4];
            if ( a4[3] - (_QWORD)v30 <= 7u )
            {
              sub_CB6200((__int64)a4, (unsigned __int8 *)".release", 8u);
            }
            else
            {
              *v30 = 0x657361656C65722ELL;
              a4[4] += 8LL;
            }
            goto LABEL_64;
          case 4uLL:
            v28 = (_QWORD *)a4[4];
            if ( a4[3] - (_QWORD)v28 <= 7u )
            {
              sub_CB6200((__int64)a4, ".acq_rel", 8u);
            }
            else
            {
              *v28 = 0x6C65725F7163612ELL;
              a4[4] += 8LL;
            }
            return;
          case 5uLL:
            v29 = (_QWORD *)a4[4];
            if ( a4[3] - (_QWORD)v29 <= 7u )
            {
              sub_CB6200((__int64)a4, ".seq_cst", 8u);
            }
            else
            {
              *v29 = 0x7473635F7165732ELL;
              a4[4] += 8LL;
            }
            return;
          case 6uLL:
            v27 = (void *)a4[4];
            if ( a4[3] - (_QWORD)v27 <= 0xCu )
            {
              sub_CB6200((__int64)a4, ".mmio.relaxed", 0xDu);
            }
            else
            {
              qmemcpy(v27, ".mmio.relaxed", 13);
              a4[4] += 13LL;
            }
LABEL_64:
            sub_35ED5A0((v8 >> 45) & 0xF, (__int64)a4);
            break;
          case 8uLL:
            v26 = a4[4];
            if ( (unsigned __int64)(a4[3] - v26) <= 8 )
            {
              sub_CB6200((__int64)a4, ".volatile", 9u);
            }
            else
            {
              *(_BYTE *)(v26 + 8) = 101;
              *(_QWORD *)v26 = 0x6C6974616C6F762ELL;
              a4[4] += 9LL;
            }
            break;
          default:
            return;
        }
      }
      return;
    case 3uLL:
      if ( *(_WORD *)a5 == 28515 && *(_BYTE *)(a5 + 2) == 112 )
      {
        switch ( (v8 >> 49) & 0xF )
        {
          case 0uLL:
            v46 = a4[4];
            if ( (unsigned __int64)(a4[3] - v46) <= 2 )
            {
              sub_CB6200((__int64)a4, byte_435F090, 3u);
            }
            else
            {
              *(_BYTE *)(v46 + 2) = 103;
              *(_WORD *)v46 = 25390;
              a4[4] += 3LL;
            }
            return;
          case 1uLL:
            v44 = a4[4];
            if ( (unsigned __int64)(a4[3] - v44) <= 2 )
            {
              sub_CB6200((__int64)a4, ".cs", 3u);
            }
            else
            {
              *(_BYTE *)(v44 + 2) = 115;
              *(_WORD *)v44 = 25390;
              a4[4] += 3LL;
            }
            return;
          case 2uLL:
            v45 = a4[4];
            if ( (unsigned __int64)(a4[3] - v45) <= 2 )
            {
              sub_CB6200((__int64)a4, byte_435F08C, 3u);
            }
            else
            {
              *(_BYTE *)(v45 + 2) = 97;
              *(_WORD *)v45 = 25390;
              a4[4] += 3LL;
            }
            return;
          case 3uLL:
            v43 = a4[4];
            if ( (unsigned __int64)(a4[3] - v43) <= 2 )
            {
              sub_CB6200((__int64)a4, ".lu", 3u);
            }
            else
            {
              *(_BYTE *)(v43 + 2) = 117;
              *(_WORD *)v43 = 27694;
              a4[4] += 3LL;
            }
            return;
          case 4uLL:
            v42 = a4[4];
            if ( (unsigned __int64)(a4[3] - v42) <= 2 )
            {
              sub_CB6200((__int64)a4, ".cv", 3u);
            }
            else
            {
              *(_BYTE *)(v42 + 2) = 118;
              *(_WORD *)v42 = 25390;
              a4[4] += 3LL;
            }
            return;
          case 5uLL:
          case 8uLL:
          case 9uLL:
          case 0xAuLL:
          case 0xBuLL:
          case 0xCuLL:
          case 0xDuLL:
          case 0xEuLL:
            goto LABEL_134;
          case 6uLL:
            v41 = a4[4];
            if ( (unsigned __int64)(a4[3] - v41) <= 2 )
            {
              sub_CB6200((__int64)a4, ".wb", 3u);
            }
            else
            {
              *(_BYTE *)(v41 + 2) = 98;
              *(_WORD *)v41 = 30510;
              a4[4] += 3LL;
            }
            return;
          case 7uLL:
            v40 = a4[4];
            if ( (unsigned __int64)(a4[3] - v40) <= 2 )
            {
              sub_CB6200((__int64)a4, ".wt", 3u);
            }
            else
            {
              *(_BYTE *)(v40 + 2) = 116;
              *(_WORD *)v40 = 30510;
              a4[4] += 3LL;
            }
            return;
          case 0xFuLL:
            return;
        }
      }
      return;
    case 5uLL:
      if ( *(_DWORD *)a5 == 1700737388 && *(_BYTE *)(a5 + 4) == 112 )
      {
        switch ( BYTE2(v8) & 0xF )
        {
          case 0:
            break;
          case 1:
          case 5:
            v33 = (__m128i *)a4[4];
            if ( a4[3] - (_QWORD)v33 <= 0x10u )
            {
              sub_CB6200((__int64)a4, ".L1::evict_normal", 0x11u);
            }
            else
            {
              si128 = _mm_load_si128((const __m128i *)&xmmword_44FE8C0);
              v33[1].m128i_i8[0] = 108;
              *v33 = si128;
              a4[4] += 17LL;
            }
            break;
          case 2:
            v39 = (void *)a4[4];
            if ( a4[3] - (_QWORD)v39 <= 0xEu )
            {
              sub_CB6200((__int64)a4, ".L1::evict_last", 0xFu);
            }
            else
            {
              qmemcpy(v39, ".L1::evict_last", 15);
              a4[4] += 15LL;
            }
            break;
          case 3:
            v38 = (__m128i *)a4[4];
            if ( a4[3] - (_QWORD)v38 <= 0xFu )
            {
              sub_CB6200((__int64)a4, ".L1::evict_first", 0x10u);
            }
            else
            {
              *v38 = _mm_load_si128((const __m128i *)&xmmword_44FE8D0);
              a4[4] += 16LL;
            }
            break;
          case 4:
            v36 = (__m128i *)a4[4];
            if ( a4[3] - (_QWORD)v36 <= 0x13u )
            {
              sub_CB6200((__int64)a4, ".L1::evict_unchanged", 0x14u);
            }
            else
            {
              v37 = _mm_load_si128((const __m128i *)&xmmword_44FE8E0);
              v36[1].m128i_i32[0] = 1684367214;
              *v36 = v37;
              a4[4] += 20LL;
            }
            break;
          case 6:
            v35 = (__m128i *)a4[4];
            if ( a4[3] - (_QWORD)v35 <= 0xFu )
            {
              sub_CB6200((__int64)a4, ".L1::no_allocate", 0x10u);
            }
            else
            {
              *v35 = _mm_load_si128((const __m128i *)&xmmword_44FE8F0);
              a4[4] += 16LL;
            }
            break;
          default:
            goto LABEL_134;
        }
      }
      if ( *(_DWORD *)a5 == 1885287020 && *(_BYTE *)(a5 + 4) == 115 )
      {
        v22 = (unsigned __int16)v8 >> 12;
        if ( (unsigned __int16)v8 >> 12 == 3 )
        {
          v24 = a4[4];
          if ( (unsigned __int64)(a4[3] - v24) <= 8 )
          {
            sub_CB6200((__int64)a4, ".L2::128B", 9u);
          }
          else
          {
            *(_BYTE *)(v24 + 8) = 66;
            *(_QWORD *)v24 = 0x3832313A3A324C2ELL;
            a4[4] += 9LL;
          }
        }
        else if ( (unsigned __int8)v22 > 3u )
        {
          if ( (_BYTE)v22 != 4 )
LABEL_134:
            BUG();
          v25 = a4[4];
          if ( (unsigned __int64)(a4[3] - v25) <= 8 )
          {
            sub_CB6200((__int64)a4, ".L2::256B", 9u);
          }
          else
          {
            *(_BYTE *)(v25 + 8) = 66;
            *(_QWORD *)v25 = 0x3635323A3A324C2ELL;
            a4[4] += 9LL;
          }
        }
        else if ( (_BYTE)v22 == 2 )
        {
          v23 = (_QWORD *)a4[4];
          if ( a4[3] - (_QWORD)v23 <= 7u )
          {
            sub_CB6200((__int64)a4, ".L2::64B", 8u);
          }
          else
          {
            *v23 = 0x4234363A3A324C2ELL;
            a4[4] += 8LL;
          }
        }
      }
      if ( *(_DWORD *)a5 == 1667183212 && *(_BYTE *)(a5 + 4) == 104 && (WORD2(v8) & 0x180) == 0x80 )
      {
        v13 = (void *)a4[4];
        if ( a4[3] - (_QWORD)v13 > 0xEu )
        {
          qmemcpy(v13, ".L2::cache_hint", 15);
          a4[4] += 15LL;
          return;
        }
        v47 = 15;
        v48 = ".L2::cache_hint";
        goto LABEL_106;
      }
      break;
    case 7uLL:
      if ( *(_DWORD *)a5 != 1718185589
        || *(_WORD *)(a5 + 4) != 25961
        || *(_BYTE *)(a5 + 6) != 100
        || (v8 & 0x1000000000LL) == 0 )
      {
        return;
      }
      v14 = (_QWORD *)a4[4];
      if ( a4[3] - (_QWORD)v14 > 7u )
      {
        *v14 = 0x64656966696E752ELL;
        a4[4] += 8LL;
        return;
      }
      v47 = 8;
      v48 = ".unified";
LABEL_106:
      sub_CB6200((__int64)a4, (unsigned __int8 *)v48, v47);
      return;
    default:
      if ( v10 == 4 && *(_DWORD *)a5 == 1668506980 && (WORD2(v8) & 0x180) == 0x80 )
      {
        v15 = (_WORD *)a4[4];
        if ( a4[3] - (_QWORD)v15 <= 1u )
        {
          sub_CB6200((__int64)a4, (unsigned __int8 *)", ", 2u);
        }
        else
        {
          *v15 = 8236;
          a4[4] += 2LL;
        }
        sub_35EE840(a1, a2, *(_DWORD *)(a2 + 24) - 1, a4, v11, v12);
      }
      break;
  }
}
