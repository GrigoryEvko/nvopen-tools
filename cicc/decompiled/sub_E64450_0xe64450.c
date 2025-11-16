// Function: sub_E64450
// Address: 0xe64450
//
__int64 __fastcall sub_E64450(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int64 a7,
        char a8,
        __int128 a9)
{
  __int64 *v14; // rdi
  __int64 v15; // rax
  _BYTE *v16; // rsi
  __int64 v17; // rdx
  __int64 v18; // r9
  __int64 v19; // rax
  __int64 *v20; // rdx
  __int64 v21; // rdi
  char *v22; // rsi
  __int64 v23; // rdx
  char *(*v24)(); // rax
  unsigned __int8 *v25; // rdi
  __int64 v26; // rdx
  size_t v27; // rcx
  __int64 v28; // rsi
  __int64 result; // rax
  __int64 v30; // kr00_8
  size_t v31; // rdx
  unsigned __int8 *v33; // [rsp+10h] [rbp-50h] BYREF
  size_t n; // [rsp+18h] [rbp-48h]
  unsigned __int8 src[64]; // [rsp+20h] [rbp-40h] BYREF

  v14 = (__int64 *)(a1 + 24);
  *((__m128i *)v14 - 1) = _mm_loadu_si128((const __m128i *)&a9);
  *(_QWORD *)(a1 + 24) = a1 + 40;
  sub_E62C60(v14, *(_BYTE **)a2, *(_QWORD *)a2 + *(_QWORD *)(a2 + 8));
  *(_QWORD *)(a1 + 56) = *(_QWORD *)(a2 + 32);
  *(_QWORD *)(a1 + 64) = *(_QWORD *)(a2 + 40);
  v15 = *(_QWORD *)(a2 + 48);
  *(_QWORD *)(a1 + 256) = a1 + 272;
  *(_QWORD *)(a1 + 72) = v15;
  *(_QWORD *)(a1 + 120) = sub_E62B10;
  *(_QWORD *)(a1 + 80) = a6;
  *(_QWORD *)(a1 + 144) = sub_E62AC0;
  *(_QWORD *)(a1 + 176) = a5;
  *(_QWORD *)(a1 + 136) = sub_E62AE0;
  *(_QWORD *)(a1 + 208) = a1 + 224;
  *(_QWORD *)(a1 + 216) = 0x400000000LL;
  *(_QWORD *)(a1 + 88) = 0;
  *(_QWORD *)(a1 + 96) = 0;
  *(_QWORD *)(a1 + 104) = 0;
  *(_QWORD *)(a1 + 112) = 0;
  *(_QWORD *)(a1 + 152) = a3;
  *(_QWORD *)(a1 + 160) = a4;
  *(_QWORD *)(a1 + 168) = 0;
  *(_QWORD *)(a1 + 184) = 0;
  *(_QWORD *)(a1 + 192) = 0;
  *(_QWORD *)(a1 + 200) = 0;
  *(_QWORD *)(a1 + 264) = 0;
  *(_QWORD *)(a1 + 272) = 0;
  *(_QWORD *)(a1 + 280) = 1;
  *(_QWORD *)(a1 + 288) = 0;
  *(_QWORD *)(a1 + 296) = 0;
  *(_QWORD *)(a1 + 304) = a1 + 320;
  *(_QWORD *)(a1 + 352) = a1 + 368;
  *(_QWORD *)(a1 + 400) = a1 + 416;
  *(_QWORD *)(a1 + 448) = a1 + 464;
  *(_QWORD *)(a1 + 496) = a1 + 512;
  *(_QWORD *)(a1 + 544) = a1 + 560;
  *(_QWORD *)(a1 + 592) = a1 + 608;
  *(_QWORD *)(a1 + 640) = a1 + 656;
  *(_QWORD *)(a1 + 688) = a1 + 704;
  *(_QWORD *)(a1 + 312) = 0x400000000LL;
  *(_QWORD *)(a1 + 408) = 0x400000000LL;
  *(_QWORD *)(a1 + 504) = 0x400000000LL;
  *(_QWORD *)(a1 + 600) = 0x400000000LL;
  *(_QWORD *)(a1 + 360) = 0;
  *(_QWORD *)(a1 + 368) = 0;
  *(_QWORD *)(a1 + 376) = 1;
  *(_QWORD *)(a1 + 384) = 0;
  *(_QWORD *)(a1 + 392) = 0;
  *(_QWORD *)(a1 + 456) = 0;
  *(_QWORD *)(a1 + 464) = 0;
  *(_QWORD *)(a1 + 472) = 0;
  *(_QWORD *)(a1 + 480) = 0;
  *(_QWORD *)(a1 + 488) = 0;
  *(_QWORD *)(a1 + 552) = 0;
  *(_QWORD *)(a1 + 560) = 0;
  *(_QWORD *)(a1 + 568) = 0;
  *(_QWORD *)(a1 + 576) = 0;
  *(_QWORD *)(a1 + 584) = 0;
  *(_QWORD *)(a1 + 648) = 0;
  *(_QWORD *)(a1 + 656) = 0;
  *(_QWORD *)(a1 + 664) = 0;
  *(_QWORD *)(a1 + 672) = 0;
  *(_QWORD *)(a1 + 680) = 0;
  *(_QWORD *)(a1 + 696) = 0x400000000LL;
  *(_QWORD *)(a1 + 736) = a1 + 752;
  *(_QWORD *)(a1 + 784) = a1 + 800;
  *(_QWORD *)(a1 + 832) = a1 + 848;
  *(_QWORD *)(a1 + 880) = a1 + 896;
  *(_QWORD *)(a1 + 928) = a1 + 944;
  *(_QWORD *)(a1 + 976) = a1 + 992;
  *(_QWORD *)(a1 + 1024) = a1 + 1040;
  *(_QWORD *)(a1 + 1072) = a1 + 1088;
  *(_QWORD *)(a1 + 792) = 0x400000000LL;
  *(_QWORD *)(a1 + 888) = 0x400000000LL;
  *(_QWORD *)(a1 + 984) = 0x400000000LL;
  *(_QWORD *)(a1 + 1080) = 0x400000000LL;
  *(_QWORD *)(a1 + 744) = 0;
  *(_QWORD *)(a1 + 752) = 0;
  *(_QWORD *)(a1 + 760) = 0;
  *(_QWORD *)(a1 + 768) = 0;
  *(_QWORD *)(a1 + 776) = 0;
  *(_QWORD *)(a1 + 840) = 0;
  *(_QWORD *)(a1 + 848) = 0;
  *(_QWORD *)(a1 + 856) = 0;
  *(_QWORD *)(a1 + 864) = 0;
  *(_QWORD *)(a1 + 872) = 0;
  *(_QWORD *)(a1 + 936) = 0;
  *(_QWORD *)(a1 + 944) = 0;
  *(_QWORD *)(a1 + 952) = 0;
  *(_QWORD *)(a1 + 960) = 0;
  *(_QWORD *)(a1 + 968) = 0;
  *(_QWORD *)(a1 + 1032) = 0;
  *(_QWORD *)(a1 + 1040) = 0;
  *(_QWORD *)(a1 + 1048) = 0;
  *(_QWORD *)(a1 + 1056) = 0;
  *(_QWORD *)(a1 + 1064) = 0;
  *(_QWORD *)(a1 + 1120) = a1 + 1136;
  *(_QWORD *)(a1 + 1168) = a1 + 1184;
  *(_QWORD *)(a1 + 1216) = a1 + 1232;
  *(_QWORD *)(a1 + 1264) = a1 + 1280;
  *(_QWORD *)(a1 + 1312) = a1 + 1328;
  *(_QWORD *)(a1 + 1368) = a1 + 192;
  *(_QWORD *)(a1 + 1432) = a1 + 192;
  *(_QWORD *)(a1 + 1176) = 0x400000000LL;
  *(_QWORD *)(a1 + 1272) = 0x400000000LL;
  *(_QWORD *)(a1 + 1360) = 0x1800000000LL;
  *(_QWORD *)(a1 + 1424) = 0x1000000000LL;
  *(_QWORD *)(a1 + 1128) = 0;
  *(_QWORD *)(a1 + 1136) = 0;
  *(_QWORD *)(a1 + 1144) = 0;
  *(_QWORD *)(a1 + 1152) = 0;
  *(_QWORD *)(a1 + 1160) = 0;
  *(_QWORD *)(a1 + 1224) = 0;
  *(_QWORD *)(a1 + 1232) = 0;
  *(_QWORD *)(a1 + 1240) = 0;
  *(_QWORD *)(a1 + 1248) = 0;
  *(_QWORD *)(a1 + 1256) = 0;
  *(_QWORD *)(a1 + 1320) = 0;
  *(_QWORD *)(a1 + 1328) = 0;
  *(_QWORD *)(a1 + 1336) = 0;
  *(_QWORD *)(a1 + 1344) = 0;
  *(_QWORD *)(a1 + 1352) = 0;
  *(_QWORD *)(a1 + 1376) = 0;
  *(_QWORD *)(a1 + 1384) = 0;
  *(_QWORD *)(a1 + 1392) = 0;
  *(_DWORD *)(a1 + 1400) = 0;
  *(_QWORD *)(a1 + 1408) = 0;
  *(_QWORD *)(a1 + 1416) = 0;
  *(_QWORD *)(a1 + 1440) = 0;
  *(_DWORD *)(a1 + 1464) = 0;
  *(_QWORD *)(a1 + 1480) = a1 + 1496;
  *(_QWORD *)(a1 + 1528) = a1 + 1552;
  *(_QWORD *)(a1 + 1680) = a1 + 1696;
  *(_QWORD *)(a1 + 1752) = a1 + 1736;
  *(_QWORD *)(a1 + 1760) = a1 + 1736;
  *(_QWORD *)(a1 + 1832) = a1 + 1848;
  *(_WORD *)(a1 + 1792) = 0;
  *(_QWORD *)(a1 + 1448) = 0;
  *(_DWORD *)(a1 + 1456) = 0;
  *(_DWORD *)(a1 + 1460) = 0;
  *(_BYTE *)(a1 + 1472) = 2;
  *(_QWORD *)(a1 + 1488) = 0;
  *(_BYTE *)(a1 + 1496) = 0;
  *(_QWORD *)(a1 + 1512) = 0;
  *(_BYTE *)(a1 + 1520) = 0;
  *(_QWORD *)(a1 + 1536) = 0;
  *(_QWORD *)(a1 + 1544) = 128;
  *(_QWORD *)(a1 + 1688) = 0;
  *(_QWORD *)(a1 + 1696) = a1 + 1712;
  *(_QWORD *)(a1 + 1704) = 0;
  *(_BYTE *)(a1 + 1712) = 0;
  *(_DWORD *)(a1 + 1736) = 0;
  *(_QWORD *)(a1 + 1744) = 0;
  *(_QWORD *)(a1 + 1768) = 0;
  *(_QWORD *)(a1 + 1776) = 0;
  *(_QWORD *)(a1 + 1784) = 0x10000;
  *(_QWORD *)(a1 + 1796) = 0;
  *(_QWORD *)(a1 + 1804) = 0;
  *(_QWORD *)(a1 + 1812) = 0;
  *(_QWORD *)(a1 + 1820) = 0;
  *(_QWORD *)(a1 + 1840) = 0;
  *(_QWORD *)(a1 + 1848) = 0;
  *(_QWORD *)(a1 + 1856) = 0;
  *(_QWORD *)(a1 + 1920) = a1 + 1968;
  *(_QWORD *)(a1 + 2024) = a1 + 2008;
  *(_QWORD *)(a1 + 2032) = a1 + 2008;
  *(_QWORD *)(a1 + 1992) = 0x1000000000LL;
  *(_QWORD *)(a1 + 2064) = 0x1000000000LL;
  *(_QWORD *)(a1 + 2096) = a1 + 2080;
  *(_QWORD *)(a1 + 2104) = a1 + 2080;
  *(_QWORD *)(a1 + 1864) = 0;
  *(_QWORD *)(a1 + 1872) = 0;
  *(_QWORD *)(a1 + 1880) = 0;
  *(_QWORD *)(a1 + 1888) = 0;
  *(_QWORD *)(a1 + 1896) = 0;
  *(_DWORD *)(a1 + 1904) = 4;
  *(_BYTE *)(a1 + 1908) = 0;
  *(_DWORD *)(a1 + 1912) = 0;
  *(_QWORD *)(a1 + 1928) = 1;
  *(_QWORD *)(a1 + 1936) = 0;
  *(_QWORD *)(a1 + 1944) = 0;
  *(_DWORD *)(a1 + 1952) = 1065353216;
  *(_QWORD *)(a1 + 1960) = 0;
  *(_QWORD *)(a1 + 1968) = 0;
  *(_QWORD *)(a1 + 1976) = 0;
  *(_QWORD *)(a1 + 1984) = 0;
  *(_DWORD *)(a1 + 2008) = 0;
  *(_QWORD *)(a1 + 2016) = 0;
  *(_QWORD *)(a1 + 2040) = 0;
  *(_QWORD *)(a1 + 2048) = 0;
  *(_QWORD *)(a1 + 2056) = 0;
  *(_DWORD *)(a1 + 2080) = 0;
  *(_QWORD *)(a1 + 2088) = 0;
  *(_QWORD *)(a1 + 2112) = 0;
  *(_DWORD *)(a1 + 2128) = 0;
  *(_QWORD *)(a1 + 2136) = 0;
  *(_QWORD *)(a1 + 2288) = 0x400000000LL;
  *(_QWORD *)(a1 + 2144) = a1 + 2128;
  *(_QWORD *)(a1 + 2152) = a1 + 2128;
  *(_QWORD *)(a1 + 2232) = 0x1000000000LL;
  *(_QWORD *)(a1 + 2256) = 0x1000000000LL;
  *(_QWORD *)(a1 + 2328) = a1 + 2344;
  *(_QWORD *)(a1 + 2160) = 0;
  *(_DWORD *)(a1 + 2176) = 0;
  *(_QWORD *)(a1 + 2184) = 0;
  *(_QWORD *)(a1 + 2192) = a1 + 2176;
  *(_QWORD *)(a1 + 2200) = a1 + 2176;
  *(_QWORD *)(a1 + 2208) = 0;
  *(_QWORD *)(a1 + 2216) = 0;
  *(_QWORD *)(a1 + 2224) = 0;
  *(_QWORD *)(a1 + 2240) = 0;
  *(_QWORD *)(a1 + 2248) = 0;
  *(_QWORD *)(a1 + 2264) = 0;
  *(_QWORD *)(a1 + 2272) = 0;
  *(_QWORD *)(a1 + 2280) = a1 + 2296;
  *(_QWORD *)(a1 + 2336) = 0;
  *(_QWORD *)(a1 + 2344) = 0;
  *(_QWORD *)(a1 + 2352) = 0;
  *(_BYTE *)(a1 + 2360) = a8;
  *(_QWORD *)(a1 + 2368) = a7;
  *(_BYTE *)(a1 + 2376) = 0;
  *(_QWORD *)(a1 + 2384) = 0;
  *(_QWORD *)(a1 + 2392) = 0;
  *(_QWORD *)(a1 + 2400) = 0x6000000000LL;
  *(_QWORD *)(a1 + 2408) = 0;
  *(_QWORD *)(a1 + 2416) = 0;
  *(_QWORD *)(a1 + 2424) = 0;
  *(_DWORD *)(a1 + 2432) = 0;
  *(_QWORD *)(a1 + 2440) = 0;
  *(_QWORD *)(a1 + 2448) = 0;
  *(_QWORD *)(a1 + 2456) = 0;
  *(_DWORD *)(a1 + 2464) = 0;
  if ( a7 )
  {
    if ( (*(_BYTE *)a7 & 0x40) != 0 )
      *(_BYTE *)(a1 + 1907) = 1;
    v16 = *(_BYTE **)(a7 + 128);
    v17 = *(_QWORD *)(a7 + 136);
    v33 = src;
    sub_E62C60((__int64 *)&v33, v16, (__int64)&v16[v17]);
    v18 = a1 + 1480;
  }
  else
  {
    v33 = src;
    sub_E62BB0((__int64 *)&v33, byte_3F871B3, (__int64)byte_3F871B3);
    v18 = a1 + 1480;
  }
  sub_2240AE0(v18, &v33);
  if ( v33 != src )
    j_j___libc_free_0(v33, *(_QWORD *)src + 1LL);
  v19 = *(_QWORD *)(a1 + 80);
  if ( v19 )
  {
    v20 = *(__int64 **)v19;
    v19 = -1431655765 * (unsigned int)((__int64)(*(_QWORD *)(v19 + 8) - *(_QWORD *)v19) >> 3);
    if ( (_DWORD)v19 )
    {
      v21 = *v20;
      v22 = "Unknown buffer";
      v23 = 14;
      v24 = *(char *(**)())(*(_QWORD *)v21 + 16LL);
      if ( v24 != sub_C1E8B0 )
        v22 = (char *)((__int64 (__fastcall *)(__int64, char *, __int64))v24)(v21, "Unknown buffer", 14);
      v33 = src;
      sub_E62BB0((__int64 *)&v33, v22, (__int64)&v22[v23]);
      v19 = (__int64)v33;
      v25 = *(unsigned __int8 **)(a1 + 1696);
      if ( v33 == src )
      {
        v31 = n;
        if ( n )
        {
          if ( n == 1 )
          {
            v19 = src[0];
            *v25 = src[0];
          }
          else
          {
            v19 = (__int64)memcpy(v25, src, n);
          }
          v31 = n;
          v25 = *(unsigned __int8 **)(a1 + 1696);
        }
        *(_QWORD *)(a1 + 1704) = v31;
        v25[v31] = 0;
        v25 = v33;
        goto LABEL_15;
      }
      v26 = *(_QWORD *)src;
      v27 = n;
      if ( (unsigned __int8 *)(a1 + 1712) == v25 )
      {
        *(_QWORD *)(a1 + 1696) = v33;
        *(_QWORD *)(a1 + 1704) = v27;
        *(_QWORD *)(a1 + 1712) = v26;
      }
      else
      {
        v28 = *(_QWORD *)(a1 + 1712);
        *(_QWORD *)(a1 + 1696) = v33;
        *(_QWORD *)(a1 + 1704) = v27;
        *(_QWORD *)(a1 + 1712) = v26;
        if ( v25 )
        {
          v33 = v25;
          *(_QWORD *)src = v28;
          goto LABEL_15;
        }
      }
      v33 = src;
      v25 = src;
LABEL_15:
      n = 0;
      *v25 = 0;
      if ( v33 != src )
        v19 = j_j___libc_free_0(v33, *(_QWORD *)src + 1LL);
    }
  }
  v30 = v19;
  result = *(unsigned int *)(a2 + 52);
  switch ( *(_DWORD *)(a2 + 52) )
  {
    case 0:
      sub_C64ED0("Cannot initialize MC for unknown object file format.", 1u);
    case 1:
      result = (unsigned int)(*(_DWORD *)(a2 + 44) - 13);
      if ( (unsigned int)result > 1 )
        sub_C64ED0("Cannot initialize MC for non-Windows COFF object files.", 1u);
      *(_DWORD *)a1 = 3;
      break;
    case 2:
      *(_DWORD *)a1 = 7;
      break;
    case 3:
      *(_DWORD *)a1 = 1;
      break;
    case 4:
      *(_DWORD *)a1 = 2;
      break;
    case 5:
      *(_DWORD *)a1 = 0;
      break;
    case 6:
      *(_DWORD *)a1 = 4;
      break;
    case 7:
      *(_DWORD *)a1 = 5;
      break;
    case 8:
      *(_DWORD *)a1 = 6;
      break;
    default:
      result = v30;
      break;
  }
  return result;
}
