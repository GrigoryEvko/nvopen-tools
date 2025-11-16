// Function: sub_2356B40
// Address: 0x2356b40
//
void __fastcall sub_2356B40(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int128 a7,
        __int64 a8,
        int a9)
{
  __int64 v10; // rax
  __m128i v11; // xmm0
  void (*v12)(); // rax
  __int64 v13; // rax
  int v14; // edx
  __int64 *v15; // rax
  __m128i *v16; // r14
  __m128i *v17; // rax
  unsigned __int64 v18; // rdi
  int v19; // ebx
  unsigned __int64 v20[7]; // [rsp+8h] [rbp-38h] BYREF

  v10 = a8;
  v11 = _mm_loadu_si128((const __m128i *)&a7);
  *(_QWORD *)a1 = a2;
  *(_BYTE *)(a1 + 192) = 0;
  *(_QWORD *)(a1 + 24) = v10;
  LODWORD(v10) = a9;
  *(__m128i *)(a1 + 8) = v11;
  *(_DWORD *)(a1 + 32) = v10;
  if ( *(_BYTE *)(a3 + 152) )
  {
    sub_23C6530(a1 + 40, a3);
    *(_BYTE *)(a1 + 192) = 1;
  }
  *(_QWORD *)(a1 + 200) = a4;
  *(_QWORD *)(a1 + 288) = a1 + 304;
  *(_QWORD *)(a1 + 368) = a1 + 384;
  *(_QWORD *)(a1 + 448) = a1 + 464;
  *(_QWORD *)(a1 + 528) = a1 + 544;
  *(_QWORD *)(a1 + 608) = a1 + 624;
  *(_QWORD *)(a1 + 688) = a1 + 704;
  *(_QWORD *)(a1 + 768) = a1 + 784;
  *(_QWORD *)(a1 + 848) = a1 + 864;
  *(_QWORD *)(a1 + 928) = a1 + 944;
  *(_QWORD *)(a1 + 1008) = a1 + 1024;
  *(_QWORD *)(a1 + 1088) = a1 + 1104;
  *(_QWORD *)(a1 + 1168) = a1 + 1184;
  *(_QWORD *)(a1 + 1248) = a1 + 1264;
  *(_QWORD *)(a1 + 1328) = a1 + 1344;
  *(_QWORD *)(a1 + 208) = a1 + 224;
  *(_QWORD *)(a1 + 1408) = a1 + 1424;
  *(_QWORD *)(a1 + 216) = 0x200000000LL;
  *(_QWORD *)(a1 + 296) = 0x200000000LL;
  *(_QWORD *)(a1 + 376) = 0x200000000LL;
  *(_QWORD *)(a1 + 456) = 0x200000000LL;
  *(_QWORD *)(a1 + 536) = 0x200000000LL;
  *(_QWORD *)(a1 + 616) = 0x200000000LL;
  *(_QWORD *)(a1 + 696) = 0x200000000LL;
  *(_QWORD *)(a1 + 776) = 0x200000000LL;
  *(_QWORD *)(a1 + 856) = 0x200000000LL;
  *(_QWORD *)(a1 + 936) = 0x200000000LL;
  *(_QWORD *)(a1 + 1016) = 0x200000000LL;
  *(_QWORD *)(a1 + 1096) = 0x200000000LL;
  *(_QWORD *)(a1 + 1176) = 0x200000000LL;
  *(_QWORD *)(a1 + 1256) = 0x200000000LL;
  *(_QWORD *)(a1 + 1336) = 0x200000000LL;
  *(_QWORD *)(a1 + 1416) = 0x200000000LL;
  *(_QWORD *)(a1 + 1488) = a1 + 1504;
  *(_QWORD *)(a1 + 1568) = a1 + 1584;
  *(_QWORD *)(a1 + 1648) = a1 + 1664;
  *(_QWORD *)(a1 + 1728) = a1 + 1744;
  *(_QWORD *)(a1 + 1808) = a1 + 1824;
  *(_QWORD *)(a1 + 1888) = a1 + 1904;
  *(_QWORD *)(a1 + 1968) = a1 + 1984;
  *(_QWORD *)(a1 + 2048) = a1 + 2064;
  *(_QWORD *)(a1 + 2128) = a1 + 2144;
  *(_QWORD *)(a1 + 1496) = 0x200000000LL;
  *(_QWORD *)(a1 + 1576) = 0x200000000LL;
  *(_QWORD *)(a1 + 1656) = 0x200000000LL;
  *(_QWORD *)(a1 + 1736) = 0x200000000LL;
  *(_QWORD *)(a1 + 1816) = 0x200000000LL;
  *(_QWORD *)(a1 + 1896) = 0x200000000LL;
  *(_QWORD *)(a1 + 1976) = 0x200000000LL;
  *(_QWORD *)(a1 + 2056) = 0x200000000LL;
  *(_QWORD *)(a1 + 2136) = 0x200000000LL;
  *(_QWORD *)(a1 + 2208) = a1 + 2224;
  *(_QWORD *)(a1 + 2216) = 0x200000000LL;
  if ( a2 )
  {
    v12 = *(void (**)())(*(_QWORD *)a2 + 112LL);
    if ( v12 != nullsub_829 )
      ((void (__fastcall *)(__int64, __int64))v12)(a2, a1);
  }
  if ( a4 )
  {
    v13 = *(unsigned int *)(a4 + 1304);
    v14 = v13;
    if ( *(_DWORD *)(a4 + 1308) <= (unsigned int)v13 )
    {
      v16 = (__m128i *)sub_C8D7D0(a4 + 1296, a4 + 1312, 0, 0x20u, v20, a6);
      v17 = &v16[2 * *(unsigned int *)(a4 + 1304)];
      if ( v17 )
      {
        v17->m128i_i64[0] = a1;
        v17->m128i_i64[1] = a4;
        v17[1].m128i_i64[1] = (__int64)&off_4CDFBF0 + 2;
      }
      sub_2356A00(a4 + 1296, v16);
      v18 = *(_QWORD *)(a4 + 1296);
      v19 = v20[0];
      if ( a4 + 1312 != v18 )
        _libc_free(v18);
      ++*(_DWORD *)(a4 + 1304);
      *(_QWORD *)(a4 + 1296) = v16;
      *(_DWORD *)(a4 + 1308) = v19;
    }
    else
    {
      v15 = (__int64 *)(*(_QWORD *)(a4 + 1296) + 32 * v13);
      if ( v15 )
      {
        *v15 = a1;
        v15[1] = a4;
        v15[3] = (__int64)&off_4CDFBF0 + 2;
        v14 = *(_DWORD *)(a4 + 1304);
      }
      *(_DWORD *)(a4 + 1304) = v14 + 1;
    }
  }
}
