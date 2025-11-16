// Function: sub_108D160
// Address: 0x108d160
//
__int64 (__fastcall **__fastcall sub_108D160(__int64 a1, _QWORD *a2, __int64 a3))()
{
  __int64 *v4; // r13
  __int64 *v5; // r15
  __int64 *v6; // r12
  __int64 *v7; // rbx
  __int64 v8; // rax
  __int64 v9; // rdi
  bool v10; // zf
  __int64 v11; // rax
  _QWORD **v12; // rax
  char *v13; // rbx
  char *v14; // r8
  char *v15; // rsi
  _QWORD *v16; // rcx
  _QWORD **v17; // rax
  char *v18; // r8
  char *v19; // rsi
  _QWORD *v20; // rcx
  _QWORD **v21; // rax
  char *v22; // r8
  char *v23; // rsi
  _QWORD *v24; // rcx
  _QWORD **v25; // rax
  char *v26; // r8
  char *v27; // rsi
  _QWORD *v28; // rcx
  _QWORD **v29; // rax
  char *v30; // rsi
  _QWORD *v31; // rcx
  void *v33; // rdi
  size_t v34; // rdx
  void *v35; // rdi
  size_t v36; // rdx
  void *v37; // rdi
  size_t v38; // rdx
  void *v39; // rdi
  size_t v40; // rdx
  void *v41; // rdi
  size_t v42; // rdx
  __int64 v43; // [rsp+18h] [rbp-B8h]
  __int64 v44; // [rsp+20h] [rbp-B0h]
  __int64 v45; // [rsp+28h] [rbp-A8h]
  __int64 *v46; // [rsp+30h] [rbp-A0h] BYREF
  __int64 *v47; // [rsp+38h] [rbp-98h] BYREF
  __int64 *v48; // [rsp+40h] [rbp-90h] BYREF
  _BYTE v49[8]; // [rsp+48h] [rbp-88h] BYREF
  __int64 v50; // [rsp+50h] [rbp-80h] BYREF
  __int64 v51; // [rsp+58h] [rbp-78h]
  __int64 v52; // [rsp+60h] [rbp-70h]
  __int64 v53; // [rsp+68h] [rbp-68h]
  __int64 v54; // [rsp+70h] [rbp-60h]
  _QWORD **v55; // [rsp+78h] [rbp-58h]
  __int64 v56; // [rsp+80h] [rbp-50h]
  void *dest; // [rsp+88h] [rbp-48h]
  __int64 v58; // [rsp+90h] [rbp-40h]
  unsigned __int64 v59; // [rsp+98h] [rbp-38h]

  v4 = (__int64 *)(a1 + 632);
  v5 = (__int64 *)(a1 + 712);
  v6 = (__int64 *)(a1 + 392);
  v7 = (__int64 *)(a1 + 472);
  *(_QWORD *)(a1 + 8) = a1 + 24;
  *(_QWORD *)(a1 + 24) = a1 + 40;
  *(_QWORD *)(a1 + 88) = a1 + 104;
  v8 = a1 + 120;
  v9 = a1 + 192;
  *(_QWORD *)(v9 - 88) = v8;
  *(_WORD *)(v9 - 112) = 0;
  *(_WORD *)(v9 - 40) = 0;
  *(_QWORD *)(v9 - 24) = a3;
  *(_QWORD *)(v9 - 176) = 0;
  *(_QWORD *)(v9 - 160) = 0;
  *(_BYTE *)(v9 - 152) = 0;
  *(_QWORD *)(v9 - 136) = 0;
  *(_QWORD *)(v9 - 128) = 0;
  *(_QWORD *)(v9 - 120) = 0;
  *(_QWORD *)(v9 - 96) = 0;
  *(_QWORD *)(v9 - 80) = 0;
  *(_BYTE *)(v9 - 72) = 0;
  *(_QWORD *)(v9 - 192) = off_49E6238;
  *(_DWORD *)(v9 - 56) = 0;
  *(_QWORD *)(v9 - 48) = 0;
  *(_DWORD *)(v9 - 36) = 0;
  *(_BYTE *)(v9 - 32) = 0;
  *(_DWORD *)(v9 - 16) = 0;
  *(_QWORD *)(v9 - 8) = *a2;
  *a2 = 0;
  sub_C0BFB0(v9, 8, 0);
  v10 = *(_BYTE *)(*(_QWORD *)(a1 + 184) + 8LL) == 0;
  v11 = -1;
  *(_QWORD *)(a1 + 248) = 0;
  *(_QWORD *)(a1 + 256) = 0;
  if ( v10 )
    v11 = 0xFFFFFFFFLL;
  *(_QWORD *)(a1 + 264) = 0;
  *(_QWORD *)(a1 + 240) = v11;
  *(_DWORD *)(a1 + 272) = 0;
  *(_QWORD *)(a1 + 280) = 0;
  *(_QWORD *)(a1 + 288) = 0;
  *(_QWORD *)(a1 + 296) = 0;
  *(_DWORD *)(a1 + 304) = 0;
  *(_QWORD *)(a1 + 312) = 0;
  *(_QWORD *)(a1 + 320) = 0;
  *(_QWORD *)(a1 + 328) = 0;
  *(_QWORD *)(a1 + 336) = 0;
  *(_QWORD *)(a1 + 344) = 0;
  *(_QWORD *)(a1 + 352) = 0;
  *(_QWORD *)(a1 + 360) = 0;
  *(_QWORD *)(a1 + 368) = 0;
  *(_QWORD *)(a1 + 376) = 0;
  *(_QWORD *)(a1 + 384) = 0;
  sub_108B430((__int64 *)(a1 + 312));
  *(_QWORD *)(a1 + 392) = 0;
  *(_QWORD *)(a1 + 400) = 0;
  *(_QWORD *)(a1 + 408) = 0;
  *(_QWORD *)(a1 + 416) = 0;
  *(_QWORD *)(a1 + 424) = 0;
  *(_QWORD *)(a1 + 432) = 0;
  *(_QWORD *)(a1 + 440) = 0;
  *(_QWORD *)(a1 + 448) = 0;
  *(_QWORD *)(a1 + 456) = 0;
  *(_QWORD *)(a1 + 464) = 0;
  sub_108B430(v6);
  *(_QWORD *)(a1 + 472) = 0;
  *(_QWORD *)(a1 + 480) = 0;
  *(_QWORD *)(a1 + 488) = 0;
  *(_QWORD *)(a1 + 496) = 0;
  *(_QWORD *)(a1 + 504) = 0;
  *(_QWORD *)(a1 + 512) = 0;
  *(_QWORD *)(a1 + 520) = 0;
  *(_QWORD *)(a1 + 528) = 0;
  *(_QWORD *)(a1 + 536) = 0;
  *(_QWORD *)(a1 + 544) = 0;
  sub_108B430(v7);
  *(_QWORD *)(a1 + 552) = 0;
  *(_QWORD *)(a1 + 560) = 0;
  *(_QWORD *)(a1 + 568) = 0;
  *(_QWORD *)(a1 + 576) = 0;
  *(_QWORD *)(a1 + 584) = 0;
  *(_QWORD *)(a1 + 592) = 0;
  *(_QWORD *)(a1 + 600) = 0;
  *(_QWORD *)(a1 + 608) = 0;
  *(_QWORD *)(a1 + 616) = 0;
  *(_QWORD *)(a1 + 624) = 0;
  sub_108B430((__int64 *)(a1 + 552));
  *(_QWORD *)(a1 + 632) = 0;
  *(_QWORD *)(a1 + 640) = 0;
  *(_QWORD *)(a1 + 648) = 0;
  *(_QWORD *)(a1 + 656) = 0;
  *(_QWORD *)(a1 + 664) = 0;
  *(_QWORD *)(a1 + 672) = 0;
  *(_QWORD *)(a1 + 680) = 0;
  *(_QWORD *)(a1 + 688) = 0;
  *(_QWORD *)(a1 + 696) = 0;
  *(_QWORD *)(a1 + 704) = 0;
  sub_108B430(v4);
  *(_QWORD *)(a1 + 712) = 0;
  *(_QWORD *)(a1 + 720) = 0;
  *(_QWORD *)(a1 + 728) = 0;
  *(_QWORD *)(a1 + 736) = 0;
  *(_QWORD *)(a1 + 744) = 0;
  *(_QWORD *)(a1 + 752) = 0;
  *(_QWORD *)(a1 + 760) = 0;
  *(_QWORD *)(a1 + 768) = 0;
  *(_QWORD *)(a1 + 776) = 0;
  *(_QWORD *)(a1 + 784) = 0;
  sub_108B430(v5);
  *(_QWORD *)(a1 + 792) = 0;
  *(_QWORD *)(a1 + 800) = 0;
  *(_QWORD *)(a1 + 808) = 0;
  *(_QWORD *)(a1 + 816) = 0;
  *(_QWORD *)(a1 + 824) = 0;
  *(_QWORD *)(a1 + 832) = 0;
  *(_QWORD *)(a1 + 840) = 0;
  *(_QWORD *)(a1 + 848) = 0;
  *(_QWORD *)(a1 + 856) = 0;
  *(_QWORD *)(a1 + 864) = 0;
  sub_108B430((__int64 *)(a1 + 792));
  *(_QWORD *)(a1 + 872) = 0;
  *(_QWORD *)(a1 + 880) = 0;
  *(_QWORD *)(a1 + 888) = 0;
  *(_QWORD *)(a1 + 896) = 0;
  *(_QWORD *)(a1 + 904) = 0;
  *(_QWORD *)(a1 + 912) = 0;
  *(_QWORD *)(a1 + 920) = 0;
  *(_QWORD *)(a1 + 928) = 0;
  *(_QWORD *)(a1 + 936) = 0;
  *(_QWORD *)(a1 + 944) = 0;
  sub_108B430((__int64 *)(a1 + 872));
  *(_QWORD *)(a1 + 952) = 0;
  *(_QWORD *)(a1 + 960) = 0;
  *(_QWORD *)(a1 + 968) = 0;
  *(_QWORD *)(a1 + 976) = 0;
  *(_QWORD *)(a1 + 984) = 0;
  *(_QWORD *)(a1 + 992) = 0;
  *(_QWORD *)(a1 + 1000) = 0;
  *(_QWORD *)(a1 + 1008) = 0;
  *(_QWORD *)(a1 + 1016) = 0;
  *(_QWORD *)(a1 + 1024) = 0;
  sub_108B430((__int64 *)(a1 + 952));
  v46 = v6;
  v45 = a1 + 1032;
  v47 = v7;
  v50 = 0;
  v51 = 0;
  v52 = 0;
  v53 = 0;
  v54 = 0;
  v55 = 0;
  v56 = 0;
  dest = 0;
  v58 = 0;
  v59 = 0;
  sub_108B4A0(&v50, 2u);
  v12 = v55;
  if ( (unsigned __int64)v55 >= v59 )
  {
    v13 = (char *)&v46;
    v39 = dest;
    v40 = 16;
    v14 = (char *)&v46;
    goto LABEL_30;
  }
  v13 = (char *)&v46;
  v14 = (char *)&v46;
  do
  {
    v15 = v14;
    v16 = *v12++;
    v14 += 512;
    *v16 = *(_QWORD *)v15;
    v16[63] = *((_QWORD *)v15 + 63);
    qmemcpy(
      (void *)((unsigned __int64)(v16 + 1) & 0xFFFFFFFFFFFFFFF8LL),
      (const void *)(v15 - ((char *)v16 - ((unsigned __int64)(v16 + 1) & 0xFFFFFFFFFFFFFFF8LL))),
      8LL * (((unsigned int)v16 - (((_DWORD)v16 + 8) & 0xFFFFFFF8) + 512) >> 3));
  }
  while ( v59 > (unsigned __int64)v12 );
  if ( v14 != (char *)&v48 )
  {
    v39 = dest;
    v40 = (char *)&v48 - v14;
LABEL_30:
    memcpy(v39, v14, v40);
  }
  sub_108C0B0(v45, ".text", 5u, 32, 0, &v50);
  sub_108AE30(&v50);
  v44 = a1 + 1176;
  v47 = v4;
  v46 = (__int64 *)(a1 + 552);
  v48 = v5;
  v50 = 0;
  v51 = 0;
  v52 = 0;
  v53 = 0;
  v54 = 0;
  v55 = 0;
  v56 = 0;
  dest = 0;
  v58 = 0;
  v59 = 0;
  sub_108B4A0(&v50, 3u);
  v17 = v55;
  if ( v59 <= (unsigned __int64)v55 )
  {
    v37 = dest;
    v38 = 24;
    v18 = (char *)&v46;
    goto LABEL_28;
  }
  v18 = (char *)&v46;
  do
  {
    v19 = v18;
    v20 = *v17++;
    v18 += 512;
    *v20 = *(_QWORD *)v19;
    v20[63] = *((_QWORD *)v19 + 63);
    qmemcpy(
      (void *)((unsigned __int64)(v20 + 1) & 0xFFFFFFFFFFFFFFF8LL),
      (const void *)(v19 - ((char *)v20 - ((unsigned __int64)(v20 + 1) & 0xFFFFFFFFFFFFFFF8LL))),
      8LL * (((unsigned int)v20 - (((_DWORD)v20 + 8) & 0xFFFFFFF8) + 512) >> 3));
  }
  while ( v59 > (unsigned __int64)v17 );
  if ( v18 != v49 )
  {
    v37 = dest;
    v38 = v49 - v18;
LABEL_28:
    memcpy(v37, v18, v38);
  }
  sub_108C0B0(v44, ".data", 5u, 64, 0, &v50);
  sub_108AE30(&v50);
  v50 = 0;
  v46 = (__int64 *)(a1 + 792);
  v51 = 0;
  v52 = 0;
  v53 = 0;
  v54 = 0;
  v55 = 0;
  v56 = 0;
  dest = 0;
  v58 = 0;
  v59 = 0;
  sub_108B4A0(&v50, 1u);
  v21 = v55;
  if ( v59 <= (unsigned __int64)v55 )
  {
    v35 = dest;
    v36 = 8;
    v22 = (char *)&v46;
    goto LABEL_26;
  }
  v22 = (char *)&v46;
  do
  {
    v23 = v22;
    v24 = *v21++;
    v22 += 512;
    *v24 = *(_QWORD *)v23;
    v24[63] = *((_QWORD *)v23 + 63);
    qmemcpy(
      (void *)((unsigned __int64)(v24 + 1) & 0xFFFFFFFFFFFFFFF8LL),
      (const void *)(v23 - ((char *)v24 - ((unsigned __int64)(v24 + 1) & 0xFFFFFFFFFFFFFFF8LL))),
      8LL * (((unsigned int)v24 - (((_DWORD)v24 + 8) & 0xFFFFFFF8) + 512) >> 3));
  }
  while ( v59 > (unsigned __int64)v21 );
  if ( v22 != (char *)&v47 )
  {
    v35 = dest;
    v36 = (char *)&v47 - v22;
LABEL_26:
    memcpy(v35, v22, v36);
  }
  sub_108C0B0(a1 + 1320, ".bss", 4u, 128, 1, &v50);
  sub_108AE30(&v50);
  v43 = a1 + 1464;
  v50 = 0;
  v46 = (__int64 *)(a1 + 872);
  v51 = 0;
  v52 = 0;
  v53 = 0;
  v54 = 0;
  v55 = 0;
  v56 = 0;
  dest = 0;
  v58 = 0;
  v59 = 0;
  sub_108B4A0(&v50, 1u);
  v25 = v55;
  if ( v59 <= (unsigned __int64)v55 )
  {
    v33 = dest;
    v34 = 8;
    v26 = (char *)&v46;
    goto LABEL_24;
  }
  v26 = (char *)&v46;
  do
  {
    v27 = v26;
    v28 = *v25++;
    v26 += 512;
    *v28 = *(_QWORD *)v27;
    v28[63] = *((_QWORD *)v27 + 63);
    qmemcpy(
      (void *)((unsigned __int64)(v28 + 1) & 0xFFFFFFFFFFFFFFF8LL),
      (const void *)(v27 - ((char *)v28 - ((unsigned __int64)(v28 + 1) & 0xFFFFFFFFFFFFFFF8LL))),
      8LL * (((unsigned int)v28 - (((_DWORD)v28 + 8) & 0xFFFFFFF8) + 512) >> 3));
  }
  while ( v59 > (unsigned __int64)v25 );
  if ( v26 != (char *)&v47 )
  {
    v33 = dest;
    v34 = (char *)&v47 - v26;
LABEL_24:
    memcpy(v33, v26, v34);
  }
  sub_108C0B0(v43, ".tdata", 6u, 1024, 0, &v50);
  sub_108AE30(&v50);
  v50 = 0;
  v46 = (__int64 *)(a1 + 952);
  v51 = 0;
  v52 = 0;
  v53 = 0;
  v54 = 0;
  v55 = 0;
  v56 = 0;
  dest = 0;
  v58 = 0;
  v59 = 0;
  sub_108B4A0(&v50, 1u);
  v29 = v55;
  if ( (unsigned __int64)v55 >= v59 )
  {
    v41 = dest;
    v42 = 8;
    goto LABEL_32;
  }
  do
  {
    v30 = v13;
    v31 = *v29++;
    v13 += 512;
    *v31 = *(_QWORD *)v30;
    v31[63] = *((_QWORD *)v30 + 63);
    qmemcpy(
      (void *)((unsigned __int64)(v31 + 1) & 0xFFFFFFFFFFFFFFF8LL),
      (const void *)(v30 - ((char *)v31 - ((unsigned __int64)(v31 + 1) & 0xFFFFFFFFFFFFFFF8LL))),
      8LL * (((unsigned int)v31 - (((_DWORD)v31 + 8) & 0xFFFFFFF8) + 512) >> 3));
  }
  while ( v59 > (unsigned __int64)v29 );
  if ( v13 != (char *)&v47 )
  {
    v41 = dest;
    v42 = (char *)&v47 - v13;
LABEL_32:
    memcpy(v41, v13, v42);
  }
  sub_108C0B0(a1 + 1608, ".tbss", 5u, 2048, 1, &v50);
  sub_108AE30(&v50);
  *(_QWORD *)(a1 + 1768) = a1 + 1320;
  *(_QWORD *)(a1 + 1752) = v45;
  *(_QWORD *)(a1 + 1784) = a1 + 1608;
  *(_QWORD *)(a1 + 1760) = v44;
  *(_QWORD *)(a1 + 1792) = 0;
  *(_QWORD *)(a1 + 1776) = v43;
  *(_QWORD *)(a1 + 1888) = 0x10000000000LL;
  *(_WORD *)(a1 + 1896) = -3;
  *(_QWORD *)(a1 + 1840) = off_497C0B0;
  *(_QWORD *)(a1 + 1928) = a1 + 1912;
  *(_QWORD *)(a1 + 1936) = a1 + 1912;
  *(_QWORD *)(a1 + 1800) = 0;
  *(_QWORD *)(a1 + 1808) = 0;
  *(_QWORD *)(a1 + 1816) = 0;
  *(_QWORD *)(a1 + 1824) = 0;
  *(_QWORD *)(a1 + 1832) = 0;
  *(_BYTE *)(a1 + 1855) = 0;
  *(_QWORD *)(a1 + 1856) = 0;
  *(_QWORD *)(a1 + 1864) = 0;
  *(_QWORD *)(a1 + 1872) = 0;
  *(_QWORD *)(a1 + 1880) = 0;
  *(_DWORD *)(a1 + 1912) = 0;
  *(_QWORD *)(a1 + 1920) = 0;
  *(_QWORD *)(a1 + 1944) = 0;
  *(_BYTE *)(a1 + 1952) = 0;
  *(_DWORD *)(a1 + 1848) = 1668834606;
  *(_WORD *)(a1 + 1852) = 28773;
  *(_BYTE *)(a1 + 1854) = 116;
  *(_WORD *)(a1 + 1973) = 0;
  *(_BYTE *)(a1 + 1975) = 0;
  *(_QWORD *)(a1 + 2008) = 0x20000000000LL;
  *(_QWORD *)(a1 + 1976) = 0;
  *(_QWORD *)(a1 + 1984) = 0;
  *(_QWORD *)(a1 + 1992) = 0;
  *(_QWORD *)(a1 + 2000) = 0;
  *(_WORD *)(a1 + 2016) = -3;
  *(_DWORD *)(a1 + 1968) = 1718511918;
  *(_BYTE *)(a1 + 1972) = 111;
  *(_QWORD *)(a1 + 1960) = off_497C0E0;
  *(_QWORD *)(a1 + 2024) = 0;
  return off_497C0E0;
}
