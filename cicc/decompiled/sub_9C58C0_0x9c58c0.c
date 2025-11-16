// Function: sub_9C58C0
// Address: 0x9c58c0
//
__int64 __fastcall sub_9C58C0(__int64 a1, __int64 a2)
{
  __int64 v3; // rdi
  __int64 v4; // rdi
  _QWORD *v5; // r13
  _QWORD *v6; // r12
  __int64 v7; // rdi
  __int64 v8; // rdi
  __int64 *v9; // r12
  unsigned __int64 v10; // r13
  __int64 v11; // rdi
  __int64 v12; // rax
  _QWORD *v13; // r12
  _QWORD *v14; // r13
  __int64 v15; // rdi
  __int64 v16; // rdi
  __int64 v17; // rsi
  __int64 v18; // rdi
  __int64 v19; // rdi
  __int64 v20; // rdi
  __int64 v21; // rdi
  __int64 v22; // rdi
  __int64 v23; // rdi
  __int64 v24; // rdi
  __int64 v25; // rsi
  __int64 v26; // rdi
  void (__fastcall *v27)(__int64, __int64, __int64); // rax
  __int64 v28; // r13
  __int64 v29; // r12
  __int64 v30; // rax
  __int64 *v31; // r14
  __int64 *v32; // r12
  __int64 i; // rax
  __int64 v34; // rdi
  unsigned int v35; // ecx
  __int64 *v36; // r12
  __int64 *v37; // r13
  __int64 v38; // rdi
  __int64 v39; // rdi
  __int64 v40; // rsi
  __int64 v41; // rax
  __int64 v42; // r12
  __int64 v43; // r13
  __int64 v44; // rdi
  __int64 v45; // rsi
  __int64 v46; // rdi
  _QWORD *v47; // r13
  _QWORD *v48; // r12
  _QWORD *v49; // r13
  _QWORD *v50; // r12
  __int64 v51; // rdi
  __int64 v52; // r8
  __int64 v53; // r14
  __int64 v54; // rdi
  __int64 v55; // rdx
  __int64 v56; // r12
  volatile signed __int32 *v57; // r13
  signed __int32 v58; // eax
  void (*v59)(); // rax
  signed __int32 v60; // eax
  __int64 (__fastcall *v61)(__int64); // rcx
  __int64 v62; // r14
  __int64 v63; // r12
  volatile signed __int32 *v64; // r13
  signed __int32 v65; // eax
  void (*v66)(); // rax
  signed __int32 v67; // eax
  __int64 (__fastcall *v68)(__int64); // rdx
  __int64 result; // rax
  _QWORD *v70; // r14
  __int64 v71; // r13
  __int64 v72; // r12
  __int64 v73; // rdi
  _QWORD *v74; // rdi
  __int64 v75; // rdx
  __int64 v76; // r12
  volatile signed __int32 *v77; // r13
  void (*v78)(); // rax
  __int64 (__fastcall *v79)(__int64); // rcx
  void (__fastcall *v80)(__int64, __int64, __int64); // rax
  __int64 v81; // [rsp+0h] [rbp-40h]
  __int64 v82; // [rsp+0h] [rbp-40h]
  __int64 v83; // [rsp+0h] [rbp-40h]
  __int64 v84; // [rsp+0h] [rbp-40h]
  __int64 v85; // [rsp+8h] [rbp-38h]
  _QWORD *v86; // [rsp+8h] [rbp-38h]

  *(_QWORD *)a1 = off_49793E0;
  v3 = *(_QWORD *)(a1 + 2008);
  if ( v3 )
  {
    a2 = *(_QWORD *)(a1 + 2024) - v3;
    j_j___libc_free_0(v3, a2);
  }
  if ( *(_BYTE *)(a1 + 2000) )
  {
    v80 = *(void (__fastcall **)(__int64, __int64, __int64))(a1 + 1984);
    *(_BYTE *)(a1 + 2000) = 0;
    if ( v80 )
    {
      a2 = a1 + 1968;
      v80(a1 + 1968, a1 + 1968, 3);
    }
  }
  v4 = *(_QWORD *)(a1 + 1936);
  if ( v4 != a1 + 1960 )
    _libc_free(v4, a2);
  v5 = *(_QWORD **)(a1 + 1920);
  v6 = *(_QWORD **)(a1 + 1912);
  if ( v5 != v6 )
  {
    do
    {
      if ( (_QWORD *)*v6 != v6 + 2 )
        j_j___libc_free_0(*v6, v6[2] + 1LL);
      v6 += 4;
    }
    while ( v5 != v6 );
    v6 = *(_QWORD **)(a1 + 1912);
  }
  if ( v6 )
    j_j___libc_free_0(v6, *(_QWORD *)(a1 + 1928) - (_QWORD)v6);
  sub_C7D6A0(*(_QWORD *)(a1 + 1888), 16LL * *(unsigned int *)(a1 + 1904), 8);
  sub_C7D6A0(*(_QWORD *)(a1 + 1856), 16LL * *(unsigned int *)(a1 + 1872), 8);
  v7 = *(_QWORD *)(a1 + 1808);
  if ( v7 )
    j_j___libc_free_0(v7, *(_QWORD *)(a1 + 1824) - v7);
  v8 = *(_QWORD *)(a1 + 1728);
  if ( v8 )
  {
    v9 = *(__int64 **)(a1 + 1768);
    v10 = *(_QWORD *)(a1 + 1800) + 8LL;
    if ( v10 > (unsigned __int64)v9 )
    {
      do
      {
        v11 = *v9++;
        j_j___libc_free_0(v11, 512);
      }
      while ( v10 > (unsigned __int64)v9 );
      v8 = *(_QWORD *)(a1 + 1728);
    }
    j_j___libc_free_0(v8, 8LL * *(_QWORD *)(a1 + 1736));
  }
  v12 = *(unsigned int *)(a1 + 1720);
  if ( (_DWORD)v12 )
  {
    v13 = *(_QWORD **)(a1 + 1704);
    v14 = &v13[4 * v12];
    do
    {
      if ( *v13 != -4096 && *v13 != -8192 )
      {
        v15 = v13[1];
        if ( v15 )
          j_j___libc_free_0(v15, v13[3] - v15);
      }
      v13 += 4;
    }
    while ( v14 != v13 );
    LODWORD(v12) = *(_DWORD *)(a1 + 1720);
  }
  sub_C7D6A0(*(_QWORD *)(a1 + 1704), 32LL * (unsigned int)v12, 8);
  v16 = *(_QWORD *)(a1 + 1672);
  if ( v16 )
    j_j___libc_free_0(v16, *(_QWORD *)(a1 + 1688) - v16);
  sub_C7D6A0(*(_QWORD *)(a1 + 1648), 16LL * *(unsigned int *)(a1 + 1664), 8);
  v17 = 16LL * *(unsigned int *)(a1 + 1624);
  sub_C7D6A0(*(_QWORD *)(a1 + 1608), v17, 8);
  v18 = *(_QWORD *)(a1 + 1576);
  if ( v18 )
  {
    v17 = *(_QWORD *)(a1 + 1592) - v18;
    j_j___libc_free_0(v18, v17);
  }
  v19 = *(_QWORD *)(a1 + 1552);
  if ( v19 )
  {
    v17 = *(_QWORD *)(a1 + 1568) - v19;
    j_j___libc_free_0(v19, v17);
  }
  sub_9C4250(*(_QWORD *)(a1 + 1520));
  v20 = *(_QWORD *)(a1 + 1480);
  if ( v20 )
  {
    v17 = *(_QWORD *)(a1 + 1496) - v20;
    j_j___libc_free_0(v20, v17);
  }
  v21 = *(_QWORD *)(a1 + 1456);
  if ( v21 )
  {
    v17 = *(_QWORD *)(a1 + 1472) - v21;
    j_j___libc_free_0(v21, v17);
  }
  v22 = *(_QWORD *)(a1 + 1432);
  if ( v22 )
  {
    v17 = *(_QWORD *)(a1 + 1448) - v22;
    j_j___libc_free_0(v22, v17);
  }
  v23 = *(_QWORD *)(a1 + 1408);
  if ( v23 )
  {
    v17 = *(_QWORD *)(a1 + 1424) - v23;
    j_j___libc_free_0(v23, v17);
  }
  v24 = *(_QWORD *)(a1 + 880);
  if ( v24 != a1 + 896 )
    _libc_free(v24, v17);
  v25 = 8LL * *(unsigned int *)(a1 + 872);
  sub_C7D6A0(*(_QWORD *)(a1 + 856), v25, 8);
  v26 = *(_QWORD *)(a1 + 824);
  if ( v26 )
  {
    v25 = *(_QWORD *)(a1 + 840) - v26;
    j_j___libc_free_0(v26, v25);
  }
  if ( *(_BYTE *)(a1 + 816) )
  {
    *(_BYTE *)(a1 + 816) = 0;
    sub_A049B0(a1 + 808);
  }
  v27 = *(void (__fastcall **)(__int64, __int64, __int64))(a1 + 792);
  if ( v27 )
  {
    v25 = a1 + 776;
    v27(a1 + 776, a1 + 776, 3);
  }
  v28 = *(_QWORD *)(a1 + 752);
  v29 = *(_QWORD *)(a1 + 744);
  if ( v28 != v29 )
  {
    do
    {
      v30 = *(_QWORD *)(v29 + 16);
      if ( v30 != -4096 && v30 != 0 && v30 != -8192 )
        sub_BD60C0(v29);
      v29 += 32;
    }
    while ( v28 != v29 );
    v29 = *(_QWORD *)(a1 + 744);
  }
  if ( v29 )
  {
    v25 = *(_QWORD *)(a1 + 760) - v29;
    j_j___libc_free_0(v29, v25);
  }
  v31 = *(__int64 **)(a1 + 664);
  v32 = &v31[*(unsigned int *)(a1 + 672)];
  if ( v31 != v32 )
  {
    for ( i = *(_QWORD *)(a1 + 664); ; i = *(_QWORD *)(a1 + 664) )
    {
      v34 = *v31;
      v35 = (unsigned int)(((__int64)v31 - i) >> 3) >> 7;
      v25 = 4096LL << v35;
      if ( v35 >= 0x1E )
        v25 = 0x40000000000LL;
      ++v31;
      sub_C7D6A0(v34, v25, 16);
      if ( v32 == v31 )
        break;
    }
  }
  v36 = *(__int64 **)(a1 + 712);
  v37 = &v36[2 * *(unsigned int *)(a1 + 720)];
  if ( v36 != v37 )
  {
    do
    {
      v25 = v36[1];
      v38 = *v36;
      v36 += 2;
      sub_C7D6A0(v38, v25, 16);
    }
    while ( v37 != v36 );
    v37 = *(__int64 **)(a1 + 712);
  }
  if ( v37 != (__int64 *)(a1 + 728) )
    _libc_free(v37, v25);
  v39 = *(_QWORD *)(a1 + 664);
  if ( v39 != a1 + 680 )
    _libc_free(v39, v25);
  sub_C7D6A0(*(_QWORD *)(a1 + 624), 16LL * *(unsigned int *)(a1 + 640), 8);
  v40 = 24LL * *(unsigned int *)(a1 + 608);
  sub_C7D6A0(*(_QWORD *)(a1 + 592), v40, 8);
  v41 = *(unsigned int *)(a1 + 576);
  if ( (_DWORD)v41 )
  {
    v42 = *(_QWORD *)(a1 + 560);
    v43 = v42 + 32 * v41;
    do
    {
      while ( 1 )
      {
        if ( *(_DWORD *)v42 <= 0xFFFFFFFD )
        {
          v44 = *(_QWORD *)(v42 + 8);
          if ( v44 != v42 + 24 )
            break;
        }
        v42 += 32;
        if ( v43 == v42 )
          goto LABEL_77;
      }
      _libc_free(v44, v40);
      v42 += 32;
    }
    while ( v43 != v42 );
LABEL_77:
    LODWORD(v41) = *(_DWORD *)(a1 + 576);
  }
  v45 = 32LL * (unsigned int)v41;
  sub_C7D6A0(*(_QWORD *)(a1 + 560), v45, 8);
  v46 = *(_QWORD *)(a1 + 528);
  if ( v46 )
  {
    v45 = *(_QWORD *)(a1 + 544) - v46;
    j_j___libc_free_0(v46, v45);
  }
  v47 = *(_QWORD **)(a1 + 512);
  v48 = *(_QWORD **)(a1 + 504);
  if ( v47 != v48 )
  {
    do
    {
      if ( (_QWORD *)*v48 != v48 + 2 )
      {
        v45 = v48[2] + 1LL;
        j_j___libc_free_0(*v48, v45);
      }
      v48 += 4;
    }
    while ( v47 != v48 );
    v48 = *(_QWORD **)(a1 + 504);
  }
  if ( v48 )
  {
    v45 = *(_QWORD *)(a1 + 520) - (_QWORD)v48;
    j_j___libc_free_0(v48, v45);
  }
  v49 = *(_QWORD **)(a1 + 488);
  v50 = *(_QWORD **)(a1 + 480);
  if ( v49 != v50 )
  {
    do
    {
      if ( (_QWORD *)*v50 != v50 + 2 )
      {
        v45 = v50[2] + 1LL;
        j_j___libc_free_0(*v50, v45);
      }
      v50 += 4;
    }
    while ( v49 != v50 );
    v50 = *(_QWORD **)(a1 + 480);
  }
  if ( v50 )
  {
    v45 = *(_QWORD *)(a1 + 496) - (_QWORD)v50;
    j_j___libc_free_0(v50, v45);
  }
  nullsub_60(a1);
  v51 = *(_QWORD *)(a1 + 400);
  if ( v51 != a1 + 416 )
  {
    v45 = *(_QWORD *)(a1 + 416) + 1LL;
    j_j___libc_free_0(v51, v45);
  }
  v52 = 32LL * *(unsigned int *)(a1 + 104);
  v85 = *(_QWORD *)(a1 + 96);
  v53 = v85 + v52;
  if ( v85 != v85 + v52 )
  {
    while ( 1 )
    {
      v54 = *(_QWORD *)(v53 - 24);
      v55 = *(_QWORD *)(v53 - 16);
      v53 -= 32;
      v56 = v54;
      if ( v55 != v54 )
        break;
LABEL_112:
      if ( v54 )
      {
        v45 = *(_QWORD *)(v53 + 24) - v54;
        j_j___libc_free_0(v54, v45);
      }
      if ( v85 == v53 )
      {
        v53 = *(_QWORD *)(a1 + 96);
        goto LABEL_116;
      }
    }
    while ( 1 )
    {
      v57 = *(volatile signed __int32 **)(v56 + 8);
      if ( !v57 )
        goto LABEL_99;
      if ( &_pthread_key_create )
      {
        v58 = _InterlockedExchangeAdd(v57 + 2, 0xFFFFFFFF);
      }
      else
      {
        v58 = *((_DWORD *)v57 + 2);
        *((_DWORD *)v57 + 2) = v58 - 1;
      }
      if ( v58 != 1 )
        goto LABEL_99;
      v59 = *(void (**)())(*(_QWORD *)v57 + 16LL);
      if ( v59 != nullsub_25 )
      {
        v83 = v55;
        ((void (__fastcall *)(volatile signed __int32 *))v59)(v57);
        v55 = v83;
      }
      if ( &_pthread_key_create )
      {
        v60 = _InterlockedExchangeAdd(v57 + 3, 0xFFFFFFFF);
      }
      else
      {
        v60 = *((_DWORD *)v57 + 3);
        *((_DWORD *)v57 + 3) = v60 - 1;
      }
      if ( v60 != 1 )
        goto LABEL_99;
      v81 = v55;
      v61 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)v57 + 24LL);
      if ( v61 == sub_9C26E0 )
      {
        (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v57 + 8LL))(v57);
        v55 = v81;
        v56 += 16;
        if ( v81 == v56 )
        {
LABEL_111:
          v54 = *(_QWORD *)(v53 + 8);
          goto LABEL_112;
        }
      }
      else
      {
        v61((__int64)v57);
        v55 = v81;
LABEL_99:
        v56 += 16;
        if ( v55 == v56 )
          goto LABEL_111;
      }
    }
  }
LABEL_116:
  if ( v53 != a1 + 112 )
    _libc_free(v53, v45);
  v62 = *(_QWORD *)(a1 + 80);
  v63 = *(_QWORD *)(a1 + 72);
  if ( v62 != v63 )
  {
    while ( 1 )
    {
      v64 = *(volatile signed __int32 **)(v63 + 8);
      if ( !v64 )
        goto LABEL_120;
      if ( &_pthread_key_create )
      {
        v65 = _InterlockedExchangeAdd(v64 + 2, 0xFFFFFFFF);
      }
      else
      {
        v65 = *((_DWORD *)v64 + 2);
        *((_DWORD *)v64 + 2) = v65 - 1;
      }
      if ( v65 != 1 )
        goto LABEL_120;
      v66 = *(void (**)())(*(_QWORD *)v64 + 16LL);
      if ( v66 != nullsub_25 )
        ((void (__fastcall *)(volatile signed __int32 *))v66)(v64);
      if ( &_pthread_key_create )
      {
        v67 = _InterlockedExchangeAdd(v64 + 3, 0xFFFFFFFF);
      }
      else
      {
        v67 = *((_DWORD *)v64 + 3);
        *((_DWORD *)v64 + 3) = v67 - 1;
      }
      if ( v67 != 1 )
        goto LABEL_120;
      v68 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)v64 + 24LL);
      if ( v68 == sub_9C26E0 )
      {
        (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v64 + 8LL))(v64);
        v63 += 16;
        if ( v62 == v63 )
        {
LABEL_132:
          v63 = *(_QWORD *)(a1 + 72);
          break;
        }
      }
      else
      {
        v68((__int64)v64);
LABEL_120:
        v63 += 16;
        if ( v62 == v63 )
          goto LABEL_132;
      }
    }
  }
  if ( v63 )
    j_j___libc_free_0(v63, *(_QWORD *)(a1 + 88) - v63);
  result = *(_QWORD *)(a1 + 16);
  v70 = *(_QWORD **)(a1 + 8);
  v86 = (_QWORD *)result;
  if ( (_QWORD *)result != v70 )
  {
    while ( 1 )
    {
      v71 = v70[9];
      v72 = v70[8];
      if ( v71 != v72 )
      {
        do
        {
          v73 = *(_QWORD *)(v72 + 8);
          if ( v73 != v72 + 24 )
            j_j___libc_free_0(v73, *(_QWORD *)(v72 + 24) + 1LL);
          v72 += 40;
        }
        while ( v71 != v72 );
        v72 = v70[8];
      }
      if ( v72 )
        j_j___libc_free_0(v72, v70[10] - v72);
      v74 = (_QWORD *)v70[4];
      result = (__int64)(v70 + 6);
      if ( v74 != v70 + 6 )
        result = j_j___libc_free_0(v74, v70[6] + 1LL);
      v75 = v70[2];
      v76 = v70[1];
      if ( v75 != v76 )
        break;
LABEL_160:
      if ( v76 )
        result = j_j___libc_free_0(v76, v70[3] - v76);
      v70 += 11;
      if ( v86 == v70 )
      {
        v70 = *(_QWORD **)(a1 + 8);
        goto LABEL_164;
      }
    }
    while ( 1 )
    {
      v77 = *(volatile signed __int32 **)(v76 + 8);
      if ( !v77 )
        goto LABEL_147;
      if ( &_pthread_key_create )
      {
        result = (unsigned int)_InterlockedExchangeAdd(v77 + 2, 0xFFFFFFFF);
      }
      else
      {
        result = *((unsigned int *)v77 + 2);
        *((_DWORD *)v77 + 2) = result - 1;
      }
      if ( (_DWORD)result != 1 )
        goto LABEL_147;
      v78 = *(void (**)())(*(_QWORD *)v77 + 16LL);
      if ( v78 != nullsub_25 )
      {
        v84 = v75;
        ((void (__fastcall *)(volatile signed __int32 *))v78)(v77);
        v75 = v84;
      }
      if ( &_pthread_key_create )
      {
        result = (unsigned int)_InterlockedExchangeAdd(v77 + 3, 0xFFFFFFFF);
      }
      else
      {
        result = *((unsigned int *)v77 + 3);
        *((_DWORD *)v77 + 3) = result - 1;
      }
      if ( (_DWORD)result != 1 )
        goto LABEL_147;
      v82 = v75;
      v79 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)v77 + 24LL);
      if ( v79 == sub_9C26E0 )
      {
        result = (*(__int64 (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v77 + 8LL))(v77);
        v75 = v82;
        v76 += 16;
        if ( v82 == v76 )
        {
LABEL_159:
          v76 = v70[1];
          goto LABEL_160;
        }
      }
      else
      {
        result = v79((__int64)v77);
        v75 = v82;
LABEL_147:
        v76 += 16;
        if ( v75 == v76 )
          goto LABEL_159;
      }
    }
  }
LABEL_164:
  if ( v70 )
    return j_j___libc_free_0(v70, *(_QWORD *)(a1 + 24) - (_QWORD)v70);
  return result;
}
