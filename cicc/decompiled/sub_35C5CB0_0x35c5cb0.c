// Function: sub_35C5CB0
// Address: 0x35c5cb0
//
__int64 __fastcall sub_35C5CB0(unsigned int *a1, unsigned int a2, __int64 *a3, unsigned int a4, _QWORD *a5, _QWORD *a6)
{
  __int64 v9; // rax
  __int64 v10; // r8
  __int64 v11; // rdx
  int v12; // edx
  __int64 v13; // rcx
  __int64 v14; // r10
  __int64 v15; // r9
  unsigned int v16; // edi
  int v17; // r13d
  unsigned int v18; // ebx
  unsigned int v19; // r11d
  unsigned int *v20; // rax
  unsigned int v21; // edx
  __int64 v22; // rcx
  __int64 v23; // rsi
  unsigned int v24; // esi
  unsigned __int64 v25; // rdx
  __int64 v26; // r10
  _QWORD *v27; // rdi
  __int64 (*v28)(); // rax
  int v29; // r11d
  unsigned __int64 v30; // rsi
  __int64 i; // rsi
  _BYTE *v32; // rdx
  __int64 v33; // rcx
  unsigned __int64 v34; // rsi
  __int64 j; // rsi
  _BYTE *v36; // rdx
  __int64 v37; // rcx
  __int64 *v39; // rax
  unsigned int *v40; // rax
  char v41; // al
  __int64 v42; // rax
  _BYTE *v43; // rax
  _BYTE *v44; // rax
  __int64 v45; // rcx
  __int64 v46; // r8
  __int64 v47; // r9
  __int64 v48; // rcx
  __int64 v49; // r8
  __int64 v50; // r9
  __int64 v51; // rcx
  __int64 v52; // r8
  __int64 v53; // r9
  unsigned int *v56; // [rsp+20h] [rbp-200h]
  int v57; // [rsp+28h] [rbp-1F8h]
  unsigned int v59; // [rsp+30h] [rbp-1F0h]
  __int64 v60; // [rsp+30h] [rbp-1F0h]
  unsigned __int8 v61; // [rsp+38h] [rbp-1E8h]
  unsigned int v62; // [rsp+38h] [rbp-1E8h]
  __int64 v63; // [rsp+38h] [rbp-1E8h]
  __int64 v64; // [rsp+38h] [rbp-1E8h]
  __m128i v65[2]; // [rsp+40h] [rbp-1E0h] BYREF
  char v66; // [rsp+60h] [rbp-1C0h]
  char v67; // [rsp+61h] [rbp-1BFh]
  __m128i v68[2]; // [rsp+70h] [rbp-1B0h] BYREF
  __int16 v69; // [rsp+90h] [rbp-190h]
  __m128i v70[3]; // [rsp+A0h] [rbp-180h] BYREF
  __m128i v71[2]; // [rsp+D0h] [rbp-150h] BYREF
  char v72; // [rsp+F0h] [rbp-130h]
  char v73; // [rsp+F1h] [rbp-12Fh]
  __m128i v74[3]; // [rsp+100h] [rbp-120h] BYREF
  __m128i v75[2]; // [rsp+130h] [rbp-F0h] BYREF
  __int16 v76; // [rsp+150h] [rbp-D0h]
  __m128i v77[3]; // [rsp+160h] [rbp-C0h] BYREF
  __m128i v78[2]; // [rsp+190h] [rbp-90h] BYREF
  char v79; // [rsp+1B0h] [rbp-70h]
  char v80; // [rsp+1B1h] [rbp-6Fh]
  __m128i v81[6]; // [rsp+1C0h] [rbp-60h] BYREF

  v9 = *(_QWORD *)(sub_2E88D60((__int64)a5) + 48);
  v11 = *(_QWORD *)(*(_QWORD *)a1 + 312LL)
      + 16LL
      * (*(unsigned __int16 *)(*a3 + 24)
       + *(_DWORD *)(*(_QWORD *)a1 + 328LL)
       * (unsigned int)((__int64)(*(_QWORD *)(*(_QWORD *)a1 + 288LL) - *(_QWORD *)(*(_QWORD *)a1 + 280LL)) >> 3));
  v10 = *(_DWORD *)(v11 + 4) >> 3;
  LODWORD(v11) = *(_DWORD *)(v11 + 8) >> 3;
  if ( (_DWORD)v11 )
  {
    _BitScanReverse64((unsigned __int64 *)&v11, (unsigned int)v11);
    v12 = v11 ^ 0x3F;
    v13 = (unsigned int)(63 - v12);
    v61 = 63 - v12;
  }
  else
  {
    v61 = -1;
    v13 = 0xFFFFFFFFLL;
  }
  v14 = *(_QWORD *)(v9 + 8);
  v15 = *(unsigned int *)(v9 + 32);
  v16 = a1[12];
  v17 = -*(_DWORD *)(v9 + 32);
  v18 = -858993459 * ((*(_QWORD *)(v9 + 16) - v14) >> 3) - v15;
  if ( !v16 )
  {
    v26 = 0;
    v25 = 1;
LABEL_37:
    v10 = v18;
    if ( a1[13] < v25 )
    {
      v64 = v26;
      sub_C8D5F0((__int64)(a1 + 10), a1 + 14, v25, 0x10u, v18, v15);
      v16 = a1[12];
      v10 = v18;
      v26 = v64;
    }
    v39 = (__int64 *)(*((_QWORD *)a1 + 5) + 16LL * v16);
    *v39 = v10;
    v39[1] = 0;
    v40 = (unsigned int *)*((_QWORD *)a1 + 5);
    ++a1[12];
    v56 = v40;
    goto LABEL_14;
  }
  v19 = v16;
  v59 = -1;
  v56 = (unsigned int *)*((_QWORD *)a1 + 5);
  v20 = v56;
  v21 = 0;
  v57 = (1LL << v13) + v10;
  do
  {
    v13 = v20[1];
    if ( !(_DWORD)v13 )
    {
      v13 = *v20;
      if ( (int)v13 >= v17 && (int)v18 > (int)v13 )
      {
        v22 = v14 + 40LL * (unsigned int)(v15 + v13);
        v23 = *(_QWORD *)(v22 + 8);
        v13 = *(unsigned __int8 *)(v22 + 16);
        if ( (unsigned int)v10 <= (unsigned int)v23 && (unsigned __int8)v13 >= v61 )
        {
          v24 = (1LL << v13) + v23 - v57;
          if ( v24 < v59 )
          {
            v59 = v24;
            v19 = v21;
          }
        }
      }
    }
    ++v21;
    v20 += 4;
  }
  while ( v16 != v21 );
  v25 = v19 + 1LL;
  v26 = 16LL * v19;
  if ( v19 == v16 )
    goto LABEL_37;
LABEL_14:
  *(unsigned int *)((char *)v56 + v26 + 4) = a2;
  v27 = *(_QWORD **)a1;
  v28 = *(__int64 (**)())(**(_QWORD **)a1 + 608LL);
  if ( v28 != sub_2FF52E0 )
  {
    v63 = v26;
    v41 = ((__int64 (__fastcall *)(_QWORD *, _QWORD, _QWORD *, _QWORD *, __int64 *, _QWORD))v28)(
            v27,
            *((_QWORD *)a1 + 3),
            a5,
            a6,
            a3,
            a2);
    v26 = v63;
    if ( v41 )
      return v26 + *((_QWORD *)a1 + 5);
    v27 = *(_QWORD **)a1;
  }
  v29 = *(_DWORD *)(*((_QWORD *)a1 + 5) + v26);
  if ( v29 < v17 || (int)v18 <= v29 )
  {
    v80 = 1;
    v78[0].m128i_i64[0] = (__int64)": Cannot scavenge register without an emergency spill slot!";
    v42 = *a3;
    v79 = 3;
    v43 = (_BYTE *)(v27[10] + *(unsigned int *)(v42 + 16));
    v76 = 257;
    if ( *v43 )
    {
      v75[0].m128i_i64[0] = (__int64)v43;
      LOBYTE(v76) = 3;
    }
    v73 = 1;
    v71[0].m128i_i64[0] = (__int64)" from class ";
    v72 = 3;
    v44 = (_BYTE *)(v27[9] + *(unsigned int *)(v27[1] + 24LL * a2));
    v69 = 257;
    if ( *v44 )
    {
      v68[0].m128i_i64[0] = (__int64)v44;
      LOBYTE(v69) = 3;
    }
    v67 = 1;
    v65[0].m128i_i64[0] = (__int64)"Error while trying to spill ";
    v66 = 3;
    sub_9C6370(v70, v65, v68, v13, v10, v15);
    sub_9C6370(v74, v70, v71, v45, v46, v47);
    sub_9C6370(v77, v74, v75, v48, v49, v50);
    sub_9C6370(v81, v77, v78, v51, v52, v53);
    sub_C64D30((__int64)v81, 1u);
  }
  v60 = v26;
  v62 = *(_DWORD *)(*((_QWORD *)a1 + 5) + v26);
  (*(void (__fastcall **)(_QWORD, _QWORD, _QWORD *, _QWORD, __int64, _QWORD, __int64 *, _QWORD *, _QWORD, _QWORD))(**((_QWORD **)a1 + 1) + 560LL))(
    *((_QWORD *)a1 + 1),
    *((_QWORD *)a1 + 3),
    a5,
    a2,
    1,
    (unsigned int)v29,
    a3,
    v27,
    0,
    0);
  v30 = *a5 & 0xFFFFFFFFFFFFFFF8LL;
  if ( !v30 )
    goto LABEL_47;
  if ( (*(_QWORD *)v30 & 4) == 0 && (*(_BYTE *)(v30 + 44) & 4) != 0 )
  {
    for ( i = *(_QWORD *)v30; ; i = *(_QWORD *)v30 )
    {
      v30 = i & 0xFFFFFFFFFFFFFFF8LL;
      if ( (*(_BYTE *)(v30 + 44) & 4) == 0 )
        break;
    }
  }
  v32 = *(_BYTE **)(v30 + 32);
  v33 = 0;
  if ( *v32 != 5 )
  {
    do
      v33 = (unsigned int)(v33 + 1);
    while ( v32[40 * v33] != 5 );
  }
  (*(void (__fastcall **)(_QWORD, unsigned __int64, _QWORD, __int64, unsigned int *))(**(_QWORD **)a1 + 624LL))(
    *(_QWORD *)a1,
    v30,
    a4,
    v33,
    a1);
  (*(void (__fastcall **)(_QWORD, _QWORD, _QWORD, _QWORD, _QWORD, __int64 *, _QWORD, _QWORD, _QWORD))(**((_QWORD **)a1 + 1) + 568LL))(
    *((_QWORD *)a1 + 1),
    *((_QWORD *)a1 + 3),
    *a6,
    a2,
    v62,
    a3,
    *(_QWORD *)a1,
    0,
    0);
  v34 = *(_QWORD *)*a6 & 0xFFFFFFFFFFFFFFF8LL;
  if ( !v34 )
LABEL_47:
    BUG();
  if ( (*(_QWORD *)v34 & 4) == 0 && (*(_BYTE *)(v34 + 44) & 4) != 0 )
  {
    for ( j = *(_QWORD *)v34; ; j = *(_QWORD *)v34 )
    {
      v34 = j & 0xFFFFFFFFFFFFFFF8LL;
      if ( (*(_BYTE *)(v34 + 44) & 4) == 0 )
        break;
    }
  }
  v36 = *(_BYTE **)(v34 + 32);
  v37 = 0;
  if ( *v36 != 5 )
  {
    do
      v37 = (unsigned int)(v37 + 1);
    while ( v36[40 * v37] != 5 );
  }
  (*(void (__fastcall **)(_QWORD, unsigned __int64, _QWORD, __int64, unsigned int *))(**(_QWORD **)a1 + 624LL))(
    *(_QWORD *)a1,
    v34,
    a4,
    v37,
    a1);
  v26 = v60;
  return v26 + *((_QWORD *)a1 + 5);
}
