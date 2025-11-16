// Function: sub_207DEC0
// Address: 0x207dec0
//
void __fastcall sub_207DEC0(__int64 a1, __int64 *a2, __m128i a3, __m128i a4, __m128i a5)
{
  unsigned int v5; // r14d
  __int64 *v6; // r12
  __int64 v7; // rax
  __int64 v8; // r15
  __int64 v9; // r13
  __int64 v10; // rdx
  int v11; // r8d
  int v12; // r9d
  int v13; // edx
  __int64 v14; // rax
  __int64 v15; // rsi
  __int64 v16; // rbx
  const __m128i *v17; // rdx
  const __m128i *v18; // rax
  unsigned int v19; // eax
  __int64 v20; // r14
  unsigned int v21; // ebx
  char v22; // al
  __int64 v23; // rax
  unsigned int v24; // edx
  __int64 v25; // rdx
  __int64 v26; // rax
  __m128i *v27; // rax
  __int64 v28; // r12
  unsigned int v29; // edx
  __int64 v30; // rbx
  __int64 *v31; // r14
  __int64 v32; // rsi
  unsigned __int64 v33; // r15
  __int64 v34; // rax
  unsigned int v35; // ebx
  unsigned int v36; // r13d
  __int64 v37; // rdx
  __int64 v38; // rcx
  __int64 v39; // r8
  __int64 v40; // r9
  __int64 v41; // rdx
  __int64 v42; // rdx
  __int64 *v43; // rax
  __int64 v44; // rdx
  __int64 v45; // rcx
  __int64 v46; // r8
  __int64 *v47; // r13
  int v48; // edx
  int v49; // ebx
  __int64 *v50; // rax
  __int64 v51; // r12
  __int64 v52; // rax
  __int64 v53; // [rsp+8h] [rbp-1C8h]
  __int64 *v54; // [rsp+10h] [rbp-1C0h]
  __int64 v55; // [rsp+10h] [rbp-1C0h]
  __int128 v56; // [rsp+30h] [rbp-1A0h]
  unsigned __int64 v57; // [rsp+30h] [rbp-1A0h]
  unsigned __int64 v58; // [rsp+48h] [rbp-188h]
  __int128 v59; // [rsp+50h] [rbp-180h]
  __int64 v60; // [rsp+50h] [rbp-180h]
  unsigned int v62; // [rsp+68h] [rbp-168h]
  const __m128i *v63; // [rsp+68h] [rbp-168h]
  __int64 *v64; // [rsp+B0h] [rbp-120h] BYREF
  int v65; // [rsp+B8h] [rbp-118h]
  __int64 *v66; // [rsp+C0h] [rbp-110h] BYREF
  int v67; // [rsp+C8h] [rbp-108h]
  unsigned __int64 v68[2]; // [rsp+D0h] [rbp-100h] BYREF
  _BYTE v69[32]; // [rsp+E0h] [rbp-F0h] BYREF
  _BYTE *v70; // [rsp+100h] [rbp-D0h] BYREF
  __int64 v71; // [rsp+108h] [rbp-C8h]
  _BYTE v72[64]; // [rsp+110h] [rbp-C0h] BYREF
  const __m128i *v73; // [rsp+150h] [rbp-80h] BYREF
  __int64 v74; // [rsp+158h] [rbp-78h]
  _BYTE v75[112]; // [rsp+160h] [rbp-70h] BYREF

  v6 = (__int64 *)a1;
  v7 = *(_QWORD *)(a1 + 552);
  v8 = *(_QWORD *)(v7 + 16);
  v9 = sub_1E0A0C0(*(_QWORD *)(v7 + 32));
  v70 = v72;
  v71 = 0x400000000LL;
  v68[1] = 0x400000000LL;
  v10 = *a2;
  v68[0] = (unsigned __int64)v69;
  sub_20C7CE0(v8, v9, v10, &v70, v68, 0);
  v13 = *(_DWORD *)(a1 + 536);
  v64 = 0;
  v62 = v71;
  v58 = (unsigned int)v71;
  v14 = *(_QWORD *)a1;
  v65 = v13;
  if ( v14 )
  {
    if ( &v64 != (__int64 **)(v14 + 48) )
    {
      v15 = *(_QWORD *)(v14 + 48);
      v64 = (__int64 *)v15;
      if ( v15 )
        sub_1623A60((__int64)&v64, v15, 2);
    }
  }
  v73 = (const __m128i *)v75;
  v74 = 0x400000000LL;
  v16 = v58;
  if ( v58 <= 4 )
  {
    v17 = (const __m128i *)&v75[v16 * 16];
    LODWORD(v74) = v62;
    v18 = (const __m128i *)v75;
    if ( &v75[v16 * 16] == v75 )
      goto LABEL_10;
    goto LABEL_7;
  }
  sub_16CD150((__int64)&v73, v75, v58, 16, v11, v12);
  LODWORD(v74) = v62;
  v18 = v73;
  v17 = &v73[v16];
  if ( &v73[v16] != v73 )
  {
    do
    {
LABEL_7:
      if ( v18 )
      {
        v18->m128i_i64[0] = 0;
        v18->m128i_i32[2] = 0;
      }
      ++v18;
    }
    while ( v18 != v17 );
LABEL_10:
    if ( !v62 )
      goto LABEL_23;
  }
  v19 = v5;
  v20 = 0;
  v21 = v19;
  do
  {
    v28 = *(_QWORD *)(a1 + 552);
    v29 = 8 * sub_15A9520(v9, 0);
    if ( v29 == 32 )
    {
      v22 = 5;
    }
    else if ( v29 <= 0x20 )
    {
      v22 = 3;
      if ( v29 != 8 )
        v22 = 4 * (v29 == 16);
    }
    else
    {
      v22 = 6;
      if ( v29 != 64 )
      {
        v22 = 0;
        if ( v29 == 128 )
          v22 = 7;
      }
    }
    LOBYTE(v21) = v22;
    v23 = sub_1D38BB0(v28, *(_QWORD *)(v68[0] + 8 * v20), (__int64)&v64, v21, 0, 1, a3, *(double *)a4.m128i_i64, a5, 0);
    v21 = v24;
    v25 = v23;
    v26 = v20++;
    v27 = (__m128i *)&v73[v26];
    v27->m128i_i64[0] = v25;
    v27->m128i_i32[2] = v21;
  }
  while ( v62 > (unsigned int)v20 );
  v6 = (__int64 *)a1;
LABEL_23:
  v30 = 1;
  v31 = (__int64 *)v6[69];
  v63 = v73;
  v32 = *a2;
  v33 = (unsigned int)sub_15A9FE0(v9, *a2);
  while ( 2 )
  {
    switch ( *(_BYTE *)(v32 + 8) )
    {
      case 0:
      case 8:
      case 0xA:
      case 0xC:
      case 0x10:
        v52 = *(_QWORD *)(v32 + 32);
        v32 = *(_QWORD *)(v32 + 24);
        v30 *= v52;
        continue;
      case 1:
        v34 = 16;
        break;
      case 2:
        v34 = 32;
        break;
      case 3:
      case 9:
        v34 = 64;
        break;
      case 4:
        v34 = 80;
        break;
      case 5:
      case 6:
        v34 = 128;
        break;
      case 7:
        v34 = 8 * (unsigned int)sub_15A9520(v9, 0);
        break;
      case 0xB:
        v34 = *(_DWORD *)(v32 + 8) >> 8;
        break;
      case 0xD:
        v34 = 8LL * *(_QWORD *)sub_15A9930(v9, v32);
        break;
      case 0xE:
        v55 = *(_QWORD *)(v32 + 24);
        v60 = *(_QWORD *)(v32 + 32);
        v57 = (unsigned int)sub_15A9FE0(v9, v55);
        v34 = 8 * v57 * v60 * ((v57 + ((unsigned __int64)(sub_127FA20(v9, v55) + 7) >> 3) - 1) / v57);
        break;
      case 0xF:
        v34 = 8 * (unsigned int)sub_15A9520(v9, *(_DWORD *)(v32 + 8) >> 8);
        break;
    }
    break;
  }
  v35 = v33 * ((v33 + ((unsigned __int64)(v34 * v30 + 7) >> 3) - 1) / v33);
  v36 = sub_15A9FE0(v9, *a2);
  *(_QWORD *)&v56 = sub_1D2AD90((_QWORD *)v6[69], *(a2 - 3), v37, v38, v39, v40);
  *((_QWORD *)&v56 + 1) = v41;
  *(_QWORD *)&v59 = sub_20685E0((__int64)v6, (__int64 *)*(a2 - 3), a3, a4, a5);
  *((_QWORD *)&v59 + 1) = v42;
  v43 = sub_2051C20(v6, *(double *)a3.m128i_i64, *(double *)a4.m128i_i64, a5);
  v45 = (__int64)v43;
  v46 = v44;
  v66 = v64;
  if ( v64 )
  {
    v53 = v44;
    v54 = v43;
    sub_1623A60((__int64)&v66, (__int64)v64, 2);
    v46 = v53;
    v45 = (__int64)v54;
  }
  v67 = v65;
  v47 = sub_1D38FD0(v31, (__int64)&v70, (__int64)&v66, v45, v46, v36, a3, v59, v56, v35, v63);
  v49 = v48;
  if ( v66 )
    sub_161E7C0((__int64)&v66, (__int64)v66);
  v66 = a2;
  v50 = sub_205F5C0((__int64)(v6 + 1), (__int64 *)&v66);
  v50[1] = (__int64)v47;
  *((_DWORD *)v50 + 4) = v49;
  v51 = v6[69];
  if ( v47 )
  {
    nullsub_686();
    *(_QWORD *)(v51 + 176) = v47;
    *(_DWORD *)(v51 + 184) = v58;
    sub_1D23870();
  }
  else
  {
    *(_QWORD *)(v51 + 176) = 0;
    *(_DWORD *)(v51 + 184) = v58;
  }
  if ( v73 != (const __m128i *)v75 )
    _libc_free((unsigned __int64)v73);
  if ( v64 )
    sub_161E7C0((__int64)&v64, (__int64)v64);
  if ( (_BYTE *)v68[0] != v69 )
    _libc_free(v68[0]);
  if ( v70 != v72 )
    _libc_free((unsigned __int64)v70);
}
