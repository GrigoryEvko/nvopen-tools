// Function: sub_374ABE0
// Address: 0x374abe0
//
__int64 __fastcall sub_374ABE0(__int64 *a1, __int64 a2)
{
  __int64 v2; // rdx
  __int64 v3; // rax
  __int64 v4; // rax
  unsigned __int64 v5; // rax
  unsigned int i; // r12d
  __int64 v8; // rcx
  __int64 v9; // r8
  __int64 v10; // r9
  __int64 v11; // r14
  __int64 v12; // rax
  __int64 v13; // rdx
  __int64 v14; // r13
  char *v15; // rax
  _BYTE *v16; // r13
  __int64 v17; // rax
  __int64 v18; // rdx
  __int64 v19; // r14
  unsigned __int16 v20; // ax
  __int64 v21; // rax
  __int64 v22; // rcx
  unsigned __int64 v23; // rsi
  unsigned __int64 v24; // rdx
  char v25; // al
  unsigned int v26; // r12d
  __int64 v28; // rsi
  __int64 v29; // rax
  __int64 v30; // rax
  __int64 v31; // r10
  __int64 v32; // rsi
  unsigned __int8 *v33; // rsi
  bool v34; // zf
  __int64 v35; // rsi
  __int64 v36; // rax
  __int64 v37; // rsi
  unsigned __int8 *v38; // rsi
  unsigned __int64 *v39; // rdx
  _BYTE *v40; // rax
  _BYTE *v41; // r10
  __m128i *v42; // rsi
  __int64 v43; // rsi
  unsigned __int8 *v44; // rsi
  __int64 v45; // rax
  char v46; // dl
  __int64 v47; // rsi
  __int64 v48; // rax
  __int64 v49; // rsi
  unsigned __int64 v50; // rdx
  unsigned __int64 v51; // rcx
  __int64 v52; // rdx
  __int64 v53; // [rsp+8h] [rbp-C8h]
  __int64 v54; // [rsp+8h] [rbp-C8h]
  __int64 v55; // [rsp+8h] [rbp-C8h]
  __int64 v56; // [rsp+8h] [rbp-C8h]
  __int64 v57; // [rsp+8h] [rbp-C8h]
  _BYTE *v58; // [rsp+8h] [rbp-C8h]
  _BYTE *v59; // [rsp+8h] [rbp-C8h]
  __int64 v60; // [rsp+10h] [rbp-C0h]
  int v61; // [rsp+1Ch] [rbp-B4h]
  __int64 v62; // [rsp+20h] [rbp-B0h]
  __int64 v63; // [rsp+28h] [rbp-A8h]
  __int32 v64; // [rsp+3Ch] [rbp-94h] BYREF
  unsigned __int8 *v65; // [rsp+40h] [rbp-90h] BYREF
  __int64 v66; // [rsp+48h] [rbp-88h]
  __int64 v67; // [rsp+50h] [rbp-80h]
  __int64 v68; // [rsp+60h] [rbp-70h] BYREF
  char *v69; // [rsp+68h] [rbp-68h]
  __int64 v70; // [rsp+70h] [rbp-60h]
  int v71; // [rsp+78h] [rbp-58h]
  char v72; // [rsp+7Ch] [rbp-54h]
  char v73; // [rsp+80h] [rbp-50h] BYREF

  v2 = a1[5];
  v69 = &v73;
  v68 = 0;
  v3 = *(_QWORD *)(v2 + 864) - *(_QWORD *)(v2 + 856);
  v70 = 4;
  v71 = 0;
  *(_DWORD *)(v2 + 880) = v3 >> 4;
  v4 = *(_QWORD *)(a2 + 48);
  v72 = 1;
  v5 = v4 & 0xFFFFFFFFFFFFFFF8LL;
  if ( v5 == a2 + 48 )
    return 1;
  if ( !v5 )
    BUG();
  v62 = v5 - 24;
  if ( (unsigned int)*(unsigned __int8 *)(v5 - 24) - 30 > 0xA )
    return 1;
  v61 = sub_B46E30(v5 - 24);
  if ( !v61 )
    return 1;
  for ( i = 0; i != v61; ++i )
  {
    while ( 1 )
    {
      v11 = sub_B46EC0(v62, i);
      v12 = *(_QWORD *)(v11 + 56);
      if ( !v12 )
LABEL_88:
        BUG();
      if ( *(_BYTE *)(v12 - 24) != 84 )
        goto LABEL_6;
      v13 = *(unsigned int *)(v11 + 44);
      v14 = *(_QWORD *)(*(_QWORD *)(a1[5] + 56) + 8 * v13);
      if ( !v72 )
        goto LABEL_69;
      v15 = v69;
      v8 = HIDWORD(v70);
      v13 = (__int64)&v69[8 * HIDWORD(v70)];
      if ( v69 != (char *)v13 )
        break;
LABEL_13:
      if ( HIDWORD(v70) < (unsigned int)v70 )
      {
        ++HIDWORD(v70);
        *(_QWORD *)v13 = v14;
        ++v68;
        goto LABEL_15;
      }
LABEL_69:
      sub_C8CC70((__int64)&v68, v14, v13, v8, v9, v10);
      if ( !v46 )
        goto LABEL_6;
LABEL_15:
      v16 = *(_BYTE **)(v14 + 56);
      v17 = sub_AA5930(v11);
      v60 = v18;
      v19 = v17;
      if ( v17 == v18 )
        goto LABEL_6;
      do
      {
        if ( *(_QWORD *)(v19 + 16) )
        {
          v20 = sub_2D5BAE0(a1[16], a1[14], *(__int64 **)(v19 + 8), 1);
          if ( v20 <= 1u || !*(_QWORD *)(a1[16] + 8LL * v20 + 112) && v20 != 2 && (unsigned __int16)(v20 - 5) > 1u )
          {
            v21 = a1[5];
            v22 = *(_QWORD *)(v21 + 856);
            v23 = *(unsigned int *)(v21 + 880);
            v24 = (*(_QWORD *)(v21 + 864) - v22) >> 4;
            if ( v23 > v24 )
            {
              sub_3741A50((const __m128i **)(v21 + 856), v23 - v24);
            }
            else if ( v23 < v24 )
            {
              v47 = v22 + 16 * v23;
              if ( *(_QWORD *)(v21 + 864) != v47 )
                *(_QWORD *)(v21 + 864) = v47;
            }
LABEL_20:
            v25 = v72;
            v26 = 0;
            goto LABEL_21;
          }
          v28 = *(_QWORD *)(v19 - 8);
          v29 = 0x1FFFFFFFE0LL;
          if ( (*(_DWORD *)(v19 + 4) & 0x7FFFFFF) != 0 )
          {
            v30 = 0;
            do
            {
              if ( a2 == *(_QWORD *)(v28 + 32LL * *(unsigned int *)(v19 + 72) + 8 * v30) )
              {
                v29 = 32 * v30;
                goto LABEL_30;
              }
              ++v30;
            }
            while ( (*(_DWORD *)(v19 + 4) & 0x7FFFFFF) != (_DWORD)v30 );
            v29 = 0x1FFFFFFFE0LL;
          }
LABEL_30:
          v31 = *(_QWORD *)(v28 + v29);
          v32 = a1[10];
          v65 = 0;
          v66 = 0;
          v67 = 0;
          v63 = (__int64)(a1 + 10);
          if ( v32 )
          {
            v53 = v31;
            sub_B91220((__int64)(a1 + 10), v32);
            v33 = v65;
            v31 = v53;
            v34 = v65 == 0;
            a1[10] = (__int64)v65;
            if ( !v34 )
            {
              sub_B976B0((__int64)&v65, v33, v63);
              v31 = v53;
            }
            a1[11] = v66;
            a1[12] = v67;
          }
          else
          {
            a1[11] = 0;
            a1[12] = 0;
          }
          if ( *(_BYTE *)v31 > 0x1Cu )
          {
            v35 = *(_QWORD *)(v31 + 48);
            v65 = (unsigned __int8 *)v35;
            if ( v35 )
            {
              v54 = v31;
              sub_B96E90((__int64)&v65, v35, 1);
              v31 = v54;
            }
            v36 = 0;
            if ( (*(_BYTE *)(v31 + 7) & 0x20) != 0 )
            {
              v55 = v31;
              v36 = sub_B91C10(v31, 37);
              v31 = v55;
            }
            v37 = a1[10];
            v66 = v36;
            v67 = 0;
            if ( v37 )
            {
              v56 = v31;
              sub_B91220(v63, v37);
              v31 = v56;
            }
            v38 = v65;
            a1[10] = (__int64)v65;
            if ( v38 )
            {
              v57 = v31;
              sub_B976B0((__int64)&v65, v38, v63);
              v31 = v57;
            }
            a1[11] = v66;
            a1[12] = v67;
          }
          v64 = sub_3746830(a1, v31);
          if ( !v64 )
          {
            v48 = a1[5];
            v49 = *(_QWORD *)(v48 + 856);
            v50 = *(unsigned int *)(v48 + 880);
            v51 = (*(_QWORD *)(v48 + 864) - v49) >> 4;
            if ( v50 > v51 )
            {
              sub_3741A50((const __m128i **)(v48 + 856), v50 - v51);
            }
            else if ( v50 < v51 )
            {
              v52 = v49 + 16 * v50;
              if ( *(_QWORD *)(v48 + 864) != v52 )
                *(_QWORD *)(v48 + 864) = v52;
            }
            goto LABEL_20;
          }
          v39 = (unsigned __int64 *)a1[5];
          if ( !v16 )
            BUG();
          v40 = v16;
          if ( (*v16 & 4) == 0 && (v16[44] & 8) != 0 )
          {
            do
              v40 = (_BYTE *)*((_QWORD *)v40 + 1);
            while ( (v40[44] & 8) != 0 );
          }
          v41 = (_BYTE *)*((_QWORD *)v40 + 1);
          v65 = v16;
          v42 = (__m128i *)v39[108];
          if ( v42 == (__m128i *)v39[109] )
          {
            v59 = v41;
            sub_3741C00(v39 + 107, v42, &v65, &v64);
            v41 = v59;
          }
          else
          {
            if ( v42 )
            {
              v42->m128i_i64[0] = (__int64)v16;
              v42->m128i_i32[2] = v64;
              v42 = (__m128i *)v39[108];
            }
            v39[108] = (unsigned __int64)&v42[1];
          }
          v43 = a1[10];
          v67 = 0;
          v65 = 0;
          v66 = 0;
          if ( v43 )
          {
            v58 = v41;
            sub_B91220(v63, v43);
            v44 = v65;
            v41 = v58;
            v34 = v65 == 0;
            a1[10] = (__int64)v65;
            if ( !v34 )
            {
              sub_B976B0((__int64)&v65, v44, v63);
              v41 = v58;
            }
            a1[11] = v66;
            a1[12] = v67;
          }
          else
          {
            a1[11] = 0;
            a1[12] = 0;
          }
          v16 = v41;
        }
        v45 = *(_QWORD *)(v19 + 32);
        if ( !v45 )
          goto LABEL_88;
        v19 = 0;
        if ( *(_BYTE *)(v45 - 24) == 84 )
          v19 = v45 - 24;
      }
      while ( v60 != v19 );
      if ( v61 == ++i )
        goto LABEL_63;
    }
    while ( v14 != *(_QWORD *)v15 )
    {
      v15 += 8;
      if ( (char *)v13 == v15 )
        goto LABEL_13;
    }
LABEL_6:
    ;
  }
LABEL_63:
  v25 = v72;
  v26 = 1;
LABEL_21:
  if ( !v25 )
    _libc_free((unsigned __int64)v69);
  return v26;
}
