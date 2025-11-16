// Function: sub_37DCC20
// Address: 0x37dcc20
//
__int64 __fastcall sub_37DCC20(__int64 a1, unsigned int a2, unsigned int a3, __int64 a4, _QWORD *a5, _QWORD *a6)
{
  __int64 v7; // r14
  unsigned int v8; // r13d
  __int64 v10; // rax
  __int64 v11; // rdx
  __int64 v12; // r12
  _DWORD *v13; // rax
  __int64 v14; // rdx
  __int64 v15; // rcx
  __int64 v16; // r8
  unsigned int *v17; // r10
  _BYTE **v18; // r11
  _QWORD *v19; // rax
  __int64 v20; // rdi
  _QWORD *v21; // r8
  __int64 v22; // rcx
  __int64 v23; // rdx
  _QWORD *v24; // rsi
  __int64 i; // rax
  unsigned int *v26; // rcx
  __int64 v27; // r12
  int v28; // r13d
  unsigned __int64 v29; // rbx
  __int64 v31; // rax
  __int64 v32; // rax
  int v33; // esi
  __m128i v34; // rax
  unsigned int v35; // r14d
  _BYTE *v36; // r15
  int v37; // r13d
  unsigned int v38; // r12d
  __int64 v39; // rdx
  int v40; // r13d
  __int64 v41; // r12
  __int64 **v42; // rdx
  __int64 **v43; // r8
  __int64 **v44; // rax
  __int64 *v45; // r9
  __int64 v46; // rcx
  __int64 v47; // rax
  int v48; // eax
  _QWORD *v49; // rdi
  unsigned int v50; // esi
  __int16 *v51; // rax
  __int16 *v52; // rcx
  int v53; // eax
  unsigned __int16 v54; // dx
  __int16 *v55; // r13
  unsigned int v56; // r15d
  int v57; // r14d
  int v58; // eax
  __int64 v59; // rdx
  unsigned int v60; // [rsp+Ch] [rbp-E4h]
  _BYTE **v61; // [rsp+10h] [rbp-E0h]
  int v62; // [rsp+10h] [rbp-E0h]
  __int64 v63; // [rsp+20h] [rbp-D0h]
  unsigned __int64 v64; // [rsp+20h] [rbp-D0h]
  unsigned int v65; // [rsp+20h] [rbp-D0h]
  int v66; // [rsp+20h] [rbp-D0h]
  unsigned int v67; // [rsp+28h] [rbp-C8h]
  __int64 v68; // [rsp+28h] [rbp-C8h]
  unsigned int *v69; // [rsp+28h] [rbp-C8h]
  unsigned int v71; // [rsp+30h] [rbp-C0h]
  __int8 v73; // [rsp+38h] [rbp-B8h]
  __int64 v74; // [rsp+40h] [rbp-B0h]
  int v75; // [rsp+40h] [rbp-B0h]
  int v76; // [rsp+40h] [rbp-B0h]
  __int64 v77; // [rsp+40h] [rbp-B0h]
  __int64 v78; // [rsp+48h] [rbp-A8h]
  unsigned int v79; // [rsp+48h] [rbp-A8h]
  unsigned int *v80; // [rsp+48h] [rbp-A8h]
  _QWORD *v81; // [rsp+48h] [rbp-A8h]
  _BYTE **v82; // [rsp+48h] [rbp-A8h]
  __m128i v83; // [rsp+60h] [rbp-90h] BYREF
  __m128i v84; // [rsp+70h] [rbp-80h] BYREF
  unsigned int v85; // [rsp+80h] [rbp-70h] BYREF
  int v86; // [rsp+84h] [rbp-6Ch]
  __int64 v87; // [rsp+88h] [rbp-68h]
  int v88; // [rsp+90h] [rbp-60h]
  _BYTE *v89; // [rsp+A0h] [rbp-50h] BYREF
  __int64 v90; // [rsp+A8h] [rbp-48h]
  _BYTE v91[64]; // [rsp+B0h] [rbp-40h] BYREF

  v7 = a3;
  v8 = a2;
  v10 = *(_QWORD *)(a4 + 24);
  v85 = a2;
  v11 = *(_QWORD *)(v10 + 32);
  v12 = v11 + 904;
  v89 = v91;
  v78 = v11;
  v86 = v7;
  v87 = 0;
  v88 = 0;
  v90 = 0x400000000LL;
  v13 = sub_37BB410(v11 + 904, &v85);
  v14 = v78;
  v15 = *(unsigned int *)(v78 + 912);
  v16 = *(_QWORD *)(v78 + 904);
  if ( v13 != (_DWORD *)(v16 + 20 * v15) )
  {
    v17 = &v85;
    v18 = &v89;
    do
    {
      if ( *v13 != v8 || v86 != v13[1] )
        break;
      v8 = v13[2];
      v7 = (unsigned int)v13[3];
      v33 = v13[4];
      v85 = v8;
      v86 = v7;
      if ( v33 )
      {
        v69 = v17;
        v77 = v14;
        v82 = v18;
        sub_9C8C60((__int64)v18, v33);
        v14 = v77;
        v17 = v69;
        v18 = v82;
        v15 = *(unsigned int *)(v77 + 912);
        v16 = *(_QWORD *)(v77 + 904);
      }
      v61 = v18;
      v63 = v16;
      v68 = v15;
      v74 = v14;
      v80 = v17;
      v13 = sub_37BB410(v12, v17);
      v15 = v68;
      v16 = v63;
      v17 = v80;
      v14 = v74;
      v18 = v61;
    }
    while ( v13 != (_DWORD *)(v63 + 20 * v68) );
  }
  v19 = *(_QWORD **)(a1 + 744);
  v20 = a1 + 736;
  v83 = 0;
  v21 = (_QWORD *)(a1 + 736);
  if ( v19 )
  {
    do
    {
      while ( 1 )
      {
        v22 = v19[2];
        v23 = v19[3];
        if ( (unsigned __int64)v8 <= v19[4] )
          break;
        v19 = (_QWORD *)v19[3];
        if ( !v23 )
          goto LABEL_9;
      }
      v21 = v19;
      v19 = (_QWORD *)v19[2];
    }
    while ( v22 );
LABEL_9:
    if ( (_QWORD *)v20 != v21 && (unsigned __int64)v8 < v21[4] )
      v21 = (_QWORD *)(a1 + 736);
  }
  v24 = *(_QWORD **)(a1 + 776);
  for ( i = *(unsigned int *)(a1 + 784); i > 0; i >>= 1 )
  {
    while ( 1 )
    {
      v26 = (unsigned int *)&v24[5 * (i >> 1)];
      if ( v8 <= *v26 )
        break;
      v24 = v26 + 10;
      i = i - (i >> 1) - 1;
      if ( i <= 0 )
        goto LABEL_16;
    }
  }
LABEL_16:
  if ( (_QWORD *)v20 == v21 )
  {
    if ( v24 == (_QWORD *)(40LL * *(unsigned int *)(a1 + 784) + *(_QWORD *)(a1 + 776)) || v8 != *v24 )
      goto LABEL_19;
    v34.m128i_i64[0] = sub_37DCA30(a1, *(_QWORD *)(*(_QWORD *)(a4 + 24) + 32LL), a5, a6, a4, v8);
    v83 = v34;
    v71 = ((unsigned __int64)v34.m128i_i64[0] >> 20) & 0xFFFFF;
    v67 = v34.m128i_i32[0] & 0xFFFFF;
    v79 = (unsigned __int32)v34.m128i_i32[1] >> 8;
    v73 = v34.m128i_i8[8];
    if ( !v34.m128i_i8[8] )
      goto LABEL_20;
  }
  else
  {
    v27 = v21[5];
    v28 = *(_DWORD *)(*(_QWORD *)(v27 + 24) + 24LL);
    if ( unk_445066C != (_DWORD)v7 )
    {
      if ( (*(_DWORD *)(v27 + 40) & 0xFFFFFFu) > (unsigned int)v7 )
      {
        v31 = *(_QWORD *)(v27 + 32) + 40 * v7;
        if ( !*(_BYTE *)v31 && (*(_BYTE *)(v31 + 3) & 0x10) != 0 )
        {
          v32 = *(unsigned int *)(v31 + 8);
          if ( (_DWORD)v32 )
          {
            v73 = 1;
            v67 = v28 & 0xFFFFF;
            v71 = v21[6] & 0xFFFFF;
            v79 = *(_DWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 408) + 64LL) + 4 * v32) & 0xFFFFFF;
            goto LABEL_36;
          }
        }
      }
LABEL_19:
      v73 = 0;
      v79 = 0;
      v71 = 0;
      v67 = 0;
      goto LABEL_20;
    }
    v81 = v21;
    sub_2E864A0(v21[5]);
    if ( v59 != 1 )
      goto LABEL_19;
    v84.m128i_i64[0] = sub_37C7180(a1, v27);
    if ( !v84.m128i_i8[4] )
      goto LABEL_19;
    v73 = 1;
    v67 = v28 & 0xFFFFF;
    v71 = v81[6] & 0xFFFFF;
    v79 = v84.m128i_i32[0] & 0xFFFFFF;
  }
LABEL_36:
  if ( !(_DWORD)v90 )
    goto LABEL_20;
  v35 = 0;
  v75 = 0;
  v64 = (unsigned __int64)v89;
  v36 = &v89[4 * (unsigned int)v90];
  do
  {
    v37 = *((_DWORD *)v36 - 1);
    v38 = sub_2FF7530(*(_QWORD *)(a1 + 16), v37);
    v75 += sub_2FF7550(*(_QWORD *)(a1 + 16), v37);
    if ( v38 < v35 || !v35 )
      v35 = v38;
    v36 -= 4;
  }
  while ( v36 != (_BYTE *)v64 );
  v39 = *(_QWORD *)(a1 + 408);
  v40 = v75;
  v65 = *(_DWORD *)(*(_QWORD *)(v39 + 88) + 4LL * v79);
  if ( v65 >= *(_DWORD *)(v39 + 284) )
    goto LABEL_58;
  v41 = *(_QWORD *)(a1 + 16);
  v42 = *(__int64 ***)(v41 + 288);
  v43 = *(__int64 ***)(v41 + 280);
  if ( v42 == v43 )
    BUG();
  v44 = *(__int64 ***)(v41 + 280);
  v45 = 0;
  do
  {
    if ( v65 - 1 <= 0x3FFFFFFE )
    {
      v46 = **v44;
      if ( v65 >> 3 < *(unsigned __int16 *)(v46 + 22)
        && (((int)*(unsigned __int8 *)(*(_QWORD *)(v46 + 8) + (v65 >> 3)) >> (v65 & 7)) & 1) != 0 )
      {
        v45 = *v44;
      }
    }
    ++v44;
  }
  while ( v42 != v44 );
  v47 = *(unsigned int *)(*(_QWORD *)(v41 + 312)
                        + 16LL * (*(_DWORD *)(v41 + 328) * (unsigned int)(v42 - v43) + *(unsigned __int16 *)(*v45 + 24)));
  v84.m128i_i8[8] = 0;
  v84.m128i_i64[0] = v47;
  v48 = sub_CA1930(&v84);
  if ( v75 || v35 != v48 )
  {
    v49 = *(_QWORD **)(a1 + 16);
    v50 = v65;
    v51 = (__int16 *)(v49[7] + 2LL * *(unsigned int *)(v49[1] + 24LL * v65 + 4));
    v52 = v51 + 1;
    v53 = *v51;
    v76 = v65 + v53;
    if ( (_WORD)v53 )
    {
      v60 = v35;
      v54 = v65 + v53;
      v62 = v40;
      v55 = v52;
      while ( 1 )
      {
        v56 = v54;
        v57 = sub_E91E30(v49, v50, v54);
        v66 = sub_2FF7530(*(_QWORD *)(a1 + 16), v57);
        if ( (unsigned int)sub_2FF7550(*(_QWORD *)(a1 + 16), v57) == v62 && v60 == v66 )
          break;
        v58 = *v55++;
        if ( !(_WORD)v58 )
          goto LABEL_58;
        v76 += v58;
        v54 = v76;
        v49 = *(_QWORD **)(a1 + 16);
      }
      if ( v56 )
      {
        v79 = sub_37BA440(*(_QWORD *)(a1 + 408), v56) & 0xFFFFFF;
        goto LABEL_20;
      }
    }
LABEL_58:
    v73 = 0;
  }
LABEL_20:
  v29 = v83.m128i_i64[0] & 0xFFFFFF0000000000LL | v67 | ((unsigned __int64)v71 << 20);
  v83.m128i_i32[0] = v67 | (v71 << 20);
  v83.m128i_i8[5] = v79;
  v83.m128i_i8[6] = BYTE1(v79);
  v83.m128i_i8[7] = BYTE2(v79);
  v83.m128i_i8[4] = BYTE4(v29);
  v83.m128i_i8[8] = v73;
  v84 = _mm_loadu_si128(&v83);
  if ( v89 != v91 )
    _libc_free((unsigned __int64)v89);
  return v84.m128i_i64[0];
}
