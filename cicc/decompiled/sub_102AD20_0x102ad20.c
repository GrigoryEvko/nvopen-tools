// Function: sub_102AD20
// Address: 0x102ad20
//
__int64 __fastcall sub_102AD20(__int64 a1, unsigned __int8 *a2, char a3, _QWORD *a4, __int64 a5, __int64 a6)
{
  _QWORD *v7; // r12
  _QWORD *v9; // r14
  _QWORD *v10; // r13
  unsigned __int64 v11; // rsi
  _QWORD *v12; // rax
  _QWORD *v13; // rdi
  __int64 v14; // rcx
  __int64 v15; // rdx
  __int64 v16; // rax
  _QWORD *v17; // rdi
  __int64 v18; // rcx
  __int64 v19; // rdx
  unsigned __int64 v20; // rbx
  unsigned __int64 v21; // r14
  __int64 *v22; // rdx
  char v23; // cl
  _QWORD *v24; // rdi
  __m128i v25; // xmm0
  __int64 *v26; // rax
  __m128i v27; // xmm1
  __m128i v28; // xmm2
  __int64 *v29; // rax
  __int64 v30; // rsi
  __int64 result; // rax
  __int64 v32; // rax
  __int64 v33; // rax
  __int64 v34; // rdx
  __int64 v35; // rsi
  int *v36; // rax
  int v37; // eax
  bool v38; // cc
  int v39; // eax
  char v41; // [rsp+13h] [rbp-35Dh]
  char v42; // [rsp+13h] [rbp-35Dh]
  int v43; // [rsp+14h] [rbp-35Ch]
  __m128i v46; // [rsp+30h] [rbp-340h] BYREF
  __m128i v47; // [rsp+40h] [rbp-330h] BYREF
  __m128i v48; // [rsp+50h] [rbp-320h] BYREF
  __m128i v49[3]; // [rsp+60h] [rbp-310h] BYREF
  char v50; // [rsp+90h] [rbp-2E0h]
  _QWORD v51[2]; // [rsp+A0h] [rbp-2D0h] BYREF
  __int64 v52; // [rsp+B0h] [rbp-2C0h]
  __int64 v53; // [rsp+B8h] [rbp-2B8h] BYREF
  unsigned int v54; // [rsp+C0h] [rbp-2B0h]
  _QWORD v55[2]; // [rsp+1F8h] [rbp-178h] BYREF
  char v56; // [rsp+208h] [rbp-168h]
  _BYTE *v57; // [rsp+210h] [rbp-160h]
  __int64 v58; // [rsp+218h] [rbp-158h]
  _BYTE v59[128]; // [rsp+220h] [rbp-150h] BYREF
  __int16 v60; // [rsp+2A0h] [rbp-D0h]
  _QWORD v61[2]; // [rsp+2A8h] [rbp-C8h] BYREF
  __int64 v62; // [rsp+2B8h] [rbp-B8h]
  __int64 v63; // [rsp+2C0h] [rbp-B0h] BYREF
  unsigned int v64; // [rsp+2C8h] [rbp-A8h]
  char v65; // [rsp+340h] [rbp-30h] BYREF

  v7 = a4;
  v9 = sub_C52410();
  v10 = v9 + 1;
  v11 = sub_C959E0();
  v12 = (_QWORD *)v9[2];
  if ( v12 )
  {
    v13 = v9 + 1;
    do
    {
      while ( 1 )
      {
        v14 = v12[2];
        v15 = v12[3];
        if ( v11 <= v12[4] )
          break;
        v12 = (_QWORD *)v12[3];
        if ( !v15 )
          goto LABEL_6;
      }
      v13 = v12;
      v12 = (_QWORD *)v12[2];
    }
    while ( v14 );
LABEL_6:
    if ( v10 != v13 && v11 >= v13[4] )
      v10 = v13;
  }
  if ( v10 == (_QWORD *)((char *)sub_C52410() + 8) )
    goto LABEL_55;
  v16 = v10[7];
  if ( !v16 )
    goto LABEL_55;
  v17 = v10 + 6;
  do
  {
    while ( 1 )
    {
      v18 = *(_QWORD *)(v16 + 16);
      v19 = *(_QWORD *)(v16 + 24);
      if ( *(_DWORD *)(v16 + 32) >= dword_4F8F208 )
        break;
      v16 = *(_QWORD *)(v16 + 24);
      if ( !v19 )
        goto LABEL_15;
    }
    v17 = (_QWORD *)v16;
    v16 = *(_QWORD *)(v16 + 16);
  }
  while ( v18 );
LABEL_15:
  if ( v17 == v10 + 6 || dword_4F8F208 < *((_DWORD *)v17 + 8) || (v43 = qword_4F8F288, !*((_DWORD *)v17 + 9)) )
  {
LABEL_55:
    v36 = (int *)sub_C94E20((__int64)qword_4F86370);
    if ( v36 )
      v37 = *v36;
    else
      v37 = qword_4F86370[2];
    v38 = v37 < 3;
    v39 = 3200;
    if ( v38 )
      v39 = 500;
    v43 = v39;
  }
  if ( a4 != *(_QWORD **)(a6 + 56) )
  {
    while ( 1 )
    {
      v20 = *v7 & 0xFFFFFFFFFFFFFFF8LL;
      v7 = (_QWORD *)v20;
      if ( !v20 )
        BUG();
      v21 = v20 - 24;
      if ( *(_BYTE *)(v20 - 24) == 85 )
      {
        v32 = *(_QWORD *)(v20 - 56);
        if ( v32 )
        {
          if ( !*(_BYTE *)v32
            && *(_QWORD *)(v32 + 24) == *(_QWORD *)(v20 + 56)
            && (*(_BYTE *)(v32 + 33) & 0x20) != 0
            && (unsigned int)(*(_DWORD *)(v32 + 36) - 68) <= 3 )
          {
            goto LABEL_41;
          }
        }
      }
      if ( !--v43 )
        return 0x6000000000000003LL;
      v22 = *(__int64 **)(a1 + 272);
      v46.m128i_i64[0] = 0;
      v46.m128i_i64[1] = -1;
      v47 = 0u;
      v48 = 0u;
      v23 = sub_102A4D0((unsigned __int8 *)(v20 - 24), &v46, v22);
      if ( v46.m128i_i64[0] )
        break;
      if ( (unsigned __int8)(*(_BYTE *)(v20 - 24) - 34) <= 0x33u
        && (v35 = 0x8000000000041LL, _bittest64(&v35, (unsigned int)*(unsigned __int8 *)(v20 - 24) - 34)) )
      {
        v42 = v23;
        if ( (unsigned __int8)sub_CF5B00(*(_QWORD **)(a1 + 256), a2, (unsigned __int8 *)(v20 - 24)) )
          return v21 | 1;
        if ( a3 && (v42 & 2) == 0 && (unsigned __int8)sub_B46130((__int64)a2, v20 - 24, 0) )
          return v21 | 2;
LABEL_41:
        if ( *(_QWORD *)(a6 + 56) == v20 )
          goto LABEL_42;
      }
      else
      {
        if ( v23 )
          return v21 | 1;
        if ( *(_QWORD *)(a6 + 56) == v20 )
          goto LABEL_42;
      }
    }
    v24 = *(_QWORD **)(a1 + 256);
    v25 = _mm_loadu_si128(&v46);
    v50 = 1;
    v26 = &v53;
    v27 = _mm_loadu_si128(&v47);
    v28 = _mm_loadu_si128(&v48);
    v51[1] = 0;
    v51[0] = v24;
    v52 = 1;
    v49[0] = v25;
    v49[1] = v27;
    v49[2] = v28;
    do
    {
      *v26 = -4;
      v26 += 5;
      *(v26 - 4) = -3;
      *(v26 - 3) = -4;
      *(v26 - 2) = -3;
    }
    while ( v26 != v55 );
    v55[1] = 0;
    v57 = v59;
    v58 = 0x400000000LL;
    v55[0] = v61;
    v56 = 0;
    v60 = 256;
    v61[1] = 0;
    v62 = 1;
    v61[0] = &unk_49DDBE8;
    v29 = &v63;
    do
    {
      *v29 = -4096;
      v29 += 2;
    }
    while ( v29 != (__int64 *)&v65 );
    v30 = (__int64)a2;
    v41 = sub_CF63E0(v24, a2, v49, (__int64)v51);
    v61[0] = &unk_49DDBE8;
    if ( (v62 & 1) == 0 )
    {
      v30 = 16LL * v64;
      sub_C7D6A0(v63, v30, 8);
    }
    nullsub_184();
    if ( v57 != v59 )
      _libc_free(v57, v30);
    if ( (v52 & 1) == 0 )
      sub_C7D6A0(v53, 40LL * v54, 8);
    if ( v41 )
      return v21 | 1;
    goto LABEL_41;
  }
LABEL_42:
  v33 = *(_QWORD *)(*(_QWORD *)(a6 + 72) + 80LL);
  if ( !v33 )
    return 0x2000000000000003LL;
  v34 = v33 - 24;
  result = 0x4000000000000003LL;
  if ( a6 != v34 )
    return 0x2000000000000003LL;
  return result;
}
