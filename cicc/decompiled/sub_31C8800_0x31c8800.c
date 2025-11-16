// Function: sub_31C8800
// Address: 0x31c8800
//
__m128i *__fastcall sub_31C8800(__m128i *a1, __int64 *a2, __int64 a3)
{
  _QWORD *v6; // rdi
  __int128 v7; // kr00_16
  __int64 *v8; // rax
  __int64 *v9; // rax
  unsigned __int8 v10; // bl
  unsigned __int8 v11; // dl
  __int64 v12; // rax
  __int64 v13; // rcx
  __int64 v14; // rsi
  unsigned int v15; // eax
  __int64 v16; // rcx
  unsigned __int64 v17; // rdx
  __int64 v18; // rax
  unsigned int v19; // r8d
  __int64 v20; // rdi
  unsigned int v21; // esi
  __int64 *v22; // rax
  __int64 v23; // r10
  int v24; // eax
  __m128i v25; // xmm1
  __m128i v26; // xmm0
  __m128i v27; // xmm2
  __m128i v28; // xmm3
  __int64 v29; // rax
  int v31; // eax
  int v32; // r11d
  __int64 v33; // [rsp+10h] [rbp-340h]
  __int64 v34; // [rsp+18h] [rbp-338h]
  __int64 v35; // [rsp+20h] [rbp-330h]
  __int64 v36; // [rsp+30h] [rbp-320h]
  __m128i v37[3]; // [rsp+40h] [rbp-310h] BYREF
  char v38; // [rsp+70h] [rbp-2E0h]
  __m128i v39; // [rsp+80h] [rbp-2D0h] BYREF
  __m128i v40; // [rsp+90h] [rbp-2C0h] BYREF
  _BYTE v41[24]; // [rsp+A0h] [rbp-2B0h] BYREF
  __int64 v42; // [rsp+B8h] [rbp-298h]
  __int64 v43; // [rsp+C0h] [rbp-290h]
  _QWORD v44[2]; // [rsp+1D8h] [rbp-178h] BYREF
  char v45; // [rsp+1E8h] [rbp-168h]
  _BYTE *v46; // [rsp+1F0h] [rbp-160h]
  __int64 v47; // [rsp+1F8h] [rbp-158h]
  _BYTE v48[128]; // [rsp+200h] [rbp-150h] BYREF
  __int16 v49; // [rsp+280h] [rbp-D0h]
  _QWORD v50[2]; // [rsp+288h] [rbp-C8h] BYREF
  __int64 v51; // [rsp+298h] [rbp-B8h]
  __int64 v52; // [rsp+2A0h] [rbp-B0h] BYREF
  unsigned int v53; // [rsp+2A8h] [rbp-A8h]
  char v54; // [rsp+320h] [rbp-30h] BYREF

  sub_D66840(&v39, (_BYTE *)a3);
  v6 = (_QWORD *)a2[1];
  v38 = 0;
  v35 = v39.m128i_i64[0];
  v39.m128i_i64[0] = (__int64)v6;
  v34 = v39.m128i_i64[1];
  v39.m128i_i64[1] = 0;
  v33 = v40.m128i_i64[0];
  v40.m128i_i64[0] = 1;
  v36 = v40.m128i_i64[1];
  v7 = *(_OWORD *)v41;
  v8 = &v40.m128i_i64[1];
  do
  {
    *v8 = -4;
    v8 += 5;
    *(v8 - 4) = -3;
    *(v8 - 3) = -4;
    *(v8 - 2) = -3;
  }
  while ( v8 != v44 );
  v44[1] = 0;
  v46 = v48;
  v47 = 0x400000000LL;
  v49 = 256;
  v44[0] = v50;
  v45 = 0;
  v50[1] = 0;
  v51 = 1;
  v50[0] = &unk_49DDBE8;
  v9 = &v52;
  do
  {
    *v9 = -4096;
    v9 += 2;
  }
  while ( v9 != (__int64 *)&v54 );
  v10 = sub_CF63E0(v6, (unsigned __int8 *)a3, v37, (__int64)&v39);
  v50[0] = &unk_49DDBE8;
  if ( (v51 & 1) == 0 )
    sub_C7D6A0(v52, 16LL * v53, 8);
  nullsub_184();
  if ( v46 != v48 )
    _libc_free((unsigned __int64)v46);
  if ( (v40.m128i_i8[0] & 1) == 0 )
    sub_C7D6A0(v40.m128i_i64[1], 40LL * *(unsigned int *)v41, 8);
  v11 = v10 & 1;
  if ( (v10 & 1) != 0 )
    v11 = ((v10 >> 1) ^ 1) & 1;
  v12 = *(_QWORD *)(a3 + 40);
  v13 = *a2;
  if ( v12 )
  {
    v14 = (unsigned int)(*(_DWORD *)(v12 + 44) + 1);
    v15 = *(_DWORD *)(v12 + 44) + 1;
  }
  else
  {
    v14 = 0;
    v15 = 0;
  }
  if ( v15 >= *(_DWORD *)(v13 + 32) || (v16 = *(_QWORD *)(*(_QWORD *)(v13 + 24) + 8 * v14)) == 0 )
  {
    a1[4].m128i_i8[8] = 0;
    return a1;
  }
  v17 = a3 & 0xFFFFFFFFFFFFFFFBLL | (4LL * v11);
  v39.m128i_i64[1] = v35;
  v40.m128i_i64[0] = v34;
  v40.m128i_i64[1] = v33;
  *(_QWORD *)v41 = v36;
  *(_OWORD *)&v41[8] = v7;
  v18 = a2[2];
  v19 = *(_DWORD *)(v18 + 32);
  v20 = *(_QWORD *)(v18 + 16);
  if ( !v19 )
    goto LABEL_23;
  v21 = (v19 - 1) & (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4));
  v22 = (__int64 *)(v20 + 16LL * v21);
  v23 = *v22;
  if ( a3 != *v22 )
  {
    v31 = 1;
    while ( v23 != -4096 )
    {
      v32 = v31 + 1;
      v21 = (v19 - 1) & (v31 + v21);
      v22 = (__int64 *)(v20 + 16LL * v21);
      v23 = *v22;
      if ( a3 == *v22 )
        goto LABEL_19;
      v31 = v32;
    }
LABEL_23:
    v22 = (__int64 *)(v20 + 16LL * v19);
  }
LABEL_19:
  v24 = *((_DWORD *)v22 + 2);
  v25 = _mm_loadu_si128(&v40);
  v39.m128i_i64[0] = v17;
  v42 = v16;
  v26 = _mm_loadu_si128(&v39);
  LODWORD(v43) = v24;
  v27 = _mm_loadu_si128((const __m128i *)v41);
  v28 = _mm_loadu_si128((const __m128i *)&v41[16]);
  v29 = v43;
  a1[4].m128i_i8[8] = 1;
  *a1 = v26;
  a1[4].m128i_i64[0] = v29;
  a1[1] = v25;
  a1[2] = v27;
  a1[3] = v28;
  return a1;
}
