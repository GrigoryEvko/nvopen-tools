// Function: sub_FD9C90
// Address: 0xfd9c90
//
__int64 __fastcall sub_FD9C90(__int64 a1, __int64 *a2, __int64 a3, __int64 a4, __m128i si128)
{
  __int64 *v7; // rax
  __int64 *v8; // rax
  __int64 v9; // r14
  __m128 *v10; // rdx
  __int64 v11; // rax
  const char *v12; // rax
  size_t v13; // rdx
  _BYTE *v14; // rdi
  unsigned __int8 *v15; // rsi
  unsigned __int64 v16; // rax
  __int64 v17; // r14
  __int64 v18; // r13
  __int64 i; // rdx
  unsigned __int64 v20; // rsi
  unsigned int v21; // eax
  _QWORD *v22; // rbx
  _QWORD *v23; // r13
  __int64 v24; // rax
  __int64 v25; // rsi
  unsigned __int64 *v26; // rbx
  unsigned __int64 *v27; // r14
  unsigned __int64 v28; // rdx
  _QWORD *v29; // r13
  _QWORD *v30; // r15
  __int64 v31; // rax
  unsigned __int64 *v32; // rdi
  unsigned __int8 *v34; // rsi
  __int64 v35; // rax
  unsigned __int64 v36; // rax
  __int64 v37; // rax
  __int64 v39; // [rsp+28h] [rbp-378h]
  size_t v40; // [rsp+28h] [rbp-378h]
  _QWORD v41[2]; // [rsp+30h] [rbp-370h] BYREF
  __int64 v42; // [rsp+40h] [rbp-360h]
  __int64 v43; // [rsp+50h] [rbp-350h]
  __int64 v44; // [rsp+58h] [rbp-348h]
  __int64 v45; // [rsp+60h] [rbp-340h]
  __int64 *v46; // [rsp+70h] [rbp-330h] BYREF
  char *v47; // [rsp+78h] [rbp-328h] BYREF
  unsigned __int64 *v48; // [rsp+80h] [rbp-320h]
  __int64 v49; // [rsp+88h] [rbp-318h]
  _QWORD *v50; // [rsp+90h] [rbp-310h]
  __int64 v51; // [rsp+98h] [rbp-308h]
  unsigned int v52; // [rsp+A0h] [rbp-300h]
  int v53; // [rsp+A8h] [rbp-2F8h]
  __int64 v54; // [rsp+B0h] [rbp-2F0h]
  _QWORD v55[3]; // [rsp+C0h] [rbp-2E0h] BYREF
  __int64 v56; // [rsp+D8h] [rbp-2C8h]
  __int64 v57; // [rsp+E0h] [rbp-2C0h] BYREF
  unsigned int v58; // [rsp+E8h] [rbp-2B8h]
  _QWORD v59[2]; // [rsp+220h] [rbp-180h] BYREF
  char v60; // [rsp+230h] [rbp-170h]
  _BYTE *v61; // [rsp+238h] [rbp-168h]
  __int64 v62; // [rsp+240h] [rbp-160h]
  _BYTE v63[128]; // [rsp+248h] [rbp-158h] BYREF
  __int16 v64; // [rsp+2C8h] [rbp-D8h]
  _QWORD v65[2]; // [rsp+2D0h] [rbp-D0h] BYREF
  __int64 v66; // [rsp+2E0h] [rbp-C0h]
  __int64 v67; // [rsp+2E8h] [rbp-B8h] BYREF
  unsigned int v68; // [rsp+2F0h] [rbp-B0h]
  char v69; // [rsp+368h] [rbp-38h] BYREF

  v55[2] = 0;
  v56 = 1;
  v55[0] = sub_BC1CD0(a4, &unk_4F86540, a3) + 8;
  v55[1] = v55[0];
  v7 = &v57;
  do
  {
    *v7 = -4;
    v7 += 5;
    *(v7 - 4) = -3;
    *(v7 - 3) = -4;
    *(v7 - 2) = -3;
  }
  while ( v7 != v59 );
  v59[1] = 0;
  v59[0] = v65;
  v61 = v63;
  v62 = 0x400000000LL;
  v60 = 0;
  v65[1] = 0;
  v66 = 1;
  v64 = 256;
  v65[0] = &unk_49DDBE8;
  v8 = &v67;
  do
  {
    *v8 = -4096;
    v8 += 2;
  }
  while ( v8 != (__int64 *)&v69 );
  v50 = 0;
  v46 = v55;
  v51 = 0;
  v47 = (char *)&v47 + 4;
  v52 = 0;
  v9 = *a2;
  v53 = 0;
  v54 = 0;
  v49 = 0;
  v10 = *(__m128 **)(v9 + 32);
  v11 = *(_QWORD *)(v9 + 24);
  v48 = (unsigned __int64 *)&v47;
  if ( (unsigned __int64)(v11 - (_QWORD)v10) <= 0x18 )
  {
    v9 = sub_CB6200(v9, "Alias sets for function '", 0x19u);
  }
  else
  {
    si128 = _mm_load_si128((const __m128i *)&xmmword_3F8C310);
    v10[1].m128_i8[8] = 39;
    v10[1].m128_u64[0] = 0x206E6F6974636E75LL;
    *v10 = (__m128)si128;
    *(_QWORD *)(v9 + 32) += 25LL;
  }
  v12 = sub_BD5D20(a3);
  v14 = *(_BYTE **)(v9 + 32);
  v15 = (unsigned __int8 *)v12;
  v16 = *(_QWORD *)(v9 + 24) - (_QWORD)v14;
  if ( v16 < v13 )
  {
    v37 = sub_CB6200(v9, v15, v13);
    v14 = *(_BYTE **)(v37 + 32);
    v9 = v37;
    v16 = *(_QWORD *)(v37 + 24) - (_QWORD)v14;
  }
  else if ( v13 )
  {
    v40 = v13;
    memcpy(v14, v15, v13);
    v14 = (_BYTE *)(v40 + *(_QWORD *)(v9 + 32));
    v36 = *(_QWORD *)(v9 + 24) - (_QWORD)v14;
    *(_QWORD *)(v9 + 32) = v14;
    if ( v36 > 2 )
      goto LABEL_10;
    goto LABEL_59;
  }
  if ( v16 > 2 )
  {
LABEL_10:
    v14[2] = 10;
    *(_WORD *)v14 = 14887;
    *(_QWORD *)(v9 + 32) += 3LL;
    goto LABEL_11;
  }
LABEL_59:
  sub_CB6200(v9, "':\n", 3u);
LABEL_11:
  v17 = a3 + 72;
  v18 = *(_QWORD *)(a3 + 80);
  if ( v17 == v18 )
  {
    i = 0;
  }
  else
  {
    if ( !v18 )
      BUG();
    while ( 1 )
    {
      i = *(_QWORD *)(v18 + 32);
      if ( i != v18 + 24 )
        break;
      v18 = *(_QWORD *)(v18 + 8);
      if ( v17 == v18 )
        goto LABEL_17;
      if ( !v18 )
        BUG();
    }
  }
  while ( v17 != v18 )
  {
    v34 = (unsigned __int8 *)(i - 24);
    v39 = i;
    if ( !i )
      v34 = 0;
    sub_FD98A0(&v46, v34, si128);
    for ( i = *(_QWORD *)(v39 + 8); ; i = *(_QWORD *)(v18 + 32) )
    {
      v35 = v18 - 24;
      if ( !v18 )
        v35 = 0;
      if ( i != v35 + 48 )
        break;
      v18 = *(_QWORD *)(v18 + 8);
      if ( v17 == v18 )
        goto LABEL_17;
      if ( !v18 )
        BUG();
    }
  }
LABEL_17:
  v20 = *a2;
  sub_FD6EF0((__int64)&v46, *a2);
  *(_QWORD *)(a1 + 48) = 0;
  *(_QWORD *)(a1 + 8) = a1 + 32;
  *(_QWORD *)(a1 + 56) = a1 + 80;
  *(_QWORD *)(a1 + 16) = 0x100000002LL;
  *(_QWORD *)(a1 + 64) = 2;
  *(_QWORD *)(a1 + 32) = &unk_4F82400;
  *(_DWORD *)(a1 + 72) = 0;
  *(_BYTE *)(a1 + 76) = 1;
  *(_DWORD *)(a1 + 24) = 0;
  *(_BYTE *)(a1 + 28) = 1;
  *(_QWORD *)a1 = 1;
  sub_FD6240((__int64)&v46, v20);
  v21 = v52;
  if ( v52 )
  {
    v22 = v50;
    v41[0] = 0;
    v41[1] = 0;
    v42 = -4096;
    v23 = &v50[4 * v52];
    v43 = 0;
    v44 = 0;
    v45 = -8192;
    do
    {
      v24 = v22[2];
      if ( v24 != 0 && v24 != -4096 && v24 != -8192 )
        sub_BD60C0(v22);
      v22 += 4;
    }
    while ( v23 != v22 );
    if ( v42 != 0 && v42 != -4096 )
      sub_BD60C0(v41);
    v21 = v52;
  }
  v25 = 32LL * v21;
  sub_C7D6A0((__int64)v50, v25, 8);
  v26 = v48;
  while ( v26 != (unsigned __int64 *)&v47 )
  {
    v27 = v26;
    v26 = (unsigned __int64 *)v26[1];
    v28 = *v27 & 0xFFFFFFFFFFFFFFF8LL;
    *v26 = v28 | *v26 & 7;
    *(_QWORD *)(v28 + 8) = v26;
    v29 = (_QWORD *)v27[6];
    v30 = (_QWORD *)v27[5];
    *v27 &= 7u;
    v27[1] = 0;
    if ( v29 != v30 )
    {
      do
      {
        v31 = v30[2];
        if ( v31 != 0 && v31 != -4096 && v31 != -8192 )
          sub_BD60C0(v30);
        v30 += 3;
      }
      while ( v29 != v30 );
      v30 = (_QWORD *)v27[5];
    }
    if ( v30 )
    {
      v25 = v27[7] - (_QWORD)v30;
      j_j___libc_free_0(v30, v25);
    }
    v32 = (unsigned __int64 *)v27[3];
    if ( v27 + 5 != v32 )
      _libc_free(v32, v25);
    v25 = 72;
    j_j___libc_free_0(v27, 72);
  }
  v65[0] = &unk_49DDBE8;
  if ( (v66 & 1) == 0 )
  {
    v25 = 16LL * v68;
    sub_C7D6A0(v67, v25, 8);
  }
  nullsub_184();
  if ( v61 != v63 )
    _libc_free(v61, v25);
  if ( (v56 & 1) == 0 )
    sub_C7D6A0(v57, 40LL * v58, 8);
  return a1;
}
