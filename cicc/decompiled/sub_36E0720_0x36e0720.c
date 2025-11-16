// Function: sub_36E0720
// Address: 0x36e0720
//
void __fastcall sub_36E0720(__int64 a1, __int64 a2)
{
  __int64 v4; // rax
  __int64 v5; // rsi
  unsigned __int64 v6; // r8
  int v7; // ecx
  __int64 v8; // r14
  int v9; // ebx
  _QWORD *v10; // rdi
  int v11; // eax
  __int64 v12; // rax
  __int64 *v13; // rcx
  __int64 v14; // r8
  unsigned __int64 v15; // rdx
  int v16; // ebx
  __int64 *v17; // rax
  __int64 *v18; // rdx
  __int64 v19; // rax
  __m128i v20; // xmm0
  __int64 v21; // rax
  __int32 v22; // edx
  __int64 v23; // rdx
  unsigned __int64 v24; // rbx
  _QWORD *v25; // r11
  const void *v26; // r8
  size_t v27; // r10
  unsigned __int16 *v28; // rcx
  unsigned __int8 *v29; // rax
  __int64 v30; // rcx
  __int64 v31; // r8
  _BYTE *v32; // r9
  __int64 v33; // rbx
  unsigned __int16 *v34; // rdi
  __int128 v35; // [rsp-10h] [rbp-120h]
  const void *v36; // [rsp+8h] [rbp-108h]
  int v37; // [rsp+10h] [rbp-100h]
  _QWORD *v38; // [rsp+10h] [rbp-100h]
  __int64 v39; // [rsp+10h] [rbp-100h]
  unsigned __int64 v40; // [rsp+18h] [rbp-F8h]
  _QWORD *v41; // [rsp+18h] [rbp-F8h]
  unsigned __int64 v42; // [rsp+18h] [rbp-F8h]
  __m128i v43; // [rsp+20h] [rbp-F0h] BYREF
  __m128i v44; // [rsp+30h] [rbp-E0h]
  __int64 v45; // [rsp+40h] [rbp-D0h] BYREF
  int v46; // [rsp+48h] [rbp-C8h]
  unsigned __int16 *v47; // [rsp+50h] [rbp-C0h] BYREF
  __int64 v48; // [rsp+58h] [rbp-B8h]
  _BYTE dest[48]; // [rsp+60h] [rbp-B0h] BYREF
  __int64 *v50; // [rsp+90h] [rbp-80h] BYREF
  __int64 v51; // [rsp+98h] [rbp-78h]
  __int64 v52; // [rsp+A0h] [rbp-70h] BYREF
  int v53; // [rsp+A8h] [rbp-68h]

  v4 = *(_QWORD *)(a2 + 40);
  v5 = *(_QWORD *)(a2 + 80);
  v6 = *(_QWORD *)(v4 + 80);
  v45 = v5;
  v7 = *(_DWORD *)(v4 + 88);
  v8 = *(_QWORD *)(v4 + 120);
  v9 = *(_DWORD *)(v4 + 128);
  v43 = _mm_loadu_si128((const __m128i *)(v4 + 40));
  if ( v5 )
  {
    v37 = v7;
    v40 = v6;
    sub_B96E90((__int64)&v45, v5, 1);
    v7 = v37;
    v6 = v40;
  }
  v10 = *(_QWORD **)(a1 + 64);
  LODWORD(v51) = v7;
  *((_QWORD *)&v35 + 1) = 2;
  v11 = *(_DWORD *)(a2 + 72);
  *(_QWORD *)&v35 = &v50;
  v50 = (__int64 *)v6;
  v52 = v8;
  v53 = v9;
  v46 = v11;
  v12 = sub_33F7800(v10, 5592, (__int64)&v45, 9u, 0, (__int64)&v50, v35);
  v50 = &v52;
  v13 = &v52;
  v14 = v12;
  v15 = (unsigned int)(*(_DWORD *)(a2 + 64) - 1);
  v51 = 0x400000000LL;
  v16 = v15;
  if ( v15 )
  {
    v17 = &v52;
    if ( v15 > 4 )
    {
      v39 = v14;
      v42 = v15;
      sub_C8D5F0((__int64)&v50, &v52, v15, 0x10u, v14, (__int64)&v50);
      v13 = v50;
      v14 = v39;
      v17 = &v50[2 * (unsigned int)v51];
      v18 = &v50[2 * v42];
      if ( v18 != v17 )
        goto LABEL_6;
    }
    else
    {
      v18 = &v52 + 2 * v15;
      if ( v18 != &v52 )
      {
        do
        {
LABEL_6:
          if ( v17 )
          {
            *v17 = 0;
            *((_DWORD *)v17 + 2) = 0;
          }
          v17 += 2;
        }
        while ( v18 != v17 );
        v13 = v50;
      }
    }
    LODWORD(v51) = v16;
  }
  v19 = *(_QWORD *)(a2 + 40);
  v20 = _mm_load_si128(&v43);
  *v13 = *(_QWORD *)v19;
  *((_DWORD *)v13 + 2) = *(_DWORD *)(v19 + 8);
  v21 = (__int64)v50;
  v44 = v20;
  v50[2] = v20.m128i_i64[0];
  v22 = v44.m128i_i32[2];
  *(_QWORD *)(v21 + 32) = v14;
  *(_DWORD *)(v21 + 24) = v22;
  *(_DWORD *)(v21 + 40) = 0;
  if ( *(_DWORD *)(a2 + 64) == 5 )
  {
    v23 = *(_QWORD *)(a2 + 40);
    *(_QWORD *)(v21 + 48) = *(_QWORD *)(v23 + 160);
    *(_DWORD *)(v21 + 56) = *(_DWORD *)(v23 + 168);
  }
  v24 = *(unsigned int *)(a2 + 68);
  v43.m128i_i64[0] = v21;
  v25 = *(_QWORD **)(a1 + 64);
  v47 = (unsigned __int16 *)dest;
  v26 = *(const void **)(a2 + 48);
  v43.m128i_i64[1] = (unsigned int)v51;
  v27 = 16 * v24;
  v48 = 0x300000000LL;
  if ( v24 > 3 )
  {
    v36 = v26;
    v38 = v25;
    sub_C8D5F0((__int64)&v47, dest, v24, 0x10u, (__int64)v26, (__int64)dest);
    v25 = v38;
    v26 = v36;
    v27 = 16 * v24;
    v34 = &v47[8 * (unsigned int)v48];
  }
  else
  {
    v28 = (unsigned __int16 *)dest;
    if ( !v27 )
      goto LABEL_15;
    v34 = (unsigned __int16 *)dest;
  }
  v41 = v25;
  memcpy(v34, v26, v27);
  v28 = v47;
  LODWORD(v27) = v48;
  v25 = v41;
LABEL_15:
  LODWORD(v48) = v24 + v27;
  v29 = sub_3411BE0(v25, 0x31u, (__int64)&v45, v28, (unsigned int)(v24 + v27), (__int64)dest, *(_OWORD *)&v43);
  v32 = dest;
  v33 = (__int64)v29;
  if ( v47 != (unsigned __int16 *)dest )
    _libc_free((unsigned __int64)v47);
  sub_34158F0(*(_QWORD *)(a1 + 64), a2, v33, v30, v31, (__int64)v32);
  sub_3421DB0(v33);
  sub_33ECEA0(*(const __m128i **)(a1 + 64), a2);
  if ( v50 != &v52 )
    _libc_free((unsigned __int64)v50);
  if ( v45 )
    sub_B91220((__int64)&v45, v45);
}
