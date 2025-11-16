// Function: sub_27BF540
// Address: 0x27bf540
//
__int64 __fastcall sub_27BF540(__int64 a1, __int64 a2, __int64 a3, __int64 a4, _QWORD *a5)
{
  unsigned __int64 v7; // r13
  __int64 v8; // rax
  __int64 v9; // rdx
  __int64 v10; // rsi
  unsigned int v11; // eax
  __int64 v12; // rcx
  __int64 v13; // rsi
  __int64 v14; // rax
  unsigned __int64 v15; // rcx
  const __m128i **v16; // rsi
  __int64 v17; // rdx
  __int64 v18; // rcx
  __m128i *v19; // r8
  __int64 v20; // r9
  _QWORD *v21; // rbx
  _QWORD *v22; // r14
  __int64 v23; // rax
  _QWORD *v24; // rbx
  _QWORD *v25; // r14
  __int64 v26; // rax
  __int64 v28; // rdx
  __int64 v29; // rcx
  __int64 v30; // r8
  void *v31; // r9
  unsigned __int64 v32; // [rsp+10h] [rbp-170h]
  char v33; // [rsp+18h] [rbp-168h]
  __int64 v34; // [rsp+28h] [rbp-158h] BYREF
  __m128i v35; // [rsp+30h] [rbp-150h] BYREF
  __int64 (__fastcall *v36)(const __m128i **, const __m128i *, int); // [rsp+40h] [rbp-140h]
  __int64 (__fastcall *v37)(__int64, __int64 *); // [rsp+48h] [rbp-138h]
  __int64 v38; // [rsp+50h] [rbp-130h] BYREF
  unsigned __int64 v39; // [rsp+58h] [rbp-128h]
  __int64 v40; // [rsp+60h] [rbp-120h]
  __int64 v41; // [rsp+68h] [rbp-118h]
  _QWORD v42[2]; // [rsp+70h] [rbp-110h] BYREF
  const __m128i *v43[2]; // [rsp+80h] [rbp-100h] BYREF
  __int64 (__fastcall *v44)(const __m128i **, const __m128i *, int); // [rsp+90h] [rbp-F0h]
  __int64 (__fastcall *v45)(__int64, __int64 *); // [rsp+98h] [rbp-E8h]
  unsigned __int64 v46[19]; // [rsp+A0h] [rbp-E0h] BYREF
  __int64 v47; // [rsp+138h] [rbp-48h]
  __int64 v48; // [rsp+140h] [rbp-40h]
  __int64 v49; // [rsp+148h] [rbp-38h]

  v34 = sub_D47840(a3);
  if ( !v34 )
    v34 = **(_QWORD **)(a3 + 32);
  v7 = a5[9];
  if ( v7 )
  {
    v8 = sub_22077B0(0x2F8u);
    if ( v8 )
    {
      *(_QWORD *)v8 = v7;
      *(_QWORD *)(v8 + 8) = v8 + 24;
      *(_QWORD *)(v8 + 416) = v8 + 440;
      *(_QWORD *)(v8 + 16) = 0x1000000000LL;
      *(_QWORD *)(v8 + 504) = v8 + 520;
      *(_QWORD *)(v8 + 408) = 0;
      *(_QWORD *)(v8 + 424) = 8;
      *(_DWORD *)(v8 + 432) = 0;
      *(_BYTE *)(v8 + 436) = 1;
      *(_QWORD *)(v8 + 512) = 0x800000000LL;
      *(_DWORD *)(v8 + 720) = 0;
      *(_QWORD *)(v8 + 728) = 0;
      *(_QWORD *)(v8 + 736) = v8 + 720;
      *(_QWORD *)(v8 + 744) = v8 + 720;
      *(_QWORD *)(v8 + 752) = 0;
    }
    v7 = v8;
  }
  v35.m128i_i64[1] = a3;
  v9 = a5[2];
  v35.m128i_i64[0] = (__int64)&v34;
  v37 = sub_27B98E0;
  v36 = sub_27B8E10;
  if ( v34 )
  {
    v10 = (unsigned int)(*(_DWORD *)(v34 + 44) + 1);
    v11 = *(_DWORD *)(v34 + 44) + 1;
  }
  else
  {
    v10 = 0;
    v11 = 0;
  }
  v12 = 0;
  if ( v11 < *(_DWORD *)(v9 + 32) )
    v12 = *(_QWORD *)(*(_QWORD *)(v9 + 24) + 8 * v10);
  v13 = a5[3];
  v14 = a5[1];
  v38 = v9;
  v40 = v13;
  v42[1] = v12;
  v39 = 0;
  v41 = v14;
  v42[0] = v7;
  v44 = 0;
  sub_27B8E10(v43, &v35, 2);
  v46[0] = v15;
  v32 = v15;
  v45 = v37;
  v46[18] = 0;
  v44 = v36;
  v46[1] = 0x1000000000LL;
  v47 = 0;
  v48 = 0;
  v49 = 0;
  v33 = sub_27BCF90(&v38);
  v16 = (const __m128i **)(8LL * (unsigned int)v49);
  sub_C7D6A0(v47, (__int64)v16, 8);
  v18 = v32;
  v19 = &v35;
  if ( v46[0] != v32 )
  {
    _libc_free(v46[0]);
    v19 = &v35;
  }
  if ( v44 )
  {
    v16 = v43;
    v44(v43, (const __m128i *)v43, 3);
    v19 = &v35;
  }
  if ( v36 )
  {
    v16 = (const __m128i **)&v35;
    v36((const __m128i **)&v35, &v35, 3);
  }
  v20 = a1 + 32;
  if ( v33 )
  {
    sub_22D0390((__int64)&v38, (__int64)v16, v17, v18, (__int64)v19, v20);
    v31 = (void *)(a1 + 32);
    if ( a5[9] )
    {
      sub_27B9A30((__int64)&v38, (__int64)&unk_4F8F810, v28, v29, v30, (__int64)v31);
      v31 = (void *)(a1 + 32);
    }
    sub_C8CF70(a1, v31, 2, (__int64)v42, (__int64)&v38);
    sub_C8CF70(a1 + 48, (void *)(a1 + 80), 2, (__int64)v46, (__int64)v43);
    if ( !BYTE4(v45) )
      _libc_free((unsigned __int64)v43[1]);
    if ( !BYTE4(v41) )
      _libc_free(v39);
  }
  else
  {
    *(_QWORD *)(a1 + 56) = a1 + 80;
    *(_QWORD *)(a1 + 16) = 0x100000002LL;
    *(_QWORD *)(a1 + 8) = v20;
    *(_QWORD *)(a1 + 48) = 0;
    *(_QWORD *)(a1 + 64) = 2;
    *(_DWORD *)(a1 + 72) = 0;
    *(_BYTE *)(a1 + 76) = 1;
    *(_DWORD *)(a1 + 24) = 0;
    *(_BYTE *)(a1 + 28) = 1;
    *(_QWORD *)(a1 + 32) = &qword_4F82400;
    *(_QWORD *)a1 = 1;
  }
  if ( v7 )
  {
    sub_27B9790(*(_QWORD **)(v7 + 728));
    v21 = *(_QWORD **)(v7 + 504);
    v22 = &v21[3 * *(unsigned int *)(v7 + 512)];
    if ( v21 != v22 )
    {
      do
      {
        v23 = *(v22 - 1);
        v22 -= 3;
        if ( v23 != 0 && v23 != -4096 && v23 != -8192 )
          sub_BD60C0(v22);
      }
      while ( v21 != v22 );
      v22 = *(_QWORD **)(v7 + 504);
    }
    if ( v22 != (_QWORD *)(v7 + 520) )
      _libc_free((unsigned __int64)v22);
    if ( !*(_BYTE *)(v7 + 436) )
      _libc_free(*(_QWORD *)(v7 + 416));
    v24 = *(_QWORD **)(v7 + 8);
    v25 = &v24[3 * *(unsigned int *)(v7 + 16)];
    if ( v24 != v25 )
    {
      do
      {
        v26 = *(v25 - 1);
        v25 -= 3;
        if ( v26 != 0 && v26 != -4096 && v26 != -8192 )
          sub_BD60C0(v25);
      }
      while ( v24 != v25 );
      v25 = *(_QWORD **)(v7 + 8);
    }
    if ( v25 != (_QWORD *)(v7 + 24) )
      _libc_free((unsigned __int64)v25);
    j_j___libc_free_0(v7);
  }
  return a1;
}
