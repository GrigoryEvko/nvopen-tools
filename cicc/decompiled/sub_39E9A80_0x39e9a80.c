// Function: sub_39E9A80
// Address: 0x39e9a80
//
__int64 __fastcall sub_39E9A80(
        __int64 a1,
        __int64 a2,
        unsigned int a3,
        unsigned __int8 *a4,
        __int64 a5,
        char *a6,
        unsigned __int8 *a7,
        __int64 a8,
        const __m128i *a9,
        unsigned int a10)
{
  __int64 v14; // rdi
  const __m128i *v15; // r14
  __int64 v16; // rax
  __int64 v17; // r11
  __int64 v18; // rcx
  __int64 v19; // rdx
  __int64 v20; // r11
  char v21; // dl
  char v22; // al
  char v23; // al
  __int64 v25; // rax
  char v26; // al
  char v27; // al
  __int64 v28; // rdi
  __m128i *v29; // rsi
  __int64 v30; // rdx
  __int64 v31; // rax
  __int64 v32; // rdx
  int v33; // [rsp+10h] [rbp-150h]
  __int64 v34; // [rsp+18h] [rbp-148h]
  unsigned int v35; // [rsp+18h] [rbp-148h]
  unsigned int v36; // [rsp+18h] [rbp-148h]
  unsigned __int8 *v37; // [rsp+20h] [rbp-140h] BYREF
  __int64 v38; // [rsp+28h] [rbp-138h]
  __int64 v39; // [rsp+30h] [rbp-130h] BYREF
  char v40; // [rsp+38h] [rbp-128h]
  _QWORD v41[2]; // [rsp+40h] [rbp-120h] BYREF
  __m128i v42; // [rsp+50h] [rbp-110h] BYREF
  __int16 v43; // [rsp+60h] [rbp-100h]
  _QWORD v44[4]; // [rsp+70h] [rbp-F0h] BYREF
  int v45; // [rsp+90h] [rbp-D0h]
  __m128i *v46; // [rsp+98h] [rbp-C8h]
  __m128i v47; // [rsp+A0h] [rbp-C0h] BYREF
  _BYTE v48[176]; // [rsp+B0h] [rbp-B0h] BYREF

  v14 = *(_QWORD *)(a2 + 8);
  v38 = a5;
  v37 = a4;
  v15 = a9;
  LODWORD(v44[0]) = a10;
  v16 = *(_QWORD *)(v14 + 992);
  v17 = v14 + 984;
  if ( !v16 )
    goto LABEL_14;
  do
  {
    while ( 1 )
    {
      v18 = *(_QWORD *)(v16 + 16);
      v19 = *(_QWORD *)(v16 + 24);
      if ( a10 <= *(_DWORD *)(v16 + 32) )
        break;
      v16 = *(_QWORD *)(v16 + 24);
      if ( !v19 )
        goto LABEL_6;
    }
    v17 = v16;
    v16 = *(_QWORD *)(v16 + 16);
  }
  while ( v18 );
LABEL_6:
  if ( v14 + 984 == v17 || a10 < *(_DWORD *)(v17 + 32) )
  {
LABEL_14:
    v36 = a3;
    v47.m128i_i64[0] = (__int64)v44;
    v25 = sub_39E9160((_QWORD *)(v14 + 976), v17, (unsigned int **)&v47);
    a3 = v36;
    v17 = v25;
  }
  v33 = *(_DWORD *)(v17 + 168);
  v48[0] = v15[1].m128i_i8[0];
  if ( v48[0] )
    v47 = _mm_loadu_si128(v15);
  v34 = v17;
  sub_38C95C0((__int64)&v39, v17 + 40, (__int64)&v37, (__int64 *)&a7, (__int64)a6, &v47, a3);
  v20 = v34;
  v21 = v40 & 1;
  v22 = (2 * (v40 & 1)) | v40 & 0xFD;
  v40 = v22;
  if ( v21 )
  {
    *(_BYTE *)(a1 + 8) |= 3u;
    v40 = v22 & 0xFD;
    v31 = v39;
    v39 = 0;
    *(_QWORD *)a1 = v31 & 0xFFFFFFFFFFFFFFFELL;
LABEL_25:
    if ( v39 )
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v39 + 8LL))(v39);
    return a1;
  }
  v35 = v39;
  if ( v33 == *(_DWORD *)(v20 + 168) )
  {
    v23 = *(_BYTE *)(a1 + 8);
    *(_DWORD *)a1 = v39;
    *(_BYTE *)(a1 + 8) = v23 & 0xFC | 2;
    return a1;
  }
  v47.m128i_i64[0] = (__int64)v48;
  v47.m128i_i64[1] = 0x8000000000LL;
  v46 = &v47;
  v45 = 1;
  v44[0] = &unk_49EFC48;
  memset(&v44[1], 0, 24);
  sub_16E7A40((__int64)v44, 0, 0, 0);
  v26 = *(_BYTE *)(a2 + 680) >> 2;
  LOBYTE(v43) = v15[1].m128i_i8[0];
  v27 = v26 & 1;
  if ( (_BYTE)v43 )
    v42 = _mm_loadu_si128(v15);
  sub_39E85A0(v35, v37, v38, a7, a8, a6, (__int64)&v42, v27, (__int64)v44);
  v28 = *(_QWORD *)(a2 + 16);
  if ( v28 )
  {
    v29 = (__m128i *)v46->m128i_i64[0];
    (*(void (__fastcall **)(__int64, __int64, _QWORD))(*(_QWORD *)v28 + 40LL))(
      v28,
      v46->m128i_i64[0],
      v46->m128i_u32[2]);
  }
  else
  {
    v29 = &v42;
    v32 = v46->m128i_i64[0];
    v41[1] = v46->m128i_u32[2];
    v43 = 261;
    v41[0] = v32;
    v42.m128i_i64[0] = (__int64)v41;
    sub_38DD5A0((__int64 *)a2, (__int64)&v42);
  }
  *(_BYTE *)(a1 + 8) = *(_BYTE *)(a1 + 8) & 0xFC | 2;
  *(_DWORD *)a1 = v35;
  v44[0] = &unk_49EFD28;
  sub_16E7960((__int64)v44);
  if ( (_BYTE *)v47.m128i_i64[0] != v48 )
    _libc_free(v47.m128i_u64[0]);
  if ( (v40 & 2) != 0 )
    sub_14F4240(&v39, (__int64)v29, v30);
  if ( (v40 & 1) != 0 )
    goto LABEL_25;
  return a1;
}
