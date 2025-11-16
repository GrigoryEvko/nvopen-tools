// Function: sub_3817A70
// Address: 0x3817a70
//
__int64 __fastcall sub_3817A70(__int64 a1, _QWORD *a2, __int64 a3)
{
  int v6; // edi
  __int32 v7; // r10d
  unsigned __int8 v8; // al
  unsigned int v9; // esi
  unsigned __int8 v10; // dl
  unsigned __int8 v11; // al
  __int64 v12; // rdx
  unsigned __int16 *v13; // rax
  int v14; // r13d
  __int64 v15; // r8
  __int64 (__fastcall *v16)(__int64, __int64, unsigned int); // r9
  _WORD *v17; // rax
  bool v18; // zf
  const __m128i *v19; // rax
  int v20; // esi
  __int64 v21; // rdi
  const __m128i *v22; // r10
  __int64 v23; // rdi
  const __m128i *v24; // rax
  unsigned __int64 v25; // rdx
  __m128i *v26; // rcx
  __int64 v27; // rax
  __int64 v28; // rdx
  __m128i v29; // xmm0
  __int64 v30; // rsi
  _WORD *v31; // r11
  __int64 *v32; // r10
  __int64 *v33; // rcx
  int v34; // eax
  __int64 v35; // r12
  __int16 v37; // si
  int v38; // eax
  __int64 v39; // rcx
  __int64 v40; // rax
  __int64 v41; // rdx
  __int64 v42; // r11
  const __m128i *v43; // rax
  unsigned __int64 v44; // r10
  unsigned __int64 v45; // rdx
  __m128i *v46; // rdx
  __int64 v47; // [rsp+8h] [rbp-F8h]
  __int64 v48; // [rsp+8h] [rbp-F8h]
  __int64 v49; // [rsp+10h] [rbp-F0h]
  const __m128i *v50; // [rsp+10h] [rbp-F0h]
  int v51; // [rsp+10h] [rbp-F0h]
  __int64 *v52; // [rsp+18h] [rbp-E8h]
  const __m128i *v53; // [rsp+18h] [rbp-E8h]
  const __m128i *v54; // [rsp+18h] [rbp-E8h]
  __int64 v55; // [rsp+18h] [rbp-E8h]
  __m128i v56; // [rsp+20h] [rbp-E0h] BYREF
  __int64 (__fastcall *v57)(__int64, __int64, unsigned int); // [rsp+30h] [rbp-D0h]
  __int64 *v58; // [rsp+38h] [rbp-C8h]
  __int64 v59; // [rsp+40h] [rbp-C0h] BYREF
  int v60; // [rsp+48h] [rbp-B8h]
  __int64 v61; // [rsp+50h] [rbp-B0h]
  __int64 v62; // [rsp+58h] [rbp-A8h]
  __int64 v63; // [rsp+60h] [rbp-A0h]
  __int64 v64; // [rsp+68h] [rbp-98h]
  __int64 v65; // [rsp+70h] [rbp-90h]
  _BYTE *v66; // [rsp+80h] [rbp-80h] BYREF
  __int64 v67; // [rsp+88h] [rbp-78h]
  _BYTE v68[112]; // [rsp+90h] [rbp-70h] BYREF

  v6 = *(_DWORD *)(a3 + 24);
  v7 = *(unsigned __int16 *)(a3 + 96);
  v8 = *(_BYTE *)(*(_QWORD *)(a3 + 112) + 37LL);
  v10 = v8 >> 4;
  v11 = v8 & 0xF;
  v9 = v11;
  if ( v11 == 4 && v10 == 5 || v11 == 5 && v10 == 4 )
  {
    v9 = 6;
  }
  else if ( !byte_3F70480[8 * v11 + v10] )
  {
    v9 = v10;
  }
  v56.m128i_i32[0] = v7;
  LODWORD(v58) = v6;
  v61 = 0;
  v12 = (int)sub_2FE6050(v6, v9, v7);
  v13 = *(unsigned __int16 **)(a3 + 48);
  v62 = 0;
  v63 = 0;
  v14 = v12;
  v15 = *v13;
  v16 = (__int64 (__fastcall *)(__int64, __int64, unsigned int))*((_QWORD *)v13 + 1);
  v64 = 0;
  v57 = (__int64 (__fastcall *)(__int64, __int64, unsigned int))&v66;
  v67 = 0x400000000LL;
  v17 = (_WORD *)*a2;
  LOBYTE(v65) = 4;
  v18 = *(_QWORD *)&v17[4 * v12 + 262644] == 0;
  v66 = v68;
  if ( v18 )
  {
    v37 = v56.m128i_i16[0];
    v56.m128i_i64[0] = v15;
    v58 = (__int64 *)v16;
    v38 = sub_2FE6480(v6, v37);
    v39 = (unsigned int)v67;
    v14 = v38;
    v40 = *(_QWORD *)(a3 + 40);
    v16 = (__int64 (__fastcall *)(__int64, __int64, unsigned int))v58;
    v15 = v56.m128i_i64[0];
    v41 = 40LL * *(unsigned int *)(a3 + 64);
    v42 = v40 + v41;
    v43 = (const __m128i *)(v40 + 40);
    v44 = 0xCCCCCCCCCCCCCCCDLL * ((v41 - 40) >> 3);
    v45 = v44 + (unsigned int)v67;
    if ( v45 > HIDWORD(v67) )
    {
      v48 = v56.m128i_i64[0];
      v51 = v44;
      v54 = v43;
      v56.m128i_i64[0] = v42;
      sub_C8D5F0((__int64)v57, v68, v45, 0x10u, v15, (__int64)v58);
      v39 = (unsigned int)v67;
      v15 = v48;
      LODWORD(v44) = v51;
      v43 = v54;
      v42 = v56.m128i_i64[0];
      v16 = (__int64 (__fastcall *)(__int64, __int64, unsigned int))v58;
    }
    v46 = (__m128i *)&v66[16 * v39];
    if ( v43 != (const __m128i *)v42 )
    {
      do
      {
        if ( v46 )
          *v46 = _mm_loadu_si128(v43);
        v43 = (const __m128i *)((char *)v43 + 40);
        ++v46;
      }
      while ( (const __m128i *)v42 != v43 );
      LODWORD(v39) = v67;
    }
    LODWORD(v67) = v44 + v39;
  }
  else
  {
    v19 = *(const __m128i **)(a3 + 40);
    v20 = 0;
    v21 = 40LL * *(unsigned int *)(a3 + 64);
    v22 = (const __m128i *)((char *)v19 + v21);
    v23 = v21 - 80;
    v24 = v19 + 5;
    v25 = 0xCCCCCCCCCCCCCCCDLL * (v23 >> 3);
    v26 = (__m128i *)v68;
    if ( (unsigned __int64)v23 > 0xA0 )
    {
      v47 = v15;
      v50 = v24;
      v53 = v22;
      v56.m128i_i64[0] = (__int64)v16;
      v58 = (__int64 *)(0xCCCCCCCCCCCCCCCDLL * (v23 >> 3));
      sub_C8D5F0((__int64)v57, v68, v25, 0x10u, v15, (__int64)v16);
      v20 = v67;
      v15 = v47;
      v24 = v50;
      v22 = v53;
      v16 = (__int64 (__fastcall *)(__int64, __int64, unsigned int))v56.m128i_i64[0];
      LODWORD(v25) = -858993459 * (v23 >> 3);
      v26 = (__m128i *)&v66[16 * (unsigned int)v67];
    }
    if ( v24 != v22 )
    {
      do
      {
        if ( v26 )
          *v26 = _mm_loadu_si128(v24);
        v24 = (const __m128i *)((char *)v24 + 40);
        ++v26;
      }
      while ( v22 != v24 );
      v20 = v67;
    }
    v27 = *(_QWORD *)(a3 + 40);
    LODWORD(v67) = v20 + v25;
    v28 = (unsigned int)(v20 + v25);
    v29 = _mm_loadu_si128((const __m128i *)(v27 + 40));
    if ( v28 + 1 > (unsigned __int64)HIDWORD(v67) )
    {
      v55 = v15;
      v58 = (__int64 *)v16;
      v56 = v29;
      sub_C8D5F0((__int64)v57, v68, v28 + 1, 0x10u, v15, (__int64)v16);
      v28 = (unsigned int)v67;
      v15 = v55;
      v29 = _mm_load_si128(&v56);
      v16 = (__int64 (__fastcall *)(__int64, __int64, unsigned int))v58;
    }
    *(__m128i *)&v66[16 * v28] = v29;
    LODWORD(v67) = v67 + 1;
  }
  v30 = *(_QWORD *)(a3 + 80);
  v31 = (_WORD *)*a2;
  v32 = &v59;
  v33 = *(__int64 **)(a3 + 40);
  v59 = v30;
  if ( v30 )
  {
    v49 = v15;
    v52 = v33;
    v57 = v16;
    v56.m128i_i64[0] = (__int64)v31;
    v58 = &v59;
    sub_B96E90((__int64)&v59, v30, 1);
    v15 = v49;
    v33 = v52;
    v16 = v57;
    v31 = (_WORD *)v56.m128i_i64[0];
    v32 = v58;
  }
  v34 = *(_DWORD *)(a3 + 72);
  v35 = a2[1];
  v58 = v32;
  v60 = v34;
  sub_3494590(
    a1,
    v31,
    v35,
    v14,
    v15,
    v16,
    (__int64)v66,
    (unsigned int)v67,
    v61,
    v62,
    v63,
    v64,
    v65,
    (__int64)v32,
    *v33,
    v33[1]);
  if ( v59 )
    sub_B91220((__int64)v58, v59);
  if ( v66 != v68 )
    _libc_free((unsigned __int64)v66);
  return a1;
}
