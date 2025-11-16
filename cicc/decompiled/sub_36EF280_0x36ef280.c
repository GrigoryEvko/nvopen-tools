// Function: sub_36EF280
// Address: 0x36ef280
//
void __fastcall sub_36EF280(__int64 a1, __int64 a2, unsigned int a3)
{
  __int64 v3; // r8
  unsigned int v6; // edx
  const __m128i *v7; // rbx
  __int64 v8; // rax
  _QWORD *v9; // rcx
  __int64 v10; // rax
  __int64 v11; // r14
  __int64 v12; // rdx
  unsigned __int64 *v13; // r10
  __int64 v14; // rsi
  __int64 v15; // r12
  const __m128i *v16; // rbx
  int v17; // ecx
  unsigned __int64 v18; // rdx
  const __m128i *v19; // r11
  unsigned __int64 v20; // r12
  __m128i *v21; // rax
  const __m128i *v22; // rax
  __int64 v23; // r12
  __m128i v24; // xmm0
  unsigned int v25; // eax
  int v26; // esi
  _QWORD *v27; // rdi
  unsigned __int64 v28; // rcx
  __int64 v29; // r8
  __int64 v30; // r12
  __int64 v31; // rcx
  __int64 v32; // r8
  __int64 v33; // r9
  unsigned int v34; // [rsp+4h] [rbp-13Ch]
  const __m128i *v35; // [rsp+8h] [rbp-138h]
  char v36; // [rsp+8h] [rbp-138h]
  __m128i v37; // [rsp+10h] [rbp-130h] BYREF
  unsigned __int64 *v38; // [rsp+20h] [rbp-120h]
  unsigned __int64 *v39; // [rsp+28h] [rbp-118h]
  __int64 v40; // [rsp+30h] [rbp-110h] BYREF
  int v41; // [rsp+38h] [rbp-108h]
  unsigned __int64 *v42; // [rsp+40h] [rbp-100h] BYREF
  __int64 v43; // [rsp+48h] [rbp-F8h]
  _BYTE v44[240]; // [rsp+50h] [rbp-F0h] BYREF

  v3 = a3;
  v6 = *(_DWORD *)(a2 + 64);
  v7 = *(const __m128i **)(a2 + 40);
  if ( (_BYTE)v3 )
  {
    v8 = *(_QWORD *)(v7[2].m128i_i64[1] + 96);
    v9 = *(_QWORD **)(v8 + 24);
    if ( *(_DWORD *)(v8 + 32) > 0x40u )
      v9 = (_QWORD *)*v9;
    if ( (_DWORD)v9 == 8334 )
      goto LABEL_8;
    if ( (unsigned int)v9 <= 0x208E )
    {
      if ( (_DWORD)v9 == 8332 )
      {
LABEL_37:
        v10 = 4;
        v11 = 3;
        goto LABEL_9;
      }
      if ( (_DWORD)v9 != 8333 )
        goto LABEL_46;
    }
    else if ( (_DWORD)v9 != 9214 )
    {
      if ( (_DWORD)v9 == 9215 )
      {
LABEL_8:
        v10 = 8;
        v11 = 5;
        goto LABEL_9;
      }
      if ( (_DWORD)v9 != 9213 )
        goto LABEL_46;
      goto LABEL_37;
    }
    v10 = 6;
    v11 = 4;
    goto LABEL_9;
  }
  v11 = v6 - 5LL;
  v10 = v11;
LABEL_9:
  v12 = *(_QWORD *)(*((_QWORD *)&v7[-2] + 5 * v6 - 1) + 96LL);
  v13 = *(unsigned __int64 **)(v12 + 24);
  if ( *(_DWORD *)(v12 + 32) > 0x40u )
    v13 = (unsigned __int64 *)*v13;
  v14 = *(_QWORD *)(a2 + 80);
  v40 = v14;
  v15 = (v13 == (unsigned __int64 *)1) + v10 + 1;
  if ( v14 )
  {
    LODWORD(v38) = v3;
    v39 = v13;
    sub_B96E90((__int64)&v40, v14, 1);
    v7 = *(const __m128i **)(a2 + 40);
    v3 = (unsigned int)v38;
    v13 = v39;
  }
  v16 = v7 + 5;
  v17 = 0;
  v18 = 40 * v15;
  v41 = *(_DWORD *)(a2 + 72);
  v19 = (const __m128i *)((char *)v16 + 40 * v15);
  v43 = 0xC00000000LL;
  v20 = 0xCCCCCCCCCCCCCCCDLL * ((40 * v15) >> 3);
  v42 = (unsigned __int64 *)v44;
  v21 = (__m128i *)v44;
  if ( v18 > 0x1E0 )
  {
    v34 = v3;
    v35 = v19;
    v37.m128i_i64[0] = (__int64)v13;
    v38 = (unsigned __int64 *)v44;
    v39 = (unsigned __int64 *)&v42;
    sub_C8D5F0((__int64)&v42, v44, v20, 0x10u, v3, (__int64)v44);
    v17 = v43;
    v3 = v34;
    v19 = v35;
    v13 = (unsigned __int64 *)v37.m128i_i64[0];
    v21 = (__m128i *)&v42[2 * (unsigned int)v43];
  }
  if ( v16 != v19 )
  {
    do
    {
      if ( v21 )
        *v21 = _mm_loadu_si128(v16);
      v16 = (const __m128i *)((char *)v16 + 40);
      ++v21;
    }
    while ( v19 != v16 );
    v17 = v43;
  }
  v22 = *(const __m128i **)(a2 + 40);
  LODWORD(v43) = v17 + v20;
  v23 = (unsigned int)(v17 + v20);
  v24 = _mm_loadu_si128(v22);
  if ( v23 + 1 > (unsigned __int64)HIDWORD(v43) )
  {
    v36 = v3;
    v38 = v13;
    v39 = (unsigned __int64 *)v44;
    v37 = v24;
    sub_C8D5F0((__int64)&v42, v44, v23 + 1, 0x10u, v3, (__int64)v44);
    v23 = (unsigned int)v43;
    LOBYTE(v3) = v36;
    v24 = _mm_load_si128(&v37);
    v13 = v38;
  }
  *(__m128i *)&v42[2 * v23] = v24;
  v25 = v43 + 1;
  LODWORD(v43) = v43 + 1;
  if ( !(_BYTE)v3 )
  {
    switch ( v11 )
    {
      case 1LL:
        v26 = (v13 == (unsigned __int64 *)1) + 887;
        goto LABEL_28;
      case 2LL:
        v26 = (v13 == (unsigned __int64 *)1) + 889;
        goto LABEL_28;
      case 3LL:
        v26 = (v13 == (unsigned __int64 *)1) + 893;
        goto LABEL_28;
      case 4LL:
        v26 = (v13 == (unsigned __int64 *)1) + 897;
        goto LABEL_28;
      case 5LL:
        v26 = (v13 == (unsigned __int64 *)1) + 901;
        goto LABEL_28;
      default:
        goto LABEL_46;
    }
  }
  if ( v11 != 4 )
  {
    if ( v11 == 5 )
    {
      v26 = (v13 == (unsigned __int64 *)1) + 899;
      goto LABEL_28;
    }
    if ( v11 == 3 )
    {
      v26 = (v13 == (unsigned __int64 *)1) + 891;
      goto LABEL_28;
    }
LABEL_46:
    BUG();
  }
  v26 = (v13 == (unsigned __int64 *)1) + 895;
LABEL_28:
  v27 = *(_QWORD **)(a1 + 64);
  v28 = *(_QWORD *)(a2 + 48);
  v29 = *(unsigned int *)(a2 + 68);
  v39 = (unsigned __int64 *)v44;
  v30 = sub_33E66D0(v27, v26, (__int64)&v40, v28, v29, (__int64)v44, v42, v25);
  sub_34158F0(*(_QWORD *)(a1 + 64), a2, v30, v31, v32, v33);
  sub_3421DB0(v30);
  sub_33ECEA0(*(const __m128i **)(a1 + 64), a2);
  if ( v42 != v39 )
    _libc_free((unsigned __int64)v42);
  if ( v40 )
    sub_B91220((__int64)&v40, v40);
}
