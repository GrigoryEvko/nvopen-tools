// Function: sub_36DAD70
// Address: 0x36dad70
//
void __fastcall sub_36DAD70(__int64 a1, __int64 a2, char a3, __m128i a4)
{
  __int64 v7; // rsi
  __int64 v8; // r9
  __int64 v9; // rax
  _QWORD *v10; // rdi
  __int64 v11; // rax
  _QWORD *v12; // rsi
  int v13; // eax
  __int64 v14; // r9
  __m128i v15; // xmm1
  __int64 v16; // r8
  unsigned __int64 *v17; // rdx
  unsigned __int64 **v18; // rdi
  __int64 v19; // rax
  __m128i v20; // xmm0
  unsigned __int64 *v21; // rdx
  __int64 v22; // rax
  _QWORD *v23; // r9
  unsigned __int64 v24; // rcx
  __int64 v25; // r8
  __int64 v26; // r12
  __int64 v27; // rcx
  __int64 v28; // r8
  __int64 v29; // r9
  __int64 v30; // rax
  _QWORD *v31; // rsi
  __int64 v32; // rax
  __int64 v33; // rdx
  unsigned __int64 v34; // rdx
  unsigned __int64 *v35; // rax
  __m128i v36; // [rsp+0h] [rbp-880h] BYREF
  __m128i v37; // [rsp+10h] [rbp-870h] BYREF
  int v38; // [rsp+28h] [rbp-858h]
  int v39; // [rsp+2Ch] [rbp-854h]
  __int64 v40; // [rsp+30h] [rbp-850h] BYREF
  int v41; // [rsp+38h] [rbp-848h]
  unsigned __int64 *v42; // [rsp+40h] [rbp-840h] BYREF
  __int64 v43; // [rsp+48h] [rbp-838h]
  _OWORD v44[131]; // [rsp+50h] [rbp-830h] BYREF

  v7 = *(_QWORD *)(a2 + 80);
  v40 = v7;
  if ( v7 )
    sub_B96E90((__int64)&v40, v7, 1);
  v8 = *(_QWORD *)(a2 + 40);
  v41 = *(_DWORD *)(a2 + 72);
  v9 = *(_QWORD *)(*(_QWORD *)(v8 + 40) + 96LL);
  v10 = *(_QWORD **)(v9 + 24);
  if ( *(_DWORD *)(v9 + 32) > 0x40u )
    v10 = (_QWORD *)*v10;
  v11 = *(_QWORD *)(*(_QWORD *)(v8 + 80) + 96LL);
  v12 = *(_QWORD **)(v11 + 24);
  if ( *(_DWORD *)(v11 + 32) > 0x40u )
    v12 = (_QWORD *)*v12;
  v13 = sub_36D6A40((int)v10, (unsigned __int8)v12 & 1, 0);
  v15 = _mm_loadu_si128((const __m128i *)(v14 + 120));
  v42 = (unsigned __int64 *)v44;
  v39 = v13;
  v43 = 0x8000000001LL;
  v44[0] = v15;
  if ( a3 )
  {
    LODWORD(v16) = 5;
    if ( *(_DWORD *)(a2 + 64) <= 5u )
      goto LABEL_23;
  }
  else
  {
    LODWORD(v16) = 4;
    if ( *(_DWORD *)(a2 + 64) <= 4u )
    {
      v20 = _mm_loadu_si128((const __m128i *)v14);
      v22 = 2;
      v21 = (unsigned __int64 *)v44;
      goto LABEL_17;
    }
  }
  v17 = (unsigned __int64 *)v44;
  v18 = &v42;
  a4 = _mm_loadu_si128((const __m128i *)(v14 + 40LL * (unsigned int)v16));
  v19 = 1;
  while ( 1 )
  {
    v16 = (unsigned int)(v16 + 1);
    *(__m128i *)&v17[2 * v19] = a4;
    v19 = (unsigned int)(v43 + 1);
    LODWORD(v43) = v43 + 1;
    if ( (unsigned int)v16 >= *(_DWORD *)(a2 + 64) )
      break;
    v14 = *(_QWORD *)(a2 + 40);
    a4 = _mm_loadu_si128((const __m128i *)(v14 + 40LL * (unsigned int)v16));
    if ( v19 + 1 > (unsigned __int64)HIDWORD(v43) )
    {
      v38 = v16;
      v37.m128i_i64[0] = (__int64)v18;
      v36 = a4;
      sub_C8D5F0((__int64)v18, v44, v19 + 1, 0x10u, v16, v14);
      v19 = (unsigned int)v43;
      LODWORD(v16) = v38;
      a4 = _mm_load_si128(&v36);
      v18 = (unsigned __int64 **)v37.m128i_i64[0];
    }
    v17 = v42;
  }
  if ( !a3 )
    goto LABEL_15;
  v14 = *(_QWORD *)(a2 + 40);
LABEL_23:
  v30 = *(_QWORD *)(*(_QWORD *)(v14 + 160) + 96LL);
  v31 = *(_QWORD **)(v30 + 24);
  if ( *(_DWORD *)(v30 + 32) > 0x40u )
    v31 = (_QWORD *)*v31;
  v16 = (__int64)sub_3400BD0(*(_QWORD *)(a1 + 64), (__int64)v31, (__int64)&v40, 7, 0, 1u, a4, 0);
  v32 = (unsigned int)v43;
  v14 = v33;
  v34 = (unsigned int)v43 + 1LL;
  if ( v34 > HIDWORD(v43) )
  {
    v37.m128i_i64[0] = v16;
    v37.m128i_i64[1] = v14;
    sub_C8D5F0((__int64)&v42, v44, v34, 0x10u, v16, v14);
    v32 = (unsigned int)v43;
    v14 = v37.m128i_i64[1];
    v16 = v37.m128i_i64[0];
  }
  v35 = &v42[2 * v32];
  *v35 = v16;
  v35[1] = v14;
  v19 = (unsigned int)(v43 + 1);
  LODWORD(v43) = v43 + 1;
LABEL_15:
  v20 = _mm_loadu_si128((const __m128i *)*(_QWORD *)(a2 + 40));
  if ( v19 + 1 > (unsigned __int64)HIDWORD(v43) )
  {
    v37 = v20;
    sub_C8D5F0((__int64)&v42, v44, v19 + 1, 0x10u, v16, v14);
    v21 = v42;
    v20 = _mm_load_si128(&v37);
    v22 = 2LL * (unsigned int)v43;
  }
  else
  {
    v21 = v42;
    v22 = 2 * v19;
  }
LABEL_17:
  *(__m128i *)&v21[v22] = v20;
  v23 = *(_QWORD **)(a1 + 64);
  v24 = *(_QWORD *)(a2 + 48);
  v25 = *(unsigned int *)(a2 + 68);
  LODWORD(v43) = v43 + 1;
  v26 = sub_33E66D0(v23, v39, (__int64)&v40, v24, v25, (__int64)v23, v42, (unsigned int)v43);
  sub_34158F0(*(_QWORD *)(a1 + 64), a2, v26, v27, v28, v29);
  sub_3421DB0(v26);
  sub_33ECEA0(*(const __m128i **)(a1 + 64), a2);
  if ( v42 != (unsigned __int64 *)v44 )
    _libc_free((unsigned __int64)v42);
  if ( v40 )
    sub_B91220((__int64)&v40, v40);
}
