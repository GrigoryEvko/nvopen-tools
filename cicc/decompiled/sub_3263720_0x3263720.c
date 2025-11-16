// Function: sub_3263720
// Address: 0x3263720
//
__int64 __fastcall sub_3263720(_QWORD *a1, __int64 **a2, __int64 *a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // rbx
  __int64 *v8; // rdi
  __int64 v9; // r12
  __int64 v10; // rsi
  char v11; // r10
  __int64 *v12; // rdi
  __int64 v13; // r15
  unsigned __int64 v14; // r13
  __int64 v15; // rbx
  __int64 v16; // rsi
  __int64 *v17; // rax
  unsigned __int64 v18; // r13
  __int64 v19; // rbx
  __int64 v20; // rsi
  __int64 *v21; // rax
  __m128i v22; // xmm0
  __int64 v23; // rax
  __int64 v24; // rbx
  __int64 v25; // r12
  __m128i v27; // [rsp+0h] [rbp-160h] BYREF
  __int64 *v28; // [rsp+18h] [rbp-148h]
  _QWORD *v29; // [rsp+20h] [rbp-140h]
  _BYTE *v30; // [rsp+28h] [rbp-138h]
  __int64 v31; // [rsp+30h] [rbp-130h] BYREF
  int v32; // [rsp+38h] [rbp-128h]
  __int64 v33; // [rsp+40h] [rbp-120h] BYREF
  __int64 *v34; // [rsp+48h] [rbp-118h]
  __int64 v35; // [rsp+50h] [rbp-110h]
  int v36; // [rsp+58h] [rbp-108h]
  char v37; // [rsp+5Ch] [rbp-104h]
  char v38; // [rsp+60h] [rbp-100h] BYREF
  _BYTE *v39; // [rsp+A0h] [rbp-C0h] BYREF
  __int64 v40; // [rsp+A8h] [rbp-B8h]
  _BYTE v41[176]; // [rsp+B0h] [rbp-B0h] BYREF

  v7 = (unsigned int)a3;
  v39 = v41;
  v29 = a1;
  v8 = *a2;
  v30 = v41;
  v40 = 0x800000000LL;
  v34 = (__int64 *)&v38;
  v33 = 0;
  v35 = 8;
  v36 = 0;
  v37 = 1;
  v9 = *v8;
  v28 = &v31;
  v10 = *(_QWORD *)(v9 + 80);
  v31 = v10;
  if ( v10 )
    sub_B96E90((__int64)&v31, v10, 1);
  v32 = *(_DWORD *)(v9 + 72);
  if ( !(_DWORD)v7 )
    goto LABEL_22;
  v11 = v37;
  v12 = *a2;
  v13 = (unsigned int)(v7 - 1);
  v14 = 0;
  v15 = 16 * v7;
  do
  {
    while ( 1 )
    {
      while ( 1 )
      {
        v16 = v12[v14 / 8];
        if ( v11 )
          break;
LABEL_29:
        v14 += 16LL;
        sub_C8CC70((__int64)&v33, v16, (__int64)a3, a4, a5, a6);
        v11 = v37;
        v12 = *a2;
        if ( v14 == v15 )
          goto LABEL_11;
      }
      v17 = v34;
      a6 = HIDWORD(v35);
      a3 = &v34[HIDWORD(v35)];
      if ( v34 != a3 )
        break;
LABEL_33:
      if ( HIDWORD(v35) >= (unsigned int)v35 )
        goto LABEL_29;
      a6 = (unsigned int)(HIDWORD(v35) + 1);
      v14 += 16LL;
      ++HIDWORD(v35);
      *a3 = v16;
      v11 = v37;
      ++v33;
      v12 = *a2;
      if ( v14 == v15 )
        goto LABEL_11;
    }
    while ( v16 != *v17 )
    {
      if ( a3 == ++v17 )
        goto LABEL_33;
    }
    v14 += 16LL;
  }
  while ( v14 != v15 );
LABEL_11:
  v18 = 0;
  v19 = 16 * v13;
  v20 = **(_QWORD **)(*v12 + 40);
  if ( !v11 )
    goto LABEL_18;
LABEL_12:
  v21 = v34;
  a4 = HIDWORD(v35);
  a3 = &v34[HIDWORD(v35)];
  if ( v34 == a3 )
  {
LABEL_31:
    if ( HIDWORD(v35) >= (unsigned int)v35 )
      goto LABEL_18;
    ++HIDWORD(v35);
    *a3 = v20;
    ++v33;
LABEL_19:
    a4 = HIDWORD(v40);
    v22 = _mm_loadu_si128((const __m128i *)*(_QWORD *)((*a2)[v18 / 8] + 40));
    v23 = (unsigned int)v40;
    a3 = (__int64 *)((unsigned int)v40 + 1LL);
    if ( (unsigned __int64)a3 > HIDWORD(v40) )
    {
      v27 = v22;
      sub_C8D5F0((__int64)&v39, v30, (unsigned __int64)a3, 0x10u, a5, a6);
      v23 = (unsigned int)v40;
      v22 = _mm_load_si128(&v27);
    }
    *(__m128i *)&v39[16 * v23] = v22;
    LODWORD(v40) = v40 + 1;
    if ( v18 != v19 )
      goto LABEL_17;
  }
  else
  {
    while ( v20 != *v21 )
    {
      if ( a3 == ++v21 )
        goto LABEL_31;
    }
    while ( v18 != v19 )
    {
LABEL_17:
      v18 += 16LL;
      v20 = **(_QWORD **)((*a2)[v18 / 8] + 40);
      if ( v37 )
        goto LABEL_12;
LABEL_18:
      sub_C8CC70((__int64)&v33, v20, (__int64)a3, a4, a5, a6);
      if ( (_BYTE)a3 )
        goto LABEL_19;
    }
  }
LABEL_22:
  v24 = (__int64)v28;
  v25 = sub_3402E70(*v29, v28, &v39);
  if ( v31 )
    sub_B91220(v24, v31);
  if ( !v37 )
    _libc_free((unsigned __int64)v34);
  if ( v39 != v30 )
    _libc_free((unsigned __int64)v39);
  return v25;
}
