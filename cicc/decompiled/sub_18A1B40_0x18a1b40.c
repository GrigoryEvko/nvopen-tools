// Function: sub_18A1B40
// Address: 0x18a1b40
//
__int64 __fastcall sub_18A1B40(
        __int64 a1,
        __int64 a2,
        __m128 a3,
        __m128i a4,
        __m128i a5,
        __m128i a6,
        double a7,
        double a8,
        double a9,
        __m128 a10)
{
  __int64 v10; // r13
  __int64 v11; // rax
  __int64 v12; // rdx
  __int64 v13; // r12
  char *v14; // rax
  char *v15; // r14
  char *v16; // rdx
  __int64 v17; // r12
  _BYTE *v18; // rsi
  _BYTE *v19; // r13
  unsigned int v20; // r14d
  __int64 *v21; // r12
  __int64 v22; // r15
  _QWORD *v23; // rax
  double v24; // xmm4_8
  double v25; // xmm5_8
  __int64 v26; // rdx
  _BYTE *v27; // rsi
  __int64 v29; // [rsp+10h] [rbp-60h] BYREF
  __int64 v30; // [rsp+18h] [rbp-58h] BYREF
  void *src; // [rsp+20h] [rbp-50h] BYREF
  _BYTE *v32; // [rsp+28h] [rbp-48h]
  char *v33; // [rsp+30h] [rbp-40h]

  v10 = a2 + 24;
  v11 = *(_QWORD *)(a2 + 32);
  src = 0;
  v32 = 0;
  v33 = 0;
  if ( a2 + 24 == v11 )
    goto LABEL_36;
  v12 = 0;
  do
  {
    v11 = *(_QWORD *)(v11 + 8);
    ++v12;
  }
  while ( v10 != v11 );
  if ( v12 > 0xFFFFFFFFFFFFFFFLL )
    sub_4262D8((__int64)"vector::reserve");
  v13 = 8 * v12;
  v14 = (char *)sub_22077B0(8 * v12);
  v15 = v14;
  if ( v32 - (_BYTE *)src > 0 )
  {
    memmove(v14, src, v32 - (_BYTE *)src);
    j_j___libc_free_0(src, v33 - (_BYTE *)src);
  }
  v16 = &v15[v13];
  v17 = *(_QWORD *)(a2 + 32);
  src = v15;
  v32 = v15;
  v33 = v16;
  if ( v10 == v17 )
  {
LABEL_36:
    v19 = src;
    v20 = 0;
    goto LABEL_30;
  }
  do
  {
    while ( 1 )
    {
      if ( !v17 )
        BUG();
      if ( !*(_QWORD *)(v17 - 48) || sub_15E4F60(v17 - 56) )
        goto LABEL_9;
      v29 = v17 - 56;
      v18 = v32;
      if ( v32 != v33 )
        break;
      sub_17E9700((__int64)&src, v32, &v29);
LABEL_9:
      v17 = *(_QWORD *)(v17 + 8);
      if ( v10 == v17 )
        goto LABEL_17;
    }
    if ( v32 )
    {
      *(_QWORD *)v32 = v17 - 56;
      v18 = v32;
    }
    v32 = v18 + 8;
    v17 = *(_QWORD *)(v17 + 8);
  }
  while ( v10 != v17 );
LABEL_17:
  v19 = v32;
  v20 = 0;
  if ( v32 != src )
  {
    while ( 1 )
    {
      v21 = (__int64 *)*((_QWORD *)v19 - 1);
      v19 -= 8;
      v32 = v19;
      v22 = v21[1];
      if ( v22 )
        break;
LABEL_29:
      if ( src == v19 )
        goto LABEL_30;
    }
    do
    {
      v23 = sub_1648700(v22);
      if ( *((_BYTE *)v23 + 16) > 0x17u && v21 == *(__int64 **)(v23[5] + 56LL) )
        goto LABEL_29;
      v22 = *(_QWORD *)(v22 + 8);
    }
    while ( v22 );
    LODWORD(v29) = sub_189E6F0(a1, v21, a3, a4, a5, a6, v24, v25, a9, a10);
    v30 = v26;
    if ( v26 )
    {
      v27 = v32;
      if ( v32 != v33 )
      {
        if ( v32 )
        {
          *(_QWORD *)v32 = v26;
          v27 = v32;
        }
        v19 = v27 + 8;
        v32 = v27 + 8;
        goto LABEL_27;
      }
      sub_14F2380((__int64)&src, v32, &v30);
    }
    v19 = v32;
LABEL_27:
    if ( (_BYTE)v29 )
      v20 = (unsigned __int8)v29;
    goto LABEL_29;
  }
LABEL_30:
  if ( v19 )
    j_j___libc_free_0(v19, v33 - v19);
  return v20;
}
