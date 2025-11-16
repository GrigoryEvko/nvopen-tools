// Function: sub_20190A0
// Address: 0x20190a0
//
__int64 __fastcall sub_20190A0(__int64 *a1, __int64 a2, unsigned int a3, double a4, double a5, __m128i a6)
{
  __int64 v8; // rsi
  __int64 v9; // rax
  char v10; // dl
  const void **v11; // rax
  int v12; // ebx
  __m128i v13; // xmm0
  __int64 v14; // rax
  char v15; // dl
  const void **v16; // rax
  int v17; // r15d
  __int64 v18; // rax
  int v19; // r8d
  int v20; // r9d
  __int64 v21; // rdx
  __int64 v22; // rdx
  int v23; // r13d
  int i; // ebx
  _BYTE *v25; // rax
  int v26; // ecx
  int v27; // edx
  int v28; // ebx
  __int64 v29; // rdx
  __int64 *v30; // r12
  __int128 v31; // rax
  __int64 result; // rax
  int v33; // [rsp-10h] [rbp-F0h]
  int v34; // [rsp-8h] [rbp-E8h]
  __int64 v35; // [rsp+10h] [rbp-D0h]
  __int64 v36; // [rsp+18h] [rbp-C8h]
  __int64 v37; // [rsp+20h] [rbp-C0h]
  __int64 v38; // [rsp+20h] [rbp-C0h]
  __int64 v39; // [rsp+30h] [rbp-B0h] BYREF
  int v40; // [rsp+38h] [rbp-A8h]
  unsigned int v41; // [rsp+40h] [rbp-A0h] BYREF
  const void **v42; // [rsp+48h] [rbp-98h]
  unsigned int v43; // [rsp+50h] [rbp-90h] BYREF
  const void **v44; // [rsp+58h] [rbp-88h]
  char *v45; // [rsp+60h] [rbp-80h] BYREF
  __int64 v46; // [rsp+68h] [rbp-78h]
  _BYTE v47[112]; // [rsp+70h] [rbp-70h] BYREF

  v8 = *(_QWORD *)(a2 + 72);
  v39 = v8;
  if ( v8 )
    sub_1623A60((__int64)&v39, v8, 2);
  v40 = *(_DWORD *)(a2 + 64);
  v9 = *(_QWORD *)(a2 + 40) + 16LL * a3;
  v10 = *(_BYTE *)v9;
  v11 = *(const void ***)(v9 + 8);
  LOBYTE(v41) = v10;
  v42 = v11;
  if ( v10 )
    v12 = word_4301260[(unsigned __int8)(v10 - 14)];
  else
    v12 = sub_1F58D30((__int64)&v41);
  v13 = _mm_loadu_si128((const __m128i *)*(_QWORD *)(a2 + 32));
  v14 = *(_QWORD *)(**(_QWORD **)(a2 + 32) + 40LL) + 16LL * *(unsigned int *)(*(_QWORD *)(a2 + 32) + 8LL);
  v15 = *(_BYTE *)v14;
  v16 = *(const void ***)(v14 + 8);
  LOBYTE(v43) = v15;
  v44 = v16;
  if ( v15 )
    v17 = word_4301260[(unsigned __int8)(v15 - 14)];
  else
    v17 = sub_1F58D30((__int64)&v43);
  v18 = sub_1D38BB0(*a1, 0, (__int64)&v39, v43, v44, 0, v13, a5, a6, 0);
  v19 = v33;
  v45 = v47;
  v35 = v18;
  v20 = v34;
  v36 = v21;
  v46 = 0x1000000000LL;
  if ( (unsigned __int64)v17 > 0x10 )
    sub_16CD150((__int64)&v45, v47, v17, 4, v33, v34);
  v22 = (unsigned int)v46;
  if ( v17 > 0 )
  {
    v23 = v12;
    for ( i = 0; i != v17; ++i )
    {
      if ( (unsigned int)v22 >= HIDWORD(v46) )
      {
        sub_16CD150((__int64)&v45, v47, 0, 4, v19, v20);
        v22 = (unsigned int)v46;
      }
      *(_DWORD *)&v45[4 * v22] = i;
      v22 = (unsigned int)(v46 + 1);
      LODWORD(v46) = v46 + 1;
    }
    v12 = v23;
  }
  v25 = (_BYTE *)sub_1E0A0C0(*(_QWORD *)(*a1 + 32));
  v26 = v17 / v12;
  v27 = v17 / v12 - 1;
  if ( !*v25 )
    v27 = 0;
  if ( v12 > 0 )
  {
    v28 = v17 + v12;
    v29 = 4LL * v27;
    do
    {
      *(_DWORD *)&v45[v29] = v17++;
      v29 += 4LL * v26;
    }
    while ( v28 != v17 );
  }
  v30 = (__int64 *)*a1;
  *(_QWORD *)&v31 = sub_1D41320(
                      *a1,
                      v43,
                      v44,
                      (__int64)&v39,
                      v35,
                      v36,
                      *(double *)v13.m128i_i64,
                      a5,
                      a6,
                      v13.m128i_i64[0],
                      v13.m128i_i64[1],
                      v45,
                      (unsigned int)v46);
  result = sub_1D309E0(v30, 158, (__int64)&v39, v41, v42, 0, *(double *)v13.m128i_i64, a5, *(double *)a6.m128i_i64, v31);
  if ( v45 != v47 )
  {
    v37 = result;
    _libc_free((unsigned __int64)v45);
    result = v37;
  }
  if ( v39 )
  {
    v38 = result;
    sub_161E7C0((__int64)&v39, v39);
    return v38;
  }
  return result;
}
