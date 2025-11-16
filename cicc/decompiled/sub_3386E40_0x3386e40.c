// Function: sub_3386E40
// Address: 0x3386e40
//
void __fastcall sub_3386E40(__int64 a1, __int64 a2, __int64 a3, __int64 a4, char a5, __int64 *a6, int a7)
{
  __int64 v9; // rsi
  __int64 *v10; // r9
  __int64 v11; // r15
  __int64 *v12; // r12
  __int64 *v13; // r14
  __int64 v14; // rax
  __int64 v15; // r9
  __int64 v16; // rax
  unsigned __int64 v17; // rdx
  __int64 v18; // rcx
  const __m128i *v19; // rdx
  __m128i *v20; // rax
  int v21; // r8d
  _BYTE *v22; // rcx
  __int64 v23; // rax
  unsigned __int64 *v24; // rax
  unsigned __int64 v25; // r13
  unsigned __int64 *v26; // r12
  __int64 v27; // rsi
  __int64 v28; // rbx
  __int64 v29; // r15
  int v30; // eax
  unsigned __int8 *v31; // rsi
  __int64 v32; // [rsp+0h] [rbp-E0h]
  int v33; // [rsp+10h] [rbp-D0h]
  int v34; // [rsp+1Ch] [rbp-C4h]
  int v35; // [rsp+20h] [rbp-C0h]
  __int64 v36; // [rsp+30h] [rbp-B0h] BYREF
  __int64 v37; // [rsp+38h] [rbp-A8h] BYREF
  __int64 v38; // [rsp+48h] [rbp-98h] BYREF
  int v39; // [rsp+50h] [rbp-90h] BYREF
  __int64 v40; // [rsp+58h] [rbp-88h]
  int *v41; // [rsp+70h] [rbp-70h] BYREF
  __int64 v42; // [rsp+78h] [rbp-68h]
  _BYTE v43[96]; // [rsp+80h] [rbp-60h] BYREF

  v37 = a3;
  v36 = a4;
  if ( !a5 )
  {
    v24 = (unsigned __int64 *)sub_3386A80(a1 + 72, *(__int64 **)a2, a3);
    v25 = v24[1];
    v26 = v24;
    if ( v25 == v24[2] )
    {
      sub_3376670(v24, v24[1], &v37, &v36, a6, &a7);
      return;
    }
    v27 = *a6;
    v28 = v37;
    v29 = v36;
    v41 = (int *)v27;
    if ( v27 )
    {
      sub_B96E90((__int64)&v41, v27, 1);
      if ( !v25 )
      {
        if ( v41 )
          sub_B91220((__int64)&v41, (__int64)v41);
        goto LABEL_20;
      }
      v30 = a7;
    }
    else
    {
      v30 = a7;
      if ( !v25 )
      {
LABEL_20:
        v26[1] += 32LL;
        return;
      }
    }
    *(_DWORD *)v25 = v30;
    *(_QWORD *)(v25 + 8) = v28;
    *(_QWORD *)(v25 + 16) = v29;
    v31 = (unsigned __int8 *)v41;
    *(_QWORD *)(v25 + 24) = v41;
    if ( v31 )
      sub_B976B0((__int64)&v41, v31, v25 + 24);
    goto LABEL_20;
  }
  v9 = *a6;
  v35 = a4;
  v34 = a7;
  v38 = v9;
  if ( v9 )
    sub_B96E90((__int64)&v38, v9, 1);
  v10 = *(__int64 **)a2;
  v41 = (int *)v43;
  v11 = *(_QWORD *)(a1 + 864);
  v33 = v37;
  v42 = 0x200000000LL;
  v12 = &v10[*(unsigned int *)(a2 + 8)];
  if ( v10 == v12 )
  {
    v21 = 0;
    v22 = v43;
  }
  else
  {
    v13 = v10;
    do
    {
      v14 = sub_ACADE0(*(__int64 ***)(*v13 + 8));
      v39 = 1;
      v40 = v14;
      v16 = (unsigned int)v42;
      v17 = (unsigned int)v42 + 1LL;
      if ( v17 > HIDWORD(v42) )
      {
        if ( v41 > &v39 || &v39 >= &v41[6 * (unsigned int)v42] )
        {
          sub_C8D5F0((__int64)&v41, v43, v17, 0x18u, (__int64)v41, v15);
          v18 = (__int64)v41;
          v16 = (unsigned int)v42;
          v19 = (const __m128i *)&v39;
        }
        else
        {
          v32 = (__int64)v41;
          sub_C8D5F0((__int64)&v41, v43, v17, 0x18u, (__int64)v41, v15);
          v18 = (__int64)v41;
          v16 = (unsigned int)v42;
          v19 = (const __m128i *)((char *)&v39 + (_QWORD)v41 - v32);
        }
      }
      else
      {
        v18 = (__int64)v41;
        v19 = (const __m128i *)&v39;
      }
      ++v13;
      v20 = (__m128i *)(v18 + 24 * v16);
      *v20 = _mm_loadu_si128(v19);
      v20[1].m128i_i64[0] = v19[1].m128i_i64[0];
      v21 = v42 + 1;
      LODWORD(v42) = v42 + 1;
    }
    while ( v12 != v13 );
    LODWORD(v22) = (_DWORD)v41;
  }
  v23 = sub_33E4BC0(v11, v33, v35, (_DWORD)v22, v21, 0, 0, 0, (__int64)&v38, v34, 1);
  sub_33F99B0(v11, v23, 0);
  if ( v41 != (int *)v43 )
    _libc_free((unsigned __int64)v41);
  if ( v38 )
    sub_B91220((__int64)&v38, v38);
}
