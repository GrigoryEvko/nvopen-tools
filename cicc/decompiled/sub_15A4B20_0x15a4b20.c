// Function: sub_15A4B20
// Address: 0x15a4b20
//
__int64 __fastcall sub_15A4B20(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        int a7,
        unsigned int a8)
{
  __int16 v9; // r14
  __int64 v12; // rcx
  __m128i v13; // xmm1
  __m128i v14; // xmm2
  __m128i v15; // xmm4
  __m128i v16; // xmm5
  char v17; // al
  __int64 v18; // r8
  __int64 v19; // r14
  __int64 *v20; // rax
  unsigned int v21; // eax
  __int64 v22; // rcx
  __int64 i; // r14
  _QWORD *v24; // rdi
  int v26; // r14d
  __int64 v27; // rax
  unsigned int v28; // edx
  __int64 v29; // rcx
  int v30; // eax
  char v31; // al
  __int64 *v32; // rdx
  unsigned int v33; // esi
  int v34; // eax
  int v35; // eax
  int v36; // esi
  __int16 v39; // [rsp+16h] [rbp-CAh]
  char v40; // [rsp+18h] [rbp-C8h]
  __int64 v41; // [rsp+18h] [rbp-C8h]
  __int64 v42; // [rsp+18h] [rbp-C8h]
  __int64 *v43; // [rsp+28h] [rbp-B8h] BYREF
  _BYTE v44[24]; // [rsp+30h] [rbp-B0h] BYREF
  __m128i v45; // [rsp+48h] [rbp-98h] BYREF
  __m128i v46; // [rsp+58h] [rbp-88h]
  unsigned __int64 v47; // [rsp+70h] [rbp-70h] BYREF
  __m128i v48; // [rsp+78h] [rbp-68h] BYREF
  _OWORD v49[2]; // [rsp+88h] [rbp-58h] BYREF
  __int64 v50; // [rsp+A8h] [rbp-38h]

  v9 = 0;
  v39 = *(_WORD *)(a4 + 18);
  v40 = *(_BYTE *)(a4 + 17) >> 1;
  if ( sub_1594520(a4) )
    v9 = sub_1594720(a4);
  v48.m128i_i64[0] = a2;
  v48.m128i_i64[1] = a3;
  sub_1594700(a4);
  v12 = *(_QWORD *)a4;
  v49[0] = 0u;
  v13 = _mm_loadu_si128((const __m128i *)&v48.m128i_u64[1]);
  v14 = _mm_loadu_si128((const __m128i *)((char *)v49 + 8));
  WORD1(v47) = v9;
  *(_QWORD *)v44 = v12;
  v45 = v13;
  LOBYTE(v47) = v39;
  v46 = v14;
  BYTE1(v47) = v40;
  *(__m128i *)&v44[8] = _mm_loadu_si128((const __m128i *)&v47);
  v47 = sub_1597510((__int64 *)v13.m128i_i64[1], v13.m128i_i64[1]);
  v43 = (__int64 *)sub_1597240(*(__int64 **)&v44[16], *(_QWORD *)&v44[16] + 8 * v13.m128i_i64[0]);
  LODWORD(v47) = sub_1597150(&v44[8], &v44[9], (__int16 *)&v44[10], (__int64 *)&v43, (__int64 *)&v47);
  LODWORD(v47) = sub_15981B0((__int64 *)v44, (int *)&v47);
  v15 = _mm_loadu_si128((const __m128i *)&v44[16]);
  v16 = _mm_loadu_si128((const __m128i *)&v45.m128i_u64[1]);
  v48 = _mm_loadu_si128((const __m128i *)v44);
  v49[0] = v15;
  v50 = v46.m128i_i64[1];
  v49[1] = v16;
  v17 = sub_1598AB0(a1, (__int64)&v47, &v43);
  v18 = *(_QWORD *)(a1 + 8);
  if ( v17 )
  {
    v19 = *(unsigned int *)(a1 + 24);
    if ( v43 != (__int64 *)(v18 + 8 * v19) )
      return *v43;
  }
  else
  {
    v19 = *(unsigned int *)(a1 + 24);
  }
  if ( !(_DWORD)v19 )
    goto LABEL_6;
  v42 = *(_QWORD *)(a1 + 8);
  v26 = v19 - 1;
  v27 = v26 & (unsigned int)sub_159D500(a4);
  v28 = v27;
  v20 = (__int64 *)(v42 + 8 * v27);
  v29 = *v20;
  if ( *v20 != a4 )
  {
    v30 = 1;
    while ( v29 != -8 )
    {
      v36 = v30 + 1;
      v28 = v26 & (v30 + v28);
      v20 = (__int64 *)(v42 + 8LL * v28);
      v29 = *v20;
      if ( *v20 == a4 )
        goto LABEL_7;
      v30 = v36;
    }
    v18 = *(_QWORD *)(a1 + 8);
    v19 = *(unsigned int *)(a1 + 24);
LABEL_6:
    v20 = (__int64 *)(v18 + 8 * v19);
  }
LABEL_7:
  *v20 = -16;
  --*(_DWORD *)(a1 + 16);
  ++*(_DWORD *)(a1 + 20);
  if ( a7 == 1 )
  {
    sub_1593B40((_QWORD *)(a4 + 24 * (a8 - (unsigned __int64)(*(_DWORD *)(a4 + 20) & 0xFFFFFFF))), a6);
  }
  else
  {
    v21 = *(_DWORD *)(a4 + 20) & 0xFFFFFFF;
    if ( v21 )
    {
      v22 = v21 - 1;
      for ( i = 0; ; ++i )
      {
        v24 = (_QWORD *)(a4 + 24 * (i - v21));
        if ( a5 == *v24 )
        {
          v41 = v22;
          sub_1593B40(v24, a6);
          v22 = v41;
        }
        if ( v22 == i )
          break;
        v21 = *(_DWORD *)(a4 + 20) & 0xFFFFFFF;
      }
    }
  }
  v31 = sub_1598AB0(a1, (__int64)&v47, &v43);
  v32 = v43;
  if ( v31 )
    return 0;
  v33 = *(_DWORD *)(a1 + 24);
  v34 = *(_DWORD *)(a1 + 16);
  ++*(_QWORD *)a1;
  v35 = v34 + 1;
  if ( 4 * v35 >= 3 * v33 )
  {
    v33 *= 2;
    goto LABEL_29;
  }
  if ( v33 - *(_DWORD *)(a1 + 20) - v35 <= v33 >> 3 )
  {
LABEL_29:
    sub_15A25C0(a1, v33);
    sub_1598AB0(a1, (__int64)&v47, &v43);
    v32 = v43;
    v35 = *(_DWORD *)(a1 + 16) + 1;
  }
  *(_DWORD *)(a1 + 16) = v35;
  if ( *v32 != -8 )
    --*(_DWORD *)(a1 + 20);
  *v32 = a4;
  return 0;
}
