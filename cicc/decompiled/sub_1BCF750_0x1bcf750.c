// Function: sub_1BCF750
// Address: 0x1bcf750
//
__int64 __fastcall sub_1BCF750(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v6; // r15
  unsigned int v7; // r13d
  __int64 v8; // rbx
  _QWORD *v9; // rdi
  __int64 v10; // rax
  __int64 v11; // r10
  __int64 *v12; // r14
  char *v13; // rbx
  __int64 v14; // rax
  __int64 v15; // r9
  __int64 v16; // rdx
  __int64 v17; // rcx
  const __m128i *v18; // r8
  __int64 v19; // rsi
  __int64 v20; // rax
  __int64 v21; // rax
  __int64 v22; // rax
  __m128i *v23; // rax
  _QWORD *v24; // rax
  _QWORD *v25; // rbx
  __int64 v26; // rdi
  unsigned __int64 *v27; // r15
  __int64 v28; // rax
  unsigned __int64 v29; // rsi
  __int64 v30; // rsi
  __int64 v31; // rsi
  __int64 v32; // rdx
  unsigned __int8 *v33; // rsi
  _DWORD *v34; // rax
  __int64 v38; // [rsp+18h] [rbp-98h]
  __int64 v39; // [rsp+20h] [rbp-90h]
  const __m128i *v40; // [rsp+20h] [rbp-90h]
  unsigned __int8 *v41; // [rsp+38h] [rbp-78h] BYREF
  char *v42[2]; // [rsp+40h] [rbp-70h] BYREF
  __int16 v43; // [rsp+50h] [rbp-60h]
  char *v44[2]; // [rsp+60h] [rbp-50h] BYREF
  int v45; // [rsp+70h] [rbp-40h]

  v6 = sub_1599EF0((__int64 **)a4);
  if ( *(_QWORD *)(a4 + 32) )
  {
    v7 = 0;
    v8 = 0;
    do
    {
      v9 = *(_QWORD **)(a1 + 1424);
      v43 = 257;
      v10 = sub_1643350(v9);
      v11 = sub_159C470(v10, v8, 0);
      v12 = (__int64 *)(a2 + 8 * v8);
      if ( *(_BYTE *)(v6 + 16) > 0x10u || *(_BYTE *)(*v12 + 16) > 0x10u || *(_BYTE *)(v11 + 16) > 0x10u )
      {
        v38 = v11;
        v39 = *v12;
        LOWORD(v45) = 257;
        v24 = sub_1648A60(56, 3u);
        v25 = v24;
        if ( v24 )
          sub_15FA480((__int64)v24, (__int64 *)v6, v39, v38, (__int64)v44, 0);
        v26 = *(_QWORD *)(a1 + 1408);
        if ( v26 )
        {
          v27 = *(unsigned __int64 **)(a1 + 1416);
          sub_157E9D0(v26 + 40, (__int64)v25);
          v28 = v25[3];
          v29 = *v27;
          v25[4] = v27;
          v29 &= 0xFFFFFFFFFFFFFFF8LL;
          v25[3] = v29 | v28 & 7;
          *(_QWORD *)(v29 + 8) = v25 + 3;
          *v27 = *v27 & 7 | (unsigned __int64)(v25 + 3);
        }
        sub_164B780((__int64)v25, (__int64 *)v42);
        v30 = *(_QWORD *)(a1 + 1400);
        if ( v30 )
        {
          v41 = *(unsigned __int8 **)(a1 + 1400);
          sub_1623A60((__int64)&v41, v30, 2);
          v31 = v25[6];
          v32 = (__int64)(v25 + 6);
          if ( v31 )
          {
            sub_161E7C0((__int64)(v25 + 6), v31);
            v32 = (__int64)(v25 + 6);
          }
          v33 = v41;
          v25[6] = v41;
          if ( v33 )
            sub_1623210((__int64)&v41, v33, v32);
        }
        v6 = (__int64)v25;
      }
      else
      {
        v6 = sub_15A3890((__int64 *)v6, *v12, v11, 0);
      }
      if ( *(_BYTE *)(v6 + 16) > 0x17u )
      {
        v42[0] = (char *)v6;
        sub_1BCF290(a1 + 1080, v42);
        v44[0] = *((char **)v42[0] + 5);
        sub_1BCF4F0(a1 + 1136, v44);
        v13 = (char *)*v12;
        v14 = sub_1BBCD20(a1, *v12);
        v16 = v14;
        if ( v14 )
        {
          LODWORD(v41) = -1;
          v17 = *(unsigned int *)(v14 + 8);
          v18 = (const __m128i *)v44;
          if ( (_DWORD)v17 )
          {
            v19 = *(_QWORD *)v14;
            v20 = 0;
            while ( v13 != *(char **)(v19 + 8 * v20) )
            {
              if ( v17 == ++v20 )
                goto LABEL_15;
            }
            LODWORD(v41) = v20;
          }
LABEL_15:
          v21 = *(unsigned int *)(v16 + 104);
          if ( (_DWORD)v21 )
          {
            v34 = sub_1BB97F0(*(_DWORD **)(v16 + 96), *(_QWORD *)(v16 + 96) + 4 * v21, (int *)&v41);
            LODWORD(v41) = ((__int64)v34 - v15) >> 2;
          }
          v44[0] = v13;
          v45 = (int)v41;
          v44[1] = v42[0];
          v22 = *(unsigned int *)(a1 + 392);
          if ( (unsigned int)v22 >= *(_DWORD *)(a1 + 396) )
          {
            v40 = v18;
            sub_16CD150(a1 + 384, (const void *)(a1 + 400), 0, 24, (int)v18, v15);
            v22 = *(unsigned int *)(a1 + 392);
            v18 = v40;
          }
          v23 = (__m128i *)(*(_QWORD *)(a1 + 384) + 24 * v22);
          *v23 = _mm_loadu_si128(v18);
          v23[1].m128i_i64[0] = v18[1].m128i_i64[0];
          ++*(_DWORD *)(a1 + 392);
        }
      }
      v8 = ++v7;
    }
    while ( (unsigned __int64)v7 < *(_QWORD *)(a4 + 32) );
  }
  return v6;
}
