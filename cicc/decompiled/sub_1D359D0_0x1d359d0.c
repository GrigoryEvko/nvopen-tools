// Function: sub_1D359D0
// Address: 0x1d359d0
//
__int64 *__fastcall sub_1D359D0(
        __int64 *a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        const void **a5,
        unsigned __int64 a6,
        double a7,
        double a8,
        __m128i a9,
        __int128 a10)
{
  __int64 v10; // r10
  __int16 v12; // r13
  unsigned int v13; // ebx
  __int64 v14; // r8
  __int128 v15; // xmm0
  __int64 *result; // rax
  __int64 v17; // rax
  __int64 v18; // r8
  int v19; // edx
  __int64 v20; // r8
  __int64 v21; // r10
  __int64 *v22; // rbx
  __int64 v23; // rsi
  __int64 v24; // r8
  __int64 v25; // rsi
  int v26; // r14d
  __int64 v27; // rbx
  unsigned __int8 *v28; // rsi
  __int64 v29; // rsi
  int v30; // ecx
  unsigned __int8 *v31; // rsi
  __int64 v32; // rax
  __int64 v33; // rax
  int v34; // [rsp+8h] [rbp-F8h]
  __int64 v35; // [rsp+10h] [rbp-F0h]
  __int64 *v36; // [rsp+18h] [rbp-E8h]
  __int64 v37; // [rsp+20h] [rbp-E0h]
  __int64 v38; // [rsp+20h] [rbp-E0h]
  __int64 v39; // [rsp+20h] [rbp-E0h]
  __int64 v40; // [rsp+20h] [rbp-E0h]
  __int64 v41; // [rsp+28h] [rbp-D8h]
  __int64 v42; // [rsp+28h] [rbp-D8h]
  __int64 v43; // [rsp+28h] [rbp-D8h]
  __int64 *v44; // [rsp+28h] [rbp-D8h]
  __int64 v46; // [rsp+28h] [rbp-D8h]
  int v47; // [rsp+28h] [rbp-D8h]
  __int64 v48; // [rsp+28h] [rbp-D8h]
  __int64 v49; // [rsp+28h] [rbp-D8h]
  int v50; // [rsp+28h] [rbp-D8h]
  __int64 *v51; // [rsp+30h] [rbp-D0h] BYREF
  unsigned __int8 *v52; // [rsp+38h] [rbp-C8h] BYREF
  unsigned __int64 v53[2]; // [rsp+40h] [rbp-C0h] BYREF
  _BYTE v54[176]; // [rsp+50h] [rbp-B0h] BYREF

  v10 = a3;
  v12 = a2;
  v13 = a4;
  v14 = *((_QWORD *)&a10 + 1);
  if ( DWORD2(a10) == 2 )
    return sub_1D332F0(a1, a2, a3, a4, a5, a6, a7, a8, a9, *(_QWORD *)a10, *(_QWORD *)(a10 + 8), *(_OWORD *)(a10 + 16));
  if ( DWORD2(a10) <= 2 )
  {
    if ( !DWORD2(a10) )
      return sub_1D2B300(a1, a2, a3, a4, (__int64)a5, a6);
    v15 = (__int128)_mm_loadu_si128((const __m128i *)a10);
    return (__int64 *)sub_1D309E0(a1, a2, a3, a4, a5, a6, *(double *)&v15, a8, *(double *)a9.m128i_i64, v15);
  }
  if ( DWORD2(a10) == 3 )
    return (__int64 *)sub_1D3A900(
                        (_DWORD)a1,
                        a2,
                        a3,
                        a4,
                        (_DWORD)a5,
                        0,
                        *(_QWORD *)a10,
                        *(_QWORD *)(a10 + 8),
                        *(_OWORD *)(a10 + 16),
                        *(_QWORD *)(a10 + 32),
                        *(_QWORD *)(a10 + 40));
  if ( (_DWORD)a2 != 107
    || (result = (__int64 *)sub_1D374B0(a3, (unsigned int)a4, a5, a10, *((_QWORD *)&a10 + 1), a1),
        v10 = a3,
        v14 = *((_QWORD *)&a10 + 1),
        !result) )
  {
    v37 = v14;
    v41 = v10;
    v17 = sub_1D29190((__int64)a1, v13, (__int64)a5, a4, v14, a6);
    v18 = v37;
    v35 = v17;
    v34 = v19;
    if ( (_BYTE)v13 == 111 )
    {
      v25 = *(_QWORD *)v41;
      v26 = *(_DWORD *)(v41 + 8);
      v53[0] = v25;
      if ( v25 )
      {
        sub_1623A60((__int64)v53, v25, 2);
        v18 = v37;
      }
      v27 = a1[26];
      if ( v27 )
      {
        a1[26] = *(_QWORD *)v27;
      }
      else
      {
        v49 = v18;
        v32 = sub_145CBF0(a1 + 27, 112, 8);
        v18 = v49;
        v27 = v32;
      }
      *(_QWORD *)v27 = 0;
      *(_QWORD *)(v27 + 8) = 0;
      *(_QWORD *)(v27 + 40) = v35;
      *(_QWORD *)(v27 + 16) = 0;
      *(_WORD *)(v27 + 24) = v12;
      *(_DWORD *)(v27 + 28) = -1;
      *(_QWORD *)(v27 + 32) = 0;
      *(_QWORD *)(v27 + 48) = 0;
      *(_DWORD *)(v27 + 56) = 0;
      *(_DWORD *)(v27 + 60) = v34;
      *(_DWORD *)(v27 + 64) = v26;
      v28 = (unsigned __int8 *)v53[0];
      *(_QWORD *)(v27 + 72) = v53[0];
      if ( v28 )
      {
        v46 = v18;
        sub_1623210((__int64)v53, v28, v27 + 72);
        v18 = v46;
      }
      *(_WORD *)(v27 + 80) &= 0xF000u;
      *(_WORD *)(v27 + 26) = 0;
      sub_1D23B60((__int64)a1, v27, a10, v18);
    }
    else
    {
      v53[0] = (unsigned __int64)v54;
      v53[1] = 0x2000000000LL;
      sub_16BD430((__int64)v53, (unsigned __int16)a2);
      sub_16BD4C0((__int64)v53, v35);
      v20 = v37;
      v21 = v41;
      v22 = (__int64 *)a10;
      v36 = (__int64 *)(a10 + 16 * v37);
      if ( (__int64 *)a10 != v36 )
      {
        do
        {
          v23 = *v22;
          v22 += 2;
          v38 = v20;
          v42 = v21;
          sub_16BD4C0((__int64)v53, v23);
          sub_16BD430((__int64)v53, *((_DWORD *)v22 - 2));
          v21 = v42;
          v20 = v38;
        }
        while ( v36 != v22 );
      }
      v39 = v20;
      v43 = v21;
      v51 = 0;
      result = sub_1D17920((__int64)a1, (__int64)v53, v21, (__int64 *)&v51);
      v24 = v39;
      if ( result )
      {
        if ( (_BYTE *)v53[0] != v54 )
        {
          v44 = result;
          _libc_free(v53[0]);
          return v44;
        }
        return result;
      }
      v29 = *(_QWORD *)v43;
      v30 = *(_DWORD *)(v43 + 8);
      v52 = (unsigned __int8 *)v29;
      if ( v29 )
      {
        v47 = v30;
        sub_1623A60((__int64)&v52, v29, 2);
        v24 = v39;
        v30 = v47;
      }
      v27 = a1[26];
      if ( v27 )
      {
        a1[26] = *(_QWORD *)v27;
      }
      else
      {
        v40 = v24;
        v50 = v30;
        v33 = sub_145CBF0(a1 + 27, 112, 8);
        v30 = v50;
        v24 = v40;
        v27 = v33;
      }
      *(_QWORD *)v27 = 0;
      v31 = v52;
      *(_QWORD *)(v27 + 8) = 0;
      *(_QWORD *)(v27 + 40) = v35;
      *(_QWORD *)(v27 + 16) = 0;
      *(_WORD *)(v27 + 24) = v12;
      *(_DWORD *)(v27 + 28) = -1;
      *(_QWORD *)(v27 + 32) = 0;
      *(_QWORD *)(v27 + 48) = 0;
      *(_DWORD *)(v27 + 56) = 0;
      *(_DWORD *)(v27 + 60) = v34;
      *(_DWORD *)(v27 + 64) = v30;
      *(_QWORD *)(v27 + 72) = v31;
      if ( v31 )
      {
        v48 = v24;
        sub_1623210((__int64)&v52, v31, v27 + 72);
        v24 = v48;
      }
      *(_WORD *)(v27 + 80) &= 0xF000u;
      *(_WORD *)(v27 + 26) = 0;
      sub_1D23B60((__int64)a1, v27, a10, v24);
      sub_16BDA20(a1 + 40, (__int64 *)v27, v51);
      if ( (_BYTE *)v53[0] != v54 )
        _libc_free(v53[0]);
    }
    sub_1D172A0((__int64)a1, v27);
    return (__int64 *)v27;
  }
  return result;
}
