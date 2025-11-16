// Function: sub_33E6BC0
// Address: 0x33e6bc0
//
__int64 *__fastcall sub_33E6BC0(
        _QWORD *a1,
        int a2,
        __int64 a3,
        unsigned __int16 a4,
        __int64 a5,
        const __m128i *a6,
        unsigned __int64 a7,
        __int64 a8,
        unsigned __int64 *a9,
        __int64 a10)
{
  __int64 v12; // rax
  int v13; // eax
  __int64 v14; // rdx
  unsigned __int64 v15; // r8
  __int64 v16; // r8
  __int64 v17; // rax
  __int64 *v18; // rax
  __int64 *v19; // r12
  __m128i *v21; // rax
  __int32 v22; // r10d
  __int64 v23; // rsi
  unsigned __int64 v24; // rdx
  __int128 v25; // [rsp-20h] [rbp-130h]
  int v26; // [rsp+Ch] [rbp-104h]
  int v27; // [rsp+Ch] [rbp-104h]
  __int64 *v31; // [rsp+38h] [rbp-D8h]
  __m128i *v32; // [rsp+38h] [rbp-D8h]
  __int64 *v33; // [rsp+38h] [rbp-D8h]
  __int32 v34; // [rsp+38h] [rbp-D8h]
  __int64 *v35; // [rsp+48h] [rbp-C8h] BYREF
  _QWORD *v36; // [rsp+50h] [rbp-C0h] BYREF
  unsigned int v37; // [rsp+58h] [rbp-B8h]
  unsigned int v38; // [rsp+5Ch] [rbp-B4h]
  _QWORD v39[22]; // [rsp+60h] [rbp-B0h] BYREF

  v12 = a4;
  if ( !a4 )
    v12 = a5;
  v39[0] = v12;
  v36 = v39;
  v38 = 32;
  v37 = 2;
  sub_33C9670((__int64)&v36, a2, a7, a9, a10, (__int64)&v36);
  v13 = sub_2EAC1E0((__int64)a6);
  v14 = v37;
  v15 = v37 + 1LL;
  if ( v15 > v38 )
  {
    v27 = v13;
    sub_C8D5F0((__int64)&v36, v39, v37 + 1LL, 4u, v15, (__int64)&v36);
    v14 = v37;
    v13 = v27;
  }
  *((_DWORD *)v36 + v14) = v13;
  v16 = a6[2].m128i_u16[0];
  v17 = ++v37;
  if ( (unsigned __int64)v37 + 1 > v38 )
  {
    v26 = v16;
    sub_C8D5F0((__int64)&v36, v39, v37 + 1LL, 4u, v16, (__int64)&v36);
    v17 = v37;
    LODWORD(v16) = v26;
  }
  *((_DWORD *)v36 + v17) = v16;
  ++v37;
  v35 = 0;
  v18 = sub_33CCCF0((__int64)a1, (__int64)&v36, a3, (__int64 *)&v35);
  if ( v18 )
  {
    v31 = v18;
    sub_2EAC4C0((__m128i *)v18[14], a6);
    v19 = v31;
    goto LABEL_9;
  }
  v21 = (__m128i *)a1[52];
  v22 = *(_DWORD *)(a3 + 8);
  if ( v21 )
  {
    a1[52] = v21->m128i_i64[0];
  }
  else
  {
    v23 = a1[53];
    a1[63] += 120LL;
    v24 = (v23 + 7) & 0xFFFFFFFFFFFFFFF8LL;
    if ( a1[54] >= v24 + 120 && v23 )
    {
      a1[53] = v24 + 120;
      if ( !v24 )
        goto LABEL_15;
      v21 = (__m128i *)((v23 + 7) & 0xFFFFFFFFFFFFFFF8LL);
    }
    else
    {
      v34 = v22;
      v21 = (__m128i *)sub_9D1E70((__int64)(a1 + 53), 120, 120, 3);
      v22 = v34;
    }
  }
  *((_QWORD *)&v25 + 1) = a5;
  *(_QWORD *)&v25 = a4;
  v32 = v21;
  sub_33CF750(v21, a2, v22, (unsigned __int8 **)a3, a7, a8, v25, (__int64)a6);
  v21 = v32;
LABEL_15:
  v33 = (__int64 *)v21;
  sub_33E4EC0((__int64)a1, (__int64)v21, (__int64)a9, a10);
  sub_C657C0(a1 + 65, v33, v35, (__int64)off_4A367D0);
  sub_33CC420((__int64)a1, (__int64)v33);
  v19 = v33;
LABEL_9:
  if ( v36 != v39 )
    _libc_free((unsigned __int64)v36);
  return v19;
}
