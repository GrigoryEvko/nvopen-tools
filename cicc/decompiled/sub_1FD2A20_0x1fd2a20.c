// Function: sub_1FD2A20
// Address: 0x1fd2a20
//
__int64 __fastcall sub_1FD2A20(__int64 a1, __int64 a2, __m128 a3, __m128 a4, __m128i a5)
{
  __int64 result; // rax
  __int64 v8; // rax
  __int64 v9; // rsi
  unsigned __int8 v10; // r15
  const void **v11; // r14
  __int64 v12; // rdx
  unsigned int v13; // eax
  unsigned int v14; // esi
  __int64 v15; // rsi
  unsigned int v16; // r10d
  const void **v17; // r11
  unsigned __int8 v18; // cl
  __int64 v19; // rax
  __int64 *v20; // rdi
  __m128i v21; // rax
  __int64 *v22; // rdi
  __int128 v23; // rax
  __int64 *v24; // rdi
  __int64 *v25; // rsi
  unsigned int v26; // edx
  unsigned __int8 *v27; // rax
  __int64 v28; // rax
  __int64 v29; // rax
  const void **v30; // rdx
  __int128 v31; // rax
  __int64 *v32; // rax
  __int64 *v33; // rdi
  __int64 v34; // rdx
  __int64 v35; // rdx
  __int64 *v36; // rdi
  unsigned int v37; // edx
  const void **v38; // rdx
  __int128 v39; // [rsp-10h] [rbp-100h]
  unsigned int v40; // [rsp+8h] [rbp-E8h]
  const void **v41; // [rsp+10h] [rbp-E0h]
  __int64 v42; // [rsp+18h] [rbp-D8h]
  __int64 v43; // [rsp+20h] [rbp-D0h]
  __int64 v44; // [rsp+28h] [rbp-C8h]
  __int64 v45; // [rsp+38h] [rbp-B8h]
  unsigned int v46; // [rsp+40h] [rbp-B0h]
  __int64 *v47; // [rsp+40h] [rbp-B0h]
  char v48; // [rsp+48h] [rbp-A8h]
  unsigned int v49; // [rsp+4Ch] [rbp-A4h]
  __int128 v50; // [rsp+50h] [rbp-A0h] BYREF
  __m128i v51; // [rsp+60h] [rbp-90h] BYREF
  __int64 v52; // [rsp+70h] [rbp-80h]
  __int64 v53; // [rsp+78h] [rbp-78h]
  __int64 *v54; // [rsp+80h] [rbp-70h]
  __int64 v55; // [rsp+88h] [rbp-68h]
  __int64 v56; // [rsp+90h] [rbp-60h] BYREF
  int v57; // [rsp+98h] [rbp-58h]
  _OWORD v58[5]; // [rsp+A0h] [rbp-50h] BYREF

  result = sub_1FD24F0(a1, a2, 0x70u, a3, a4, a5);
  if ( !result )
  {
    v8 = *(_QWORD *)(a2 + 40);
    v9 = *(_QWORD *)(a2 + 72);
    v10 = *(_BYTE *)v8;
    v11 = *(const void ***)(v8 + 8);
    v56 = v9;
    if ( v9 )
      sub_1623A60((__int64)&v56, v9, 2);
    v57 = *(_DWORD *)(a2 + 64);
    if ( !v10 || (unsigned __int8)(v10 - 14) <= 0x5Fu )
      goto LABEL_6;
    v13 = sub_1F6C8D0(v10);
    v14 = 2 * v13;
    v49 = v13;
    if ( 2 * v13 == 32 )
    {
      v15 = *(_QWORD *)(a1 + 8);
      v16 = 5;
      v17 = 0;
      v18 = 5;
      goto LABEL_15;
    }
    if ( v14 > 0x20 )
    {
      if ( v14 == 64 )
      {
        v15 = *(_QWORD *)(a1 + 8);
        v16 = 6;
        v17 = 0;
        v18 = 6;
        goto LABEL_15;
      }
      if ( v14 == 128 )
      {
        v15 = *(_QWORD *)(a1 + 8);
        v16 = 7;
        v17 = 0;
        v18 = 7;
        goto LABEL_15;
      }
    }
    else
    {
      if ( v14 == 8 )
      {
        v15 = *(_QWORD *)(a1 + 8);
        v16 = 3;
        v17 = 0;
        v18 = 3;
        goto LABEL_15;
      }
      if ( v14 == 16 )
      {
        v15 = *(_QWORD *)(a1 + 8);
        v16 = 4;
        v17 = 0;
        v18 = 4;
        goto LABEL_15;
      }
    }
    LODWORD(v19) = sub_1F58CC0(*(_QWORD **)(*(_QWORD *)a1 + 48LL), v14);
    v15 = *(_QWORD *)(a1 + 8);
    v16 = v19;
    v18 = v19;
    v17 = v38;
    v19 = (unsigned __int8)v19;
    if ( (_BYTE)v16 == 1 )
    {
LABEL_16:
      if ( !*(_BYTE *)(v15 + 259 * v19 + 2476) )
      {
        LOBYTE(v16) = v18;
        v20 = *(__int64 **)a1;
        v39 = *(_OWORD *)*(_QWORD *)(a2 + 32);
        v46 = v16;
        *(_QWORD *)&v50 = v17;
        v21.m128i_i64[0] = sub_1D309E0(
                             v20,
                             143,
                             (__int64)&v56,
                             v16,
                             v17,
                             0,
                             *(double *)a3.m128_u64,
                             *(double *)a4.m128_u64,
                             *(double *)a5.m128i_i64,
                             v39);
        v22 = *(__int64 **)a1;
        v51 = v21;
        *(_QWORD *)&v23 = sub_1D309E0(
                            v22,
                            143,
                            (__int64)&v56,
                            v46,
                            (const void **)v50,
                            0,
                            *(double *)a3.m128_u64,
                            *(double *)a4.m128_u64,
                            *(double *)a5.m128i_i64,
                            *(_OWORD *)(*(_QWORD *)(a2 + 32) + 40LL));
        v24 = *(__int64 **)a1;
        v40 = v46;
        v41 = (const void **)v50;
        v50 = v23;
        v25 = sub_1D332F0(
                v24,
                54,
                (__int64)&v56,
                v46,
                v41,
                0,
                *(double *)a3.m128_u64,
                *(double *)a4.m128_u64,
                a5,
                v51.m128i_i64[0],
                v51.m128i_u64[1],
                v23);
        v45 = v26;
        v27 = (unsigned __int8 *)(v25[5] + 16LL * v26);
        v44 = *(_QWORD *)(a1 + 8);
        v48 = *(_BYTE *)(a1 + 25);
        v42 = *((_QWORD *)v27 + 1);
        v43 = *v27;
        v47 = *(__int64 **)a1;
        v28 = sub_1E0A0C0(*(_QWORD *)(*(_QWORD *)a1 + 32LL));
        v29 = sub_1F40B60(v44, v43, v42, v28, v48);
        *(_QWORD *)&v31 = sub_1D38BB0(
                            (__int64)v47,
                            v49,
                            (__int64)&v56,
                            v29,
                            v30,
                            0,
                            (__m128i)a3,
                            *(double *)a4.m128_u64,
                            a5,
                            0);
        v51.m128i_i64[0] = (__int64)v25;
        v51.m128i_i64[1] = v45 | v51.m128i_i64[1] & 0xFFFFFFFF00000000LL;
        v32 = sub_1D332F0(
                v47,
                124,
                (__int64)&v56,
                v40,
                v41,
                0,
                *(double *)a3.m128_u64,
                *(double *)a4.m128_u64,
                a5,
                (__int64)v25,
                v51.m128i_u64[1],
                v31);
        v33 = *(__int64 **)a1;
        v54 = v32;
        v55 = v34;
        *((_QWORD *)&v50 + 1) = (unsigned int)v34 | *((_QWORD *)&v50 + 1) & 0xFFFFFFFF00000000LL;
        v52 = sub_1D309E0(
                v33,
                145,
                (__int64)&v56,
                v10,
                v11,
                0,
                *(double *)a3.m128_u64,
                *(double *)a4.m128_u64,
                *(double *)a5.m128i_i64,
                __PAIR128__(*((unsigned __int64 *)&v50 + 1), (unsigned __int64)v32));
        *(_QWORD *)&v50 = v52;
        v53 = v35;
        v36 = *(__int64 **)a1;
        *((_QWORD *)&v50 + 1) = (unsigned int)v35 | *((_QWORD *)&v50 + 1) & 0xFFFFFFFF00000000LL;
        v51.m128i_i64[0] = sub_1D309E0(
                             v36,
                             145,
                             (__int64)&v56,
                             v10,
                             v11,
                             0,
                             *(double *)a3.m128_u64,
                             *(double *)a4.m128_u64,
                             *(double *)a5.m128i_i64,
                             *(_OWORD *)&v51);
        v58[1] = _mm_load_si128((const __m128i *)&v50);
        v51.m128i_i64[1] = v37 | v51.m128i_i64[1] & 0xFFFFFFFF00000000LL;
        v58[0] = _mm_load_si128(&v51);
        result = sub_1F994A0(a1, a2, (__int64 *)v58, 2, 1);
LABEL_7:
        if ( v56 )
        {
          *(_QWORD *)&v50 = v12;
          v51.m128i_i64[0] = result;
          sub_161E7C0((__int64)&v56, v56);
          return v51.m128i_i64[0];
        }
        return result;
      }
LABEL_6:
      result = 0;
      v12 = 0;
      goto LABEL_7;
    }
    if ( !(_BYTE)v16 )
      goto LABEL_6;
LABEL_15:
    v19 = v18;
    if ( !*(_QWORD *)(v15 + 8LL * v18 + 120) )
      goto LABEL_6;
    goto LABEL_16;
  }
  return result;
}
