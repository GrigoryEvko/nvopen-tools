// Function: sub_33AA620
// Address: 0x33aa620
//
__int64 __fastcall sub_33AA620(__int64 a1, __int64 a2, unsigned __int8 a3)
{
  __int64 v5; // r12
  __int64 v6; // rsi
  __int64 v7; // r13
  __int64 v8; // rax
  int v9; // eax
  int v10; // eax
  __int64 v11; // rax
  __int64 v12; // rdx
  __int64 v13; // rdx
  __int64 v14; // rcx
  __int64 v15; // r8
  __int64 v16; // r9
  __int64 v17; // rdx
  __int64 v18; // r8
  __int64 v19; // r9
  __int64 v20; // rax
  __int64 v21; // rsi
  unsigned int v22; // r12d
  __int64 v24; // r13
  __int64 v25; // r12
  __m128i v26; // xmm0
  _QWORD *v27; // rax
  __int64 v28; // r14
  __int64 v29; // r13
  __int64 v30; // rbx
  __int64 v31; // [rsp+0h] [rbp-110h]
  __int64 v32; // [rsp+8h] [rbp-108h]
  __int64 v33; // [rsp+10h] [rbp-100h]
  __int64 v34; // [rsp+18h] [rbp-F8h]
  __int64 v35; // [rsp+20h] [rbp-F0h]
  __int64 v36; // [rsp+28h] [rbp-E8h]
  __m128i v37; // [rsp+30h] [rbp-E0h] BYREF
  __int64 v38; // [rsp+48h] [rbp-C8h]
  __int64 v39; // [rsp+50h] [rbp-C0h]
  __int64 v40; // [rsp+58h] [rbp-B8h]
  __m128i v41; // [rsp+60h] [rbp-B0h]
  __int64 v42; // [rsp+70h] [rbp-A0h] BYREF
  int v43; // [rsp+78h] [rbp-98h]
  unsigned __int64 v44; // [rsp+80h] [rbp-90h]
  __int64 v45; // [rsp+88h] [rbp-88h]
  __int64 v46; // [rsp+90h] [rbp-80h]
  unsigned __int64 v47; // [rsp+A0h] [rbp-70h] BYREF
  __int64 v48; // [rsp+A8h] [rbp-68h]
  __int64 v49; // [rsp+B0h] [rbp-60h]
  __m128i v50; // [rsp+C0h] [rbp-50h] BYREF
  __int64 v51; // [rsp+D0h] [rbp-40h]
  __int64 v52; // [rsp+D8h] [rbp-38h]

  v5 = *(_QWORD *)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF));
  v6 = *(_QWORD *)(a2 + 32 * (1LL - (*(_DWORD *)(a2 + 4) & 0x7FFFFFF)));
  v7 = *(_QWORD *)(*(_QWORD *)(a1 + 864) + 8LL);
  v37.m128i_i64[0] = *(_QWORD *)(*(_QWORD *)v7 + 80LL);
  if ( v6 )
  {
    v48 = 0;
    BYTE4(v49) = 0;
    v47 = v6 & 0xFFFFFFFFFFFFFFFBLL;
    v8 = *(_QWORD *)(v6 + 8);
    if ( (unsigned int)*(unsigned __int8 *)(v8 + 8) - 17 <= 1 )
      v8 = **(_QWORD **)(v8 + 16);
    v9 = *(_DWORD *)(v8 + 8) >> 8;
  }
  else
  {
    v47 = 0;
    v9 = 0;
    v48 = 0;
    BYTE4(v49) = 0;
  }
  LODWORD(v49) = v9;
  BYTE4(v46) = 0;
  v44 = v5 & 0xFFFFFFFFFFFFFFFBLL;
  v10 = 0;
  v45 = 0;
  if ( v5 )
  {
    v11 = *(_QWORD *)(v5 + 8);
    if ( (unsigned int)*(unsigned __int8 *)(v11 + 8) - 17 <= 1 )
      v11 = **(_QWORD **)(v11 + 16);
    v10 = *(_DWORD *)(v11 + 8) >> 8;
  }
  LODWORD(v46) = v10;
  v33 = sub_338B750(a1, v6);
  v34 = v12;
  v35 = sub_338B750(a1, v5);
  v36 = v13;
  v42 = 0;
  v18 = sub_33738B0(a1, v5, v13, v14, v15, v16);
  v19 = v17;
  v20 = *(_QWORD *)a1;
  v43 = *(_DWORD *)(a1 + 848);
  if ( v20 )
  {
    if ( &v42 != (__int64 *)(v20 + 48) )
    {
      v21 = *(_QWORD *)(v20 + 48);
      v42 = v21;
      if ( v21 )
      {
        v31 = v18;
        v32 = v17;
        sub_B96E90((__int64)&v42, v21, 1);
        v18 = v31;
        v19 = v32;
      }
    }
  }
  if ( (_OWORD *(__fastcall *)(_OWORD *))v37.m128i_i64[0] == sub_3364F90 )
  {
    v22 = 0;
    if ( v42 )
      sub_B91220((__int64)&v42, v42);
  }
  else
  {
    ((void (__fastcall *)(__m128i *, __int64, _QWORD, __int64 *, __int64, __int64, __int64, __int64, __int64, __int64, unsigned __int64, __int64, __int64, unsigned __int64, __int64, __int64, _QWORD))v37.m128i_i64[0])(
      &v50,
      v7,
      *(_QWORD *)(a1 + 864),
      &v42,
      v18,
      v19,
      v35,
      v36,
      v33,
      v34,
      v44,
      v45,
      v46,
      v47,
      v48,
      v49,
      a3);
    v24 = v50.m128i_i64[0];
    v25 = v51;
    if ( v42 )
      sub_B91220((__int64)&v42, v42);
    if ( v24 )
    {
      v26 = _mm_loadu_si128(&v50);
      v47 = a2;
      v37 = v26;
      v27 = sub_337DC20(a1 + 8, (__int64 *)&v47);
      v28 = v52;
      v41 = _mm_load_si128(&v37);
      *v27 = v41.m128i_i64[0];
      *((_DWORD *)v27 + 2) = v41.m128i_i32[2];
      v29 = *(_QWORD *)(a1 + 864);
      v30 = v51;
      if ( v25 )
      {
        nullsub_1875(v25, v29, 0);
        v40 = v28;
        v39 = v30;
        *(_QWORD *)(v29 + 384) = v30;
        *(_DWORD *)(v29 + 392) = v40;
        sub_33E2B60(v29, 0);
      }
      else
      {
        v38 = v28;
        *(_QWORD *)(v29 + 384) = 0;
        *(_DWORD *)(v29 + 392) = v38;
      }
      return 1;
    }
    else
    {
      return 0;
    }
  }
  return v22;
}
