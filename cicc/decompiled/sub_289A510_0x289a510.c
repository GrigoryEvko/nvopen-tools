// Function: sub_289A510
// Address: 0x289a510
//
__m128i *__fastcall sub_289A510(
        __m128i *a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int16 a6,
        __int64 a7,
        char a8,
        __int64 a9)
{
  __int64 v9; // r12
  unsigned int v10; // eax
  __int64 v11; // rax
  _BYTE *v12; // rax
  __int64 v13; // rax
  __int64 v14; // r14
  __int64 v15; // rbx
  char v16; // r12
  char v17; // r15
  __int64 v18; // r14
  _QWORD *v19; // rax
  __int64 v20; // r9
  __int64 v21; // r12
  unsigned int *v22; // r14
  __int64 v23; // rbx
  __int64 v24; // rdx
  unsigned int v25; // esi
  __int64 v26; // rdi
  unsigned int v27; // r12d
  __int64 v28; // r13
  __int64 v29; // rax
  double v30; // xmm0_8
  __int64 v31; // rax
  __int64 v32; // r8
  __int64 v33; // r9
  __m128d v34; // xmm0
  __int64 v35; // rdx
  __int64 v36; // rdi
  __int64 v37; // rdx
  double v38; // xmm1_8
  __m128d v39; // xmm1
  bool v40; // al
  _BYTE *v41; // rdi
  __int64 v43; // rdx
  __int64 v44; // r12
  _QWORD *v45; // rax
  unsigned __int64 v46; // rax
  unsigned __int64 v47; // rax
  unsigned __int64 v48; // rdx
  char v51; // [rsp+14h] [rbp-13Ch]
  char v52; // [rsp+16h] [rbp-13Ah]
  __int64 v55; // [rsp+28h] [rbp-128h]
  __int64 v57; // [rsp+40h] [rbp-110h]
  __int64 *v58; // [rsp+48h] [rbp-108h]
  double v59; // [rsp+48h] [rbp-108h]
  __int64 v60; // [rsp+60h] [rbp-F0h] BYREF
  __int64 v61; // [rsp+68h] [rbp-E8h]
  __int64 v62; // [rsp+70h] [rbp-E0h] BYREF
  __int64 v63; // [rsp+78h] [rbp-D8h]
  _BYTE v64[16]; // [rsp+80h] [rbp-D0h] BYREF
  __int16 v65; // [rsp+90h] [rbp-C0h]
  __m128i v66; // [rsp+100h] [rbp-50h] BYREF
  bool v67; // [rsp+110h] [rbp-40h]

  v52 = a6;
  v58 = *(__int64 **)a4;
  v55 = *(_QWORD *)a4 + 8LL * *(unsigned int *)(a4 + 8);
  if ( v55 != *(_QWORD *)a4 )
  {
    v57 = 0;
    v51 = HIBYTE(a6);
    while ( 1 )
    {
      v9 = *(_QWORD *)(a3 + 24);
      v10 = sub_BCB060(*(_QWORD *)(a7 + 8));
      v11 = sub_BCD140(*(_QWORD **)(a9 + 72), v10);
      v12 = (_BYTE *)sub_ACD640(v11, v57, 0);
      v13 = sub_289A440(a5, v12, (_BYTE *)a7, v9, (unsigned int **)a9);
      v14 = *(_QWORD *)(a3 + 24);
      v15 = v13;
      if ( v51 )
      {
        v16 = v52;
        if ( (_DWORD)v57 )
          goto LABEL_25;
      }
      else
      {
        v16 = sub_AE5020(*(_QWORD *)(a2 + 8), v14);
        if ( (_DWORD)v57 )
        {
LABEL_25:
          v60 = sub_9208B0(*(_QWORD *)(a2 + 8), v14);
          v61 = v43;
          v44 = 1LL << v16;
          if ( *(_BYTE *)a7 == 17 )
          {
            v45 = *(_QWORD **)(a7 + 24);
            if ( *(_DWORD *)(a7 + 32) > 0x40u )
              v45 = (_QWORD *)*v45;
            v62 = v60 * (_QWORD)v45;
            LOBYTE(v63) = v61;
            v46 = (unsigned int)v57 * ((unsigned __int64)sub_CA1930(&v62) >> 3);
          }
          else
          {
            v46 = (unsigned __int64)sub_CA1930(&v60) >> 3;
          }
          v47 = v44 | v46;
          v16 = -1;
          if ( (v47 & -(__int64)v47) != 0 )
          {
            _BitScanReverse64(&v48, v47 & -(__int64)v47);
            v16 = 63 - (v48 ^ 0x3F);
          }
        }
      }
      v17 = v16;
      v18 = *v58;
      v65 = 257;
      v19 = sub_BD2C40(80, unk_3F10A10);
      v21 = (__int64)v19;
      if ( v19 )
        sub_B4D3C0((__int64)v19, v18, v15, a8, v17, v20, 0, 0);
      (*(void (__fastcall **)(_QWORD, __int64, __int64 *, _QWORD, _QWORD))(**(_QWORD **)(a9 + 88) + 16LL))(
        *(_QWORD *)(a9 + 88),
        v21,
        &v62,
        *(_QWORD *)(a9 + 56),
        *(_QWORD *)(a9 + 64));
      v22 = *(unsigned int **)a9;
      v23 = *(_QWORD *)a9 + 16LL * *(unsigned int *)(a9 + 8);
      if ( *(_QWORD *)a9 != v23 )
      {
        do
        {
          v24 = *((_QWORD *)v22 + 1);
          v25 = *v22;
          v22 += 4;
          sub_B99FD0(v21, v25, v24);
        }
        while ( (unsigned int *)v23 != v22 );
      }
      ++v58;
      ++v57;
      if ( (__int64 *)v55 == v58 )
      {
        v58 = *(__int64 **)a4;
        break;
      }
    }
  }
  v66 = 0u;
  v63 = 0x1000000000LL;
  v62 = (__int64)v64;
  v67 = dword_5003CC8 == 0;
  v26 = *(_QWORD *)(*v58 + 8);
  v27 = *(_DWORD *)(v26 + 32);
  if ( (unsigned int)*(unsigned __int8 *)(v26 + 8) - 17 <= 1 )
    v26 = **(_QWORD **)(v26 + 16);
  v28 = *(_QWORD *)(a2 + 16);
  v29 = sub_BCAE30(v26) * v27;
  if ( v29 < 0 )
    v30 = (double)(int)(v29 & 1 | ((unsigned __int64)v29 >> 1)) + (double)(int)(v29 & 1 | ((unsigned __int64)v29 >> 1));
  else
    v30 = (double)(int)v29;
  v59 = v30;
  v31 = sub_DFB1B0(v28);
  v34 = (__m128d)*(unsigned __int64 *)&v30;
  v36 = v35;
  v60 = v31;
  v37 = v31;
  v61 = v36;
  if ( v31 < 0 )
  {
    v37 = v31 & 1 | ((unsigned __int64)v31 >> 1);
    v38 = (double)(int)v37 + (double)(int)v37;
  }
  else
  {
    v38 = (double)(int)v31;
  }
  v34.m128d_f64[0] = v30 / v38;
  if ( fabs(v59 / v38) < 4.503599627370496e15 )
  {
    v39.m128d_f64[0] = (double)(int)v34.m128d_f64[0];
    *(_QWORD *)&v34.m128d_f64[0] = COERCE_UNSIGNED_INT64(
                                     v39.m128d_f64[0]
                                   + COERCE_DOUBLE(*(_OWORD *)&_mm_cmpgt_sd(v34, v39) & 0x3FF0000000000000LL))
                                 | *(_QWORD *)&v34.m128d_f64[0] & 0x8000000000000000LL;
  }
  v66.m128i_i32[0] += *(_DWORD *)(a4 + 8) * (int)v34.m128d_f64[0];
  a1->m128i_i64[0] = (__int64)a1[1].m128i_i64;
  a1->m128i_i64[1] = 0x1000000000LL;
  if ( (_DWORD)v63 )
    sub_2894AD0((__int64)a1, (__int64)&v62, v37, (__int64)a1, v32, v33);
  v40 = v67;
  v41 = (_BYTE *)v62;
  a1[9] = _mm_loadu_si128(&v66);
  a1[10].m128i_i8[0] = v40;
  if ( v41 != v64 )
    _libc_free((unsigned __int64)v41);
  return a1;
}
