// Function: sub_289A9E0
// Address: 0x289a9e0
//
__int64 __fastcall sub_289A9E0(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int16 a5,
        __int64 a6,
        char a7,
        int a8,
        int a9,
        char a10,
        __int64 a11)
{
  unsigned int v11; // r14d
  __int64 v12; // rax
  __int64 v13; // r14
  unsigned int v14; // eax
  __int64 v15; // rax
  _BYTE *v16; // rax
  __int64 v17; // rax
  char v18; // bl
  __int64 v19; // r12
  char v20; // r15
  _QWORD *v21; // rax
  __int64 v22; // rbx
  __int64 v23; // r8
  __int64 v24; // r9
  unsigned int *v25; // r15
  __int64 v26; // r12
  __int64 v27; // rdx
  unsigned int v28; // esi
  __int64 v29; // rax
  unsigned __int64 v30; // rdx
  __int64 v31; // rdi
  unsigned int v32; // ebx
  __int64 v33; // r12
  __int64 v34; // rax
  __int64 v35; // rdx
  double v36; // xmm0_8
  __int64 v37; // rax
  __int64 v38; // r8
  __int64 v39; // r9
  __m128d v40; // xmm0
  __int64 v41; // rdx
  double v42; // xmm1_8
  __m128d v43; // xmm1
  __int64 v44; // rdx
  __int64 v45; // rcx
  __m128i v46; // xmm5
  _BYTE *v47; // rdi
  __int64 v49; // rdx
  __int64 v50; // rbx
  _QWORD *v51; // rax
  unsigned __int64 v52; // rax
  unsigned __int64 v53; // rax
  unsigned __int64 v54; // rdx
  __int64 v56; // [rsp+20h] [rbp-190h]
  __int64 v58; // [rsp+30h] [rbp-180h]
  __int64 *v62; // [rsp+50h] [rbp-160h]
  double v63; // [rsp+58h] [rbp-158h]
  __int64 v64; // [rsp+60h] [rbp-150h] BYREF
  __int64 v65; // [rsp+68h] [rbp-148h]
  _QWORD v66[4]; // [rsp+70h] [rbp-140h] BYREF
  char v67; // [rsp+90h] [rbp-120h]
  char v68; // [rsp+91h] [rbp-11Fh]
  __int64 v69; // [rsp+A0h] [rbp-110h] BYREF
  __int64 v70; // [rsp+A8h] [rbp-108h]
  __int16 v71; // [rsp+C0h] [rbp-F0h]
  _BYTE *v72; // [rsp+D0h] [rbp-E0h] BYREF
  __int64 v73; // [rsp+D8h] [rbp-D8h]
  _BYTE v74[128]; // [rsp+E0h] [rbp-D0h] BYREF
  __m128i v75; // [rsp+160h] [rbp-50h] BYREF
  bool v76; // [rsp+170h] [rbp-40h]

  v11 = a8;
  v62 = *(__int64 **)(a3 + 24);
  if ( a10 )
  {
    v11 = a9;
    v75 = 0u;
    v58 = sub_BCDA70(v62, a8);
  }
  else
  {
    v75 = 0u;
    v58 = sub_BCDA70(v62, a9);
  }
  v72 = v74;
  v73 = 0x1000000000LL;
  v76 = dword_5003CC8 == 0;
  if ( v11 )
  {
    v12 = v11;
    v13 = 0;
    v56 = v12;
    do
    {
      v14 = sub_BCB060(*(_QWORD *)(a6 + 8));
      v15 = sub_BCD140(*(_QWORD **)(a11 + 72), v14);
      v16 = (_BYTE *)sub_ACD640(v15, v13, 0);
      v17 = sub_289A440(a4, v16, (_BYTE *)a6, (__int64)v62, (unsigned int **)a11);
      v18 = a5;
      v68 = 1;
      v19 = v17;
      v67 = 3;
      v66[0] = "col.load";
      if ( HIBYTE(a5) )
      {
        if ( v13 )
          goto LABEL_28;
      }
      else
      {
        v18 = sub_AE5020(*(_QWORD *)(a2 + 8), (__int64)v62);
        if ( v13 )
        {
LABEL_28:
          v64 = sub_9208B0(*(_QWORD *)(a2 + 8), (__int64)v62);
          v65 = v49;
          v50 = 1LL << v18;
          if ( *(_BYTE *)a6 == 17 )
          {
            v51 = *(_QWORD **)(a6 + 24);
            if ( *(_DWORD *)(a6 + 32) > 0x40u )
              v51 = (_QWORD *)*v51;
            v69 = v64 * (_QWORD)v51;
            LOBYTE(v70) = v65;
            v52 = v13 * ((unsigned __int64)sub_CA1930(&v69) >> 3);
          }
          else
          {
            v52 = (unsigned __int64)sub_CA1930(&v64) >> 3;
          }
          v53 = v50 | v52;
          v18 = -1;
          if ( (v53 & -(__int64)v53) != 0 )
          {
            _BitScanReverse64(&v54, v53 & -(__int64)v53);
            v18 = 63 - (v54 ^ 0x3F);
          }
        }
      }
      v71 = 257;
      v20 = v18;
      v21 = sub_BD2C40(80, unk_3F10A14);
      v22 = (__int64)v21;
      if ( v21 )
        sub_B4D190((__int64)v21, v58, v19, (__int64)&v69, a7, v20, 0, 0);
      (*(void (__fastcall **)(_QWORD, __int64, _QWORD *, _QWORD, _QWORD))(**(_QWORD **)(a11 + 88) + 16LL))(
        *(_QWORD *)(a11 + 88),
        v22,
        v66,
        *(_QWORD *)(a11 + 56),
        *(_QWORD *)(a11 + 64));
      v25 = *(unsigned int **)a11;
      v26 = *(_QWORD *)a11 + 16LL * *(unsigned int *)(a11 + 8);
      if ( *(_QWORD *)a11 != v26 )
      {
        do
        {
          v27 = *((_QWORD *)v25 + 1);
          v28 = *v25;
          v25 += 4;
          sub_B99FD0(v22, v28, v27);
        }
        while ( (unsigned int *)v26 != v25 );
      }
      v29 = (unsigned int)v73;
      v30 = (unsigned int)v73 + 1LL;
      if ( v30 > HIDWORD(v73) )
      {
        sub_C8D5F0((__int64)&v72, v74, v30, 8u, v23, v24);
        v29 = (unsigned int)v73;
      }
      ++v13;
      *(_QWORD *)&v72[8 * v29] = v22;
      LODWORD(v73) = v73 + 1;
    }
    while ( v56 != v13 );
  }
  v31 = *(_QWORD *)(*(_QWORD *)v72 + 8LL);
  v32 = *(_DWORD *)(v31 + 32);
  if ( (unsigned int)*(unsigned __int8 *)(v31 + 8) - 17 <= 1 )
    v31 = **(_QWORD **)(v31 + 16);
  v33 = *(_QWORD *)(a2 + 16);
  v66[0] = sub_BCAE30(v31);
  v34 = v66[0] * v32;
  v66[1] = v35;
  if ( v34 < 0 )
    v36 = (double)(int)((LOBYTE(v66[0]) * (_BYTE)v32) & 1 | ((v66[0] * (unsigned __int64)v32) >> 1))
        + (double)(int)((LOBYTE(v66[0]) * (_BYTE)v32) & 1 | ((v66[0] * (unsigned __int64)v32) >> 1));
  else
    v36 = (double)(int)v34;
  v63 = v36;
  v37 = sub_DFB1B0(v33);
  v40 = (__m128d)*(unsigned __int64 *)&v36;
  v69 = v37;
  v70 = v41;
  if ( v37 < 0 )
    v42 = (double)(int)(v37 & 1 | ((unsigned __int64)v37 >> 1)) + (double)(int)(v37 & 1 | ((unsigned __int64)v37 >> 1));
  else
    v42 = (double)(int)v37;
  v40.m128d_f64[0] = v36 / v42;
  if ( fabs(v63 / v42) < 4.503599627370496e15 )
  {
    v43.m128d_f64[0] = (double)(int)v40.m128d_f64[0];
    *(_QWORD *)&v40.m128d_f64[0] = COERCE_UNSIGNED_INT64(
                                     v43.m128d_f64[0]
                                   + COERCE_DOUBLE(*(_OWORD *)&_mm_cmpgt_sd(v40, v43) & 0x3FF0000000000000LL))
                                 | *(_QWORD *)&v40.m128d_f64[0] & 0x8000000000000000LL;
  }
  v44 = (unsigned int)v73;
  v45 = v76;
  v75.m128i_i32[1] += v73 * (int)v40.m128d_f64[0];
  *(_QWORD *)a1 = a1 + 16;
  *(_QWORD *)(a1 + 8) = 0x1000000000LL;
  if ( (_DWORD)v44 )
  {
    sub_2894AD0(a1, (__int64)&v72, v44, v45, v38, v39);
    LOBYTE(v45) = v76;
  }
  v46 = _mm_loadu_si128(&v75);
  v47 = v72;
  *(_BYTE *)(a1 + 160) = v45;
  *(__m128i *)(a1 + 144) = v46;
  if ( v47 != v74 )
    _libc_free((unsigned __int64)v47);
  return a1;
}
