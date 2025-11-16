// Function: sub_1BB8E30
// Address: 0x1bb8e30
//
__int64 __fastcall sub_1BB8E30(
        __int64 a1,
        __m128 a2,
        __m128i a3,
        __m128i a4,
        double a5,
        double a6,
        double a7,
        double a8,
        __m128 a9,
        __int64 a10,
        __int64 a11,
        __int64 a12,
        __int64 *a13,
        __int64 a14,
        __int64 a15,
        __int64 a16,
        __int64 a17,
        __int64 a18,
        __int64 a19,
        __int64 a20,
        __int64 a21)
{
  double v22; // xmm4_8
  double v23; // xmm5_8
  __int128 v24; // rdi
  unsigned int v25; // r12d
  _QWORD *v26; // r13
  _QWORD *i; // r14
  int v28; // eax
  _QWORD *v29; // r14
  _QWORD *v30; // r13
  int j; // eax
  __int64 v32; // rsi
  __int64 v33; // r15
  __int64 v34; // rcx
  int v35; // eax
  __int64 v36; // rdx
  __int64 v37; // rcx
  __int64 v38; // r8
  double v39; // xmm4_8
  double v40; // xmm5_8
  int v41; // r14d
  _BYTE *v43; // [rsp+10h] [rbp-80h] BYREF
  __int64 v44; // [rsp+18h] [rbp-78h]
  _BYTE v45[112]; // [rsp+20h] [rbp-70h] BYREF

  *(_QWORD *)(a1 + 8) = a11;
  *(_QWORD *)(a1 + 16) = a12;
  *(_QWORD *)(a1 + 40) = a15;
  *(_QWORD *)(a1 + 24) = a13;
  *(_QWORD *)(a1 + 48) = a16;
  *(_QWORD *)(a1 + 32) = a14;
  *(_QWORD *)(a1 + 64) = a18;
  *(_QWORD *)(a1 + 72) = a19;
  *(_QWORD *)(a1 + 80) = a20;
  *(_QWORD *)(a1 + 56) = a17;
  *(_QWORD *)(a1 + 88) = a21;
  if ( (unsigned int)sub_14A3140(a13, 1u) || (v25 = 0, (unsigned int)sub_14A3320(*(_QWORD *)(a1 + 24)) > 1) )
  {
    *((_QWORD *)&v24 + 1) = *(_QWORD *)(a1 + 16);
    v25 = 0;
    v26 = *(_QWORD **)(*((_QWORD *)&v24 + 1) + 32LL);
    for ( i = *(_QWORD **)(*((_QWORD *)&v24 + 1) + 40LL); i != v26; v25 |= v28 )
    {
      *(_QWORD *)&v24 = *v26++;
      v28 = sub_1AFB400(
              v24,
              *(_QWORD *)(a1 + 32),
              *((__int64 *)&v24 + 1),
              *(_QWORD *)(a1 + 8),
              *(_QWORD *)(a1 + 72),
              0,
              a2,
              *(double *)a3.m128i_i64,
              *(double *)a4.m128i_i64,
              a5,
              v22,
              v23,
              a8,
              a9);
      *((_QWORD *)&v24 + 1) = *(_QWORD *)(a1 + 16);
    }
    v29 = *(_QWORD **)(*((_QWORD *)&v24 + 1) + 40LL);
    v43 = v45;
    v44 = 0x800000000LL;
    if ( v29 != *(_QWORD **)(*((_QWORD *)&v24 + 1) + 32LL) )
    {
      v30 = *(_QWORD **)(*((_QWORD *)&v24 + 1) + 32LL);
      while ( 1 )
      {
        *(_QWORD *)&v24 = *v30++;
        sub_1B965E0(v24, *(_QWORD **)(a1 + 88), (__int64)&v43);
        if ( v29 == v30 )
          break;
        *((_QWORD *)&v24 + 1) = *(_QWORD *)(a1 + 16);
      }
      for ( j = v44; (_DWORD)v44; v25 |= v41 )
      {
        v32 = *(_QWORD *)(a1 + 32);
        v33 = *(_QWORD *)&v43[8 * j - 8];
        v34 = *(_QWORD *)(a1 + 8);
        LODWORD(v44) = j - 1;
        LOBYTE(v35) = sub_1AE5120(v33, v32, *(_QWORD *)(a1 + 16), v34);
        v41 = sub_1BB6740((unsigned __int8 *)a1, v33, (__m128i)a2, a3, a4, a5, v39, v40, a8, a9, v36, v37, v38) | v35;
        j = v44;
      }
      if ( v43 != v45 )
        _libc_free((unsigned __int64)v43);
    }
  }
  return v25;
}
