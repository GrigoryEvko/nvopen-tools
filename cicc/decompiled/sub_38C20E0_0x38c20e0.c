// Function: sub_38C20E0
// Address: 0x38c20e0
//
__int64 __fastcall sub_38C20E0(
        __int64 a1,
        _BYTE *a2,
        __int64 a3,
        int a4,
        unsigned __int8 a5,
        unsigned int a6,
        __int64 a7,
        __int64 a8,
        unsigned int a9,
        _BYTE *a10)
{
  _BYTE *v10; // r10
  __int64 v12; // r14
  __int64 v13; // rcx
  _QWORD *v14; // rsi
  _QWORD *v15; // rdx
  __m128i v16; // xmm0
  char v17; // dl
  _QWORD *v18; // r14
  __int64 v19; // r13
  _BYTE *v21; // rax
  __int64 *v22; // rcx
  __int64 v23; // [rsp+0h] [rbp-D0h]
  __int64 v24; // [rsp+0h] [rbp-D0h]
  _BYTE *v27; // [rsp+10h] [rbp-C0h]
  __int64 v28; // [rsp+18h] [rbp-B8h]
  char v29; // [rsp+18h] [rbp-B8h]
  __int64 v30; // [rsp+18h] [rbp-B8h]
  _QWORD *v31; // [rsp+20h] [rbp-B0h] BYREF
  __int64 v32; // [rsp+28h] [rbp-A8h]
  _BYTE v33[16]; // [rsp+30h] [rbp-A0h] BYREF
  __m128i v34; // [rsp+40h] [rbp-90h] BYREF
  unsigned __int64 v35; // [rsp+50h] [rbp-80h]
  __m128i v36; // [rsp+60h] [rbp-70h] BYREF
  _WORD v37[8]; // [rsp+70h] [rbp-60h] BYREF
  __m128i v38; // [rsp+80h] [rbp-50h]
  unsigned __int64 v39; // [rsp+90h] [rbp-40h]
  __int64 v40; // [rsp+98h] [rbp-38h]

  v10 = a2;
  v12 = a8;
  if ( a8 )
  {
    v24 = a3;
    v37[0] = 261;
    v36.m128i_i64[0] = (__int64)&a7;
    v21 = (_BYTE *)sub_38BF510(a1, (__int64)&v36);
    v10 = a2;
    a3 = v24;
    v27 = v21;
    if ( (*v21 & 4) != 0 )
    {
      v22 = (__int64 *)*((_QWORD *)v21 - 1);
      v12 = *v22;
      v13 = (__int64)(v22 + 2);
    }
    else
    {
      v12 = 0;
      v13 = 0;
    }
    a7 = v13;
    a8 = v12;
  }
  else
  {
    v27 = 0;
    v13 = a7;
  }
  if ( v10 )
  {
    v28 = v13;
    v31 = v33;
    sub_38BB9D0((__int64 *)&v31, v10, (__int64)&v10[a3]);
    v14 = v31;
    v13 = v28;
    v15 = (_QWORD *)((char *)v31 + v32);
  }
  else
  {
    v15 = v33;
    v33[0] = 0;
    v31 = v33;
    v14 = v33;
    v32 = 0;
  }
  v35 = __PAIR64__(a9, a6);
  v34.m128i_i64[0] = v13;
  v36.m128i_i64[0] = (__int64)v37;
  v34.m128i_i64[1] = v12;
  sub_38BBC60(v36.m128i_i64, v14, (__int64)v15);
  v16 = _mm_loadu_si128(&v34);
  v40 = 0;
  v39 = v35;
  v38 = v16;
  v18 = (_QWORD *)sub_38C1DA0((_QWORD *)(a1 + 1248), &v36);
  if ( (_WORD *)v36.m128i_i64[0] != v37 )
  {
    v29 = v17;
    j_j___libc_free_0(v36.m128i_u64[0]);
    v17 = v29;
  }
  if ( v17 )
  {
    if ( a10 )
    {
      v37[0] = 257;
      if ( *a10 )
      {
        v36.m128i_i64[0] = (__int64)a10;
        LOBYTE(v37[0]) = 3;
      }
      a10 = (_BYTE *)sub_38BF8E0(a1, (__int64)&v36, 0, 1);
    }
    v23 = v18[4];
    v30 = v18[5];
    v19 = sub_145CBF0((__int64 *)(a1 + 152), 192, 8);
    sub_38D76F0(v19, 0, a5, a10);
    *(_DWORD *)(v19 + 172) = -1;
    *(_DWORD *)(v19 + 184) = a6;
    *(_QWORD *)(v19 + 152) = v23;
    *(_QWORD *)v19 = &unk_4A3E570;
    *(_QWORD *)(v19 + 160) = v30;
    *(_DWORD *)(v19 + 168) = a4;
    *(_QWORD *)(v19 + 176) = v27;
    v18[11] = v19;
  }
  else
  {
    v19 = v18[11];
  }
  if ( v31 != (_QWORD *)v33 )
    j_j___libc_free_0((unsigned __int64)v31);
  return v19;
}
