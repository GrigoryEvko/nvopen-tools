// Function: sub_1AE5F80
// Address: 0x1ae5f80
//
__int64 __fastcall sub_1AE5F80(__int64 a1, unsigned int a2, unsigned __int16 a3, float a4, __m128i a5, double a6)
{
  unsigned int v6; // r14d
  _QWORD *v7; // rax
  unsigned __int8 *v8; // rsi
  __int64 v9; // rax
  __int16 *v10; // rbx
  __int64 v11; // r14
  __int64 v12; // r14
  __int64 v13; // rbx
  __int64 v14; // rax
  __int64 v15; // rdx
  __int64 v16; // rbx
  __int64 v18; // rax
  __int64 *v19; // r13
  __int64 v20; // rax
  __int64 v21; // rcx
  __int64 v22; // rsi
  unsigned __int8 *v23; // rsi
  __int64 v24; // rdx
  __int64 v25; // rsi
  __int64 v26; // rbx
  __int64 v27; // r12
  __int64 v28; // rdx
  __int64 v29; // rsi
  __int64 v30; // r14
  __int64 v31; // rbx
  __int16 *v32; // [rsp+0h] [rbp-108h]
  __int64 v33; // [rsp+8h] [rbp-100h]
  void *v36; // [rsp+20h] [rbp-E8h]
  __int64 v37; // [rsp+28h] [rbp-E0h]
  __int64 ***v38; // [rsp+30h] [rbp-D8h]
  unsigned __int8 *v39; // [rsp+40h] [rbp-C8h] BYREF
  __int64 v40[2]; // [rsp+48h] [rbp-C0h] BYREF
  __int16 v41; // [rsp+58h] [rbp-B0h]
  unsigned __int8 *v42; // [rsp+68h] [rbp-A0h] BYREF
  void *v43; // [rsp+70h] [rbp-98h] BYREF
  __int64 v44; // [rsp+78h] [rbp-90h]
  unsigned __int8 *v45; // [rsp+88h] [rbp-80h] BYREF
  __int64 v46; // [rsp+90h] [rbp-78h]
  __int64 *v47; // [rsp+98h] [rbp-70h]
  _QWORD *v48; // [rsp+A0h] [rbp-68h]
  __int64 v49; // [rsp+A8h] [rbp-60h]
  int v50; // [rsp+B0h] [rbp-58h]
  __int64 v51; // [rsp+B8h] [rbp-50h]
  __int64 v52; // [rsp+C0h] [rbp-48h]

  v6 = _mm_cvtsi128_si32(a5);
  v7 = (_QWORD *)sub_16498A0(a1);
  v8 = *(unsigned __int8 **)(a1 + 48);
  v45 = 0;
  v48 = v7;
  v9 = *(_QWORD *)(a1 + 40);
  v49 = 0;
  v46 = v9;
  v50 = 0;
  v51 = 0;
  v52 = 0;
  v47 = (__int64 *)(a1 + 24);
  v42 = v8;
  if ( v8 )
  {
    sub_1623A60((__int64)&v42, (__int64)v8, 2);
    if ( v45 )
      sub_161E7C0((__int64)&v45, (__int64)v45);
    v45 = v42;
    if ( v42 )
      sub_1623210((__int64)&v42, v42, (__int64)&v45);
  }
  v38 = *(__int64 ****)(a1 - 24LL * (*(_DWORD *)(a1 + 20) & 0xFFFFFFF));
  v10 = (__int16 *)sub_1698270();
  sub_169D3B0((__int64)v40, _mm_cvtsi32_si128(v6));
  sub_169E320(&v43, v40, v10);
  sub_1698460((__int64)v40);
  v11 = sub_159CCF0(v48, (__int64)&v42);
  v36 = sub_16982C0();
  if ( v43 == v36 )
  {
    v28 = v44;
    if ( v44 )
    {
      v29 = 32LL * *(_QWORD *)(v44 - 8);
      if ( v44 != v44 + v29 )
      {
        v33 = v11;
        v30 = v44;
        v32 = v10;
        v31 = v44 + v29;
        do
        {
          v31 -= 32;
          sub_127D120((_QWORD *)(v31 + 8));
        }
        while ( v30 != v31 );
        v28 = v30;
        v10 = v32;
        v11 = v33;
      }
      j_j_j___libc_free_0_0(v28 - 8);
    }
  }
  else
  {
    sub_1698460((__int64)&v43);
  }
  if ( *((_BYTE *)*v38 + 8) != 2 )
    v11 = sub_15A3E10(v11, *v38, 0);
  LOWORD(v44) = 257;
  v12 = sub_1289B20((__int64 *)&v45, a3, v38, v11, (__int64)&v42, 0);
  sub_169D3B0((__int64)v40, (__m128i)LODWORD(a4));
  sub_169E320(&v43, v40, v10);
  sub_1698460((__int64)v40);
  v13 = sub_159CCF0(v48, (__int64)&v42);
  if ( v36 == v43 )
  {
    v24 = v44;
    if ( v44 )
    {
      v25 = 32LL * *(_QWORD *)(v44 - 8);
      if ( v44 != v44 + v25 )
      {
        v37 = v13;
        v26 = v44;
        v27 = v44 + v25;
        do
        {
          v27 -= 32;
          sub_127D120((_QWORD *)(v27 + 8));
        }
        while ( v26 != v27 );
        v24 = v26;
        v13 = v37;
      }
      j_j_j___libc_free_0_0(v24 - 8);
    }
  }
  else
  {
    sub_1698460((__int64)&v43);
  }
  if ( *((_BYTE *)*v38 + 8) != 2 )
    v13 = sub_15A3E10(v13, *v38, 0);
  LOWORD(v44) = 257;
  v14 = sub_1289B20((__int64 *)&v45, a2, v38, v13, (__int64)&v42, 0);
  v41 = 257;
  v16 = v14;
  if ( *(_BYTE *)(v12 + 16) <= 0x10u )
  {
    if ( sub_1593BB0(v12, a2, v15, 257) )
      goto LABEL_18;
    if ( *(_BYTE *)(v16 + 16) <= 0x10u )
    {
      v16 = sub_15A2D10((__int64 *)v16, v12, COERCE_DOUBLE((unsigned __int64)LODWORD(a4)), *(double *)a5.m128i_i64, a6);
LABEL_18:
      if ( v45 )
        sub_161E7C0((__int64)&v45, (__int64)v45);
      return v16;
    }
  }
  LOWORD(v44) = 257;
  v18 = sub_15FB440(27, (__int64 *)v16, v12, (__int64)&v42, 0);
  v16 = v18;
  if ( v46 )
  {
    v19 = v47;
    sub_157E9D0(v46 + 40, v18);
    v20 = *(_QWORD *)(v16 + 24);
    v21 = *v19;
    *(_QWORD *)(v16 + 32) = v19;
    v21 &= 0xFFFFFFFFFFFFFFF8LL;
    *(_QWORD *)(v16 + 24) = v21 | v20 & 7;
    *(_QWORD *)(v21 + 8) = v16 + 24;
    *v19 = *v19 & 7 | (v16 + 24);
  }
  sub_164B780(v16, v40);
  if ( v45 )
  {
    v39 = v45;
    sub_1623A60((__int64)&v39, (__int64)v45, 2);
    v22 = *(_QWORD *)(v16 + 48);
    if ( v22 )
      sub_161E7C0(v16 + 48, v22);
    v23 = v39;
    *(_QWORD *)(v16 + 48) = v39;
    if ( v23 )
      sub_1623210((__int64)&v39, v23, v16 + 48);
    goto LABEL_18;
  }
  return v16;
}
