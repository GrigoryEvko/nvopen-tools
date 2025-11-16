// Function: sub_18B6C20
// Address: 0x18b6c20
//
void __fastcall sub_18B6C20(
        __int64 a1,
        _BYTE *a2,
        __int64 a3,
        _BYTE *a4,
        __int64 a5,
        __int64 a6,
        __int64 (__fastcall *a7)(__int64, __int64),
        __int64 a8)
{
  __int64 v10; // r8
  unsigned __int64 v11; // rax
  __int64 v12; // r9
  __int64 v13; // rsi
  __int64 v14; // r10
  char *v15; // rbx
  char *v16; // r12
  char *v17; // rdi
  __int64 v18; // [rsp+0h] [rbp-380h]
  __int64 v21; // [rsp+18h] [rbp-368h]
  _QWORD *v22; // [rsp+18h] [rbp-368h]
  __int64 v23; // [rsp+28h] [rbp-358h] BYREF
  __m128i v24[2]; // [rsp+30h] [rbp-350h] BYREF
  _BYTE *v25[2]; // [rsp+50h] [rbp-330h] BYREF
  __int64 v26; // [rsp+60h] [rbp-320h] BYREF
  __int64 *v27; // [rsp+70h] [rbp-310h]
  __int64 v28; // [rsp+78h] [rbp-308h]
  __int64 v29; // [rsp+80h] [rbp-300h] BYREF
  __m128i v30; // [rsp+90h] [rbp-2F0h] BYREF
  __int64 v31; // [rsp+A0h] [rbp-2E0h]
  _BYTE *v32[2]; // [rsp+B0h] [rbp-2D0h] BYREF
  __int64 v33; // [rsp+C0h] [rbp-2C0h] BYREF
  __int64 *v34; // [rsp+D0h] [rbp-2B0h]
  __int64 v35; // [rsp+D8h] [rbp-2A8h]
  __int64 v36; // [rsp+E0h] [rbp-2A0h] BYREF
  __m128i v37; // [rsp+F0h] [rbp-290h] BYREF
  __int64 v38; // [rsp+100h] [rbp-280h]
  __m128i v39; // [rsp+110h] [rbp-270h] BYREF
  _QWORD v40[2]; // [rsp+120h] [rbp-260h] BYREF
  __int64 v41[2]; // [rsp+130h] [rbp-250h] BYREF
  _QWORD v42[2]; // [rsp+140h] [rbp-240h] BYREF
  __m128i v43; // [rsp+150h] [rbp-230h]
  __int64 v44; // [rsp+160h] [rbp-220h]
  _QWORD v45[11]; // [rsp+170h] [rbp-210h] BYREF
  char *v46; // [rsp+1C8h] [rbp-1B8h]
  unsigned int v47; // [rsp+1D0h] [rbp-1B0h]
  char v48; // [rsp+1D8h] [rbp-1A8h] BYREF

  v10 = a8;
  v11 = *(_QWORD *)(a1 + 8) & 0xFFFFFFFFFFFFFFF8LL;
  v12 = *(_QWORD *)(v11 + 40);
  v13 = *(_QWORD *)(v11 + 48);
  v14 = *(_QWORD *)(v12 + 56);
  v23 = v13;
  if ( v13 )
  {
    v21 = v14;
    sub_1623A60((__int64)&v23, v13, 2);
    v10 = a8;
    v14 = v21;
    v12 = *(_QWORD *)((*(_QWORD *)(a1 + 8) & 0xFFFFFFFFFFFFFFF8LL) + 40);
  }
  v18 = v12;
  v22 = (_QWORD *)a7(v10, v14);
  sub_15C9090((__int64)v24, &v23);
  sub_15CA330((__int64)v45, (__int64)"wholeprogramdevirt", (__int64)a2, a3, v24, v18);
  sub_15C9800((__int64)v25, "Optimization", 12, a2, a3);
  v39.m128i_i64[0] = (__int64)v40;
  sub_18B4BA0(v39.m128i_i64, v25[0], (__int64)&v25[0][(unsigned __int64)v25[1]]);
  v41[0] = (__int64)v42;
  sub_18B4BA0(v41, v27, (__int64)v27 + v28);
  v43 = _mm_loadu_si128(&v30);
  v44 = v31;
  sub_15CAC60((__int64)v45, &v39);
  if ( (_QWORD *)v41[0] != v42 )
    j_j___libc_free_0(v41[0], v42[0] + 1LL);
  if ( (_QWORD *)v39.m128i_i64[0] != v40 )
    j_j___libc_free_0(v39.m128i_i64[0], v40[0] + 1LL);
  sub_15CAB20((__int64)v45, ": devirtualized a call to ", 0x1Au);
  sub_15C9800((__int64)v32, "FunctionName", 12, a4, a5);
  v39.m128i_i64[0] = (__int64)v40;
  sub_18B4BA0(v39.m128i_i64, v32[0], (__int64)&v32[0][(unsigned __int64)v32[1]]);
  v41[0] = (__int64)v42;
  sub_18B4BA0(v41, v34, (__int64)v34 + v35);
  v43 = _mm_loadu_si128(&v37);
  v44 = v38;
  sub_15CAC60((__int64)v45, &v39);
  if ( (_QWORD *)v41[0] != v42 )
    j_j___libc_free_0(v41[0], v42[0] + 1LL);
  if ( (_QWORD *)v39.m128i_i64[0] != v40 )
    j_j___libc_free_0(v39.m128i_i64[0], v40[0] + 1LL);
  sub_143AA50(v22, (__int64)v45);
  if ( v34 != &v36 )
    j_j___libc_free_0(v34, v36 + 1);
  if ( (__int64 *)v32[0] != &v33 )
    j_j___libc_free_0(v32[0], v33 + 1);
  if ( v27 != &v29 )
    j_j___libc_free_0(v27, v29 + 1);
  if ( (__int64 *)v25[0] != &v26 )
    j_j___libc_free_0(v25[0], v26 + 1);
  v15 = v46;
  v45[0] = &unk_49ECF68;
  v16 = &v46[88 * v47];
  if ( v46 != v16 )
  {
    do
    {
      v16 -= 88;
      v17 = (char *)*((_QWORD *)v16 + 4);
      if ( v17 != v16 + 48 )
        j_j___libc_free_0(v17, *((_QWORD *)v16 + 6) + 1LL);
      if ( *(char **)v16 != v16 + 16 )
        j_j___libc_free_0(*(_QWORD *)v16, *((_QWORD *)v16 + 2) + 1LL);
    }
    while ( v15 != v16 );
    v16 = v46;
  }
  if ( v16 != &v48 )
    _libc_free((unsigned __int64)v16);
  if ( v23 )
    sub_161E7C0((__int64)&v23, v23);
}
