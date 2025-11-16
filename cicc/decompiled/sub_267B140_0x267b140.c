// Function: sub_267B140
// Address: 0x267b140
//
__int64 __fastcall sub_267B140(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v5; // rdi
  __int64 v6; // rdx
  __int64 v7; // rcx
  __int64 v8; // r8
  __int64 v9; // r9
  __m128i v10; // xmm2
  __m128i v11; // xmm3
  __m128i v12; // xmm4
  __int64 v13; // rax
  __int8 *v14; // rdi
  __int64 v16; // rax
  __int8 *v17; // rdx
  __int8 *v18; // [rsp+10h] [rbp-F0h] BYREF
  _BYTE *v19; // [rsp+18h] [rbp-E8h]
  _QWORD v20[2]; // [rsp+20h] [rbp-E0h] BYREF
  _BYTE *v21[2]; // [rsp+30h] [rbp-D0h] BYREF
  __int64 v22; // [rsp+40h] [rbp-C0h] BYREF
  __int64 *v23; // [rsp+50h] [rbp-B0h]
  __int64 v24; // [rsp+58h] [rbp-A8h]
  __int64 v25; // [rsp+60h] [rbp-A0h] BYREF
  __m128i v26; // [rsp+70h] [rbp-90h] BYREF
  unsigned __int64 v27; // [rsp+80h] [rbp-80h] BYREF
  __int64 v28; // [rsp+88h] [rbp-78h]
  __int64 v29; // [rsp+90h] [rbp-70h] BYREF
  char v30; // [rsp+98h] [rbp-68h] BYREF
  _QWORD *v31; // [rsp+A0h] [rbp-60h] BYREF
  _QWORD v32[2]; // [rsp+B0h] [rbp-50h] BYREF
  __m128i v33; // [rsp+C0h] [rbp-40h]

  sub_B18290(a3, "OpenMP ICV ", 0xBu);
  sub_B16430((__int64)v21, "OpenMPICV", 9u, *(_BYTE **)(*(_QWORD *)a2 + 8LL), *(_QWORD *)(*(_QWORD *)a2 + 16LL));
  v27 = (unsigned __int64)&v29;
  sub_266F100((__int64 *)&v27, v21[0], (__int64)&v21[0][(unsigned __int64)v21[1]]);
  v31 = v32;
  sub_266F100((__int64 *)&v31, v23, (__int64)v23 + v24);
  v33 = _mm_loadu_si128(&v26);
  sub_B180C0(a3, (unsigned __int64)&v27);
  if ( v31 != v32 )
    j_j___libc_free_0((unsigned __int64)v31);
  if ( (__int64 *)v27 != &v29 )
    j_j___libc_free_0(v27);
  sub_B18290(a3, " Value: ", 8u);
  v5 = *(_QWORD *)(*(_QWORD *)a2 + 48LL);
  if ( v5 )
  {
    v28 = 0;
    v27 = (unsigned __int64)&v30;
    v29 = 40;
    sub_C48A30(v5 + 24, &v27, 0xAu, 1, 0, 1, 0);
    v18 = (__int8 *)v20;
    sub_266E6F0((__int64 *)&v18, (_BYTE *)v27, v27 + v28);
    if ( (char *)v27 != &v30 )
      _libc_free(v27);
  }
  else
  {
    v27 = 22;
    v18 = (__int8 *)v20;
    v16 = sub_22409D0((__int64)&v18, &v27, 0);
    v18 = (__int8 *)v16;
    v20[0] = v27;
    *(__m128i *)v16 = _mm_load_si128((const __m128i *)&xmmword_438FCE0);
    *(_WORD *)(v16 + 20) = 17477;
    v17 = v18;
    *(_DWORD *)(v16 + 16) = 1313424965;
    v19 = (_BYTE *)v27;
    v17[v27] = 0;
  }
  sub_B18290(a3, v18, (size_t)v19);
  *(_DWORD *)(a1 + 8) = *(_DWORD *)(a3 + 8);
  *(_BYTE *)(a1 + 12) = *(_BYTE *)(a3 + 12);
  v10 = _mm_loadu_si128((const __m128i *)(a3 + 24));
  *(_QWORD *)(a1 + 16) = *(_QWORD *)(a3 + 16);
  *(__m128i *)(a1 + 24) = v10;
  v11 = _mm_loadu_si128((const __m128i *)(a3 + 48));
  v12 = _mm_loadu_si128((const __m128i *)(a3 + 64));
  *(_QWORD *)a1 = &unk_49D9D40;
  v13 = *(_QWORD *)(a3 + 40);
  *(__m128i *)(a1 + 48) = v11;
  *(_QWORD *)(a1 + 40) = v13;
  *(_QWORD *)(a1 + 80) = a1 + 96;
  *(_QWORD *)(a1 + 88) = 0x400000000LL;
  *(__m128i *)(a1 + 64) = v12;
  if ( *(_DWORD *)(a3 + 88) )
    sub_26781A0(a1 + 80, a3 + 80, v6, v7, v8, v9);
  v14 = v18;
  *(_BYTE *)(a1 + 416) = *(_BYTE *)(a3 + 416);
  *(_DWORD *)(a1 + 420) = *(_DWORD *)(a3 + 420);
  *(_QWORD *)(a1 + 424) = *(_QWORD *)(a3 + 424);
  *(_QWORD *)a1 = &unk_49D9DE8;
  if ( v14 != (__int8 *)v20 )
    j_j___libc_free_0((unsigned __int64)v14);
  if ( v23 != &v25 )
    j_j___libc_free_0((unsigned __int64)v23);
  if ( (__int64 *)v21[0] != &v22 )
    j_j___libc_free_0((unsigned __int64)v21[0]);
  return a1;
}
