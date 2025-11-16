// Function: sub_211D470
// Address: 0x211d470
//
void __fastcall sub_211D470(__int64 *a1, unsigned __int64 a2, __int64 a3, __int64 a4, double a5, double a6, __m128i a7)
{
  __int64 v10; // rsi
  const __m128i *v11; // roff
  __int128 v12; // xmm0
  __int64 v13; // r14
  __int64 v14; // r15
  __int64 v15; // rdx
  __int64 v16; // rsi
  __int64 v17; // rdx
  _QWORD *v18; // rdi
  __int64 v19; // r9
  __int64 v20; // rax
  __int64 v21; // rax
  char v22; // di
  int v23; // edx
  void *v24; // rax
  void *v25; // rbx
  const __m128i *v26; // r9
  int v27; // edx
  const void **v28; // r14
  __int64 v29; // rsi
  const void **v30; // rbx
  void *v31; // [rsp+8h] [rbp-D8h]
  __int64 v32; // [rsp+18h] [rbp-C8h]
  __int64 v34; // [rsp+30h] [rbp-B0h]
  __int64 v35; // [rsp+60h] [rbp-80h] BYREF
  int v36; // [rsp+68h] [rbp-78h]
  unsigned int v37; // [rsp+70h] [rbp-70h] BYREF
  const void **v38; // [rsp+78h] [rbp-68h]
  __int64 v39; // [rsp+80h] [rbp-60h] BYREF
  unsigned int v40; // [rsp+88h] [rbp-58h]
  char v41[8]; // [rsp+90h] [rbp-50h] BYREF
  void *v42; // [rsp+98h] [rbp-48h] BYREF
  const void **v43; // [rsp+A0h] [rbp-40h]

  if ( *(_WORD *)(a2 + 24) == 185 && (*(_BYTE *)(a2 + 27) & 0xC) == 0 && (*(_WORD *)(a2 + 26) & 0x380) == 0 )
  {
    sub_2144300();
    return;
  }
  v10 = *(_QWORD *)(a2 + 72);
  v11 = *(const __m128i **)(a2 + 32);
  v12 = (__int128)_mm_loadu_si128(v11);
  v13 = v11[2].m128i_i64[1];
  v35 = v10;
  v14 = v11[3].m128i_i64[0];
  if ( v10 )
    sub_1623A60((__int64)&v35, v10, 2);
  v15 = a1[1];
  v16 = *a1;
  v36 = *(_DWORD *)(a2 + 64);
  sub_1F40D10(
    (__int64)v41,
    v16,
    *(_QWORD *)(v15 + 48),
    **(unsigned __int8 **)(a2 + 40),
    *(_QWORD *)(*(_QWORD *)(a2 + 40) + 8LL));
  v17 = *(_QWORD *)(a2 + 96);
  LOBYTE(v16) = *(_BYTE *)(a2 + 27);
  v18 = (_QWORD *)a1[1];
  v19 = *(_QWORD *)(a2 + 104);
  LOBYTE(v37) = (_BYTE)v42;
  v20 = *(unsigned __int8 *)(a2 + 88);
  v38 = v43;
  v21 = sub_1D2B590(
          v18,
          ((unsigned __int8)v16 >> 2) & 3,
          (__int64)&v35,
          v37,
          (__int64)v43,
          v19,
          v12,
          v13,
          v14,
          v20,
          v17);
  v22 = v37;
  *(_QWORD *)a4 = v21;
  v34 = v21;
  *(_DWORD *)(a4 + 8) = v23;
  v32 = a1[1];
  if ( v22 )
  {
    v40 = sub_211A7A0(v22);
    if ( v40 <= 0x40 )
      goto LABEL_9;
LABEL_21:
    sub_16A4EF0((__int64)&v39, 0, 0);
    goto LABEL_10;
  }
  v40 = sub_1F58D40((__int64)&v37);
  if ( v40 > 0x40 )
    goto LABEL_21;
LABEL_9:
  v39 = 0;
LABEL_10:
  v31 = sub_1D15FA0(v37, (__int64)v38);
  v24 = sub_16982C0();
  v25 = v24;
  if ( v31 == v24 )
    sub_169D060(&v42, (__int64)v24, &v39);
  else
    sub_169D050((__int64)&v42, v31, &v39);
  *(_QWORD *)a3 = sub_1D36490(v32, (__int64)v41, (__int64)&v35, v37, v38, 0, *(double *)&v12, a6, a7);
  *(_DWORD *)(a3 + 8) = v27;
  if ( v42 == v25 )
  {
    v28 = v43;
    if ( v43 )
    {
      v29 = 4LL * (_QWORD)*(v43 - 1);
      v30 = &v43[v29];
      if ( v43 != &v43[v29] )
      {
        do
        {
          v30 -= 4;
          sub_127D120(v30 + 1);
        }
        while ( v28 != v30 );
      }
      j_j_j___libc_free_0_0(v28 - 1);
    }
  }
  else
  {
    sub_1698460((__int64)&v42);
  }
  if ( v40 > 0x40 && v39 )
    j_j___libc_free_0_0(v39);
  sub_2013400((__int64)a1, a2, 1, v34, (__m128i *)(*((_QWORD *)&v12 + 1) & 0xFFFFFFFF00000000LL | 1), v26);
  if ( v35 )
    sub_161E7C0((__int64)&v35, v35);
}
