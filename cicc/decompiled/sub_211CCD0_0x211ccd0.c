// Function: sub_211CCD0
// Address: 0x211ccd0
//
unsigned __int64 __fastcall sub_211CCD0(
        __int64 *a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        double a5,
        double a6,
        __m128i a7)
{
  __int64 v10; // rsi
  __int64 *v11; // rdi
  __int64 v12; // rax
  char v13; // di
  int v14; // edx
  __int64 v15; // r12
  void *v16; // rax
  void *v17; // rbx
  unsigned int v18; // edx
  unsigned __int64 result; // rax
  const void **v20; // r12
  __int64 v21; // rsi
  const void **v22; // rbx
  void *v23; // [rsp+10h] [rbp-B0h]
  unsigned int v25; // [rsp+40h] [rbp-80h] BYREF
  const void **v26; // [rsp+48h] [rbp-78h]
  __int64 v27; // [rsp+50h] [rbp-70h] BYREF
  int v28; // [rsp+58h] [rbp-68h]
  __int64 v29; // [rsp+60h] [rbp-60h] BYREF
  unsigned int v30; // [rsp+68h] [rbp-58h]
  char v31[8]; // [rsp+70h] [rbp-50h] BYREF
  void *v32; // [rsp+78h] [rbp-48h] BYREF
  const void **v33; // [rsp+80h] [rbp-40h]

  sub_1F40D10(
    (__int64)v31,
    *a1,
    *(_QWORD *)(a1[1] + 48),
    **(unsigned __int8 **)(a2 + 40),
    *(_QWORD *)(*(_QWORD *)(a2 + 40) + 8LL));
  v10 = *(_QWORD *)(a2 + 72);
  LOBYTE(v25) = (_BYTE)v32;
  v27 = v10;
  v26 = v33;
  if ( v10 )
    sub_1623A60((__int64)&v27, v10, 2);
  v11 = (__int64 *)a1[1];
  v28 = *(_DWORD *)(a2 + 64);
  v12 = sub_1D309E0(
          v11,
          157,
          (__int64)&v27,
          v25,
          v26,
          0,
          a5,
          a6,
          *(double *)a7.m128i_i64,
          *(_OWORD *)*(_QWORD *)(a2 + 32));
  v13 = v25;
  *(_QWORD *)a4 = v12;
  *(_DWORD *)(a4 + 8) = v14;
  v15 = a1[1];
  if ( v13 )
  {
    v30 = sub_211A7A0(v13);
    if ( v30 <= 0x40 )
    {
LABEL_5:
      v29 = 0;
      goto LABEL_6;
    }
  }
  else
  {
    v30 = sub_1F58D40((__int64)&v25);
    if ( v30 <= 0x40 )
      goto LABEL_5;
  }
  sub_16A4EF0((__int64)&v29, 0, 0);
LABEL_6:
  v23 = sub_1D15FA0(v25, (__int64)v26);
  v16 = sub_16982C0();
  v17 = v16;
  if ( v23 == v16 )
    sub_169D060(&v32, (__int64)v16, &v29);
  else
    sub_169D050((__int64)&v32, v23, &v29);
  *(_QWORD *)a3 = sub_1D36490(v15, (__int64)v31, (__int64)&v27, v25, v26, 0, a5, a6, a7);
  result = v18;
  *(_DWORD *)(a3 + 8) = v18;
  if ( v32 == v17 )
  {
    v20 = v33;
    if ( v33 )
    {
      v21 = 4LL * (_QWORD)*(v33 - 1);
      v22 = &v33[v21];
      if ( v33 != &v33[v21] )
      {
        do
        {
          v22 -= 4;
          sub_127D120(v22 + 1);
        }
        while ( v20 != v22 );
      }
      result = j_j_j___libc_free_0_0(v20 - 1);
    }
  }
  else
  {
    result = sub_1698460((__int64)&v32);
  }
  if ( v30 > 0x40 && v29 )
    result = j_j___libc_free_0_0(v29);
  if ( v27 )
    return sub_161E7C0((__int64)&v27, v27);
  return result;
}
