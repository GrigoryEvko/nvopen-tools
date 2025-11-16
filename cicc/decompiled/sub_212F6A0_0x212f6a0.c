// Function: sub_212F6A0
// Address: 0x212f6a0
//
unsigned __int64 __fastcall sub_212F6A0(
        __int64 *a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __m128i a5,
        double a6,
        __m128i a7)
{
  unsigned int v8; // r15d
  __int64 v9; // r11
  __int64 v10; // rsi
  bool v11; // r8
  bool v12; // cl
  int v13; // eax
  __int64 v14; // r10
  unsigned __int8 v15; // r13
  __int64 v16; // rax
  bool v17; // cc
  __int64 v18; // r11
  int v19; // edx
  unsigned int v20; // eax
  __int64 v21; // rbx
  __int64 v22; // rax
  int v23; // edx
  unsigned __int64 result; // rax
  unsigned __int64 v25; // [rsp-10h] [rbp-E0h]
  __int64 v26; // [rsp+0h] [rbp-D0h]
  __int64 v27; // [rsp+0h] [rbp-D0h]
  bool v28; // [rsp+8h] [rbp-C8h]
  __int64 v29; // [rsp+8h] [rbp-C8h]
  __int64 *v32; // [rsp+20h] [rbp-B0h]
  bool v33; // [rsp+2Ch] [rbp-A4h]
  unsigned __int8 v34; // [rsp+2Ch] [rbp-A4h]
  unsigned int v35; // [rsp+50h] [rbp-80h] BYREF
  const void **v36; // [rsp+58h] [rbp-78h]
  __int64 v37; // [rsp+60h] [rbp-70h] BYREF
  int v38; // [rsp+68h] [rbp-68h]
  unsigned __int64 v39; // [rsp+70h] [rbp-60h] BYREF
  unsigned int v40; // [rsp+78h] [rbp-58h]
  __int64 v41; // [rsp+80h] [rbp-50h] BYREF
  unsigned int v42; // [rsp+88h] [rbp-48h]
  const void **v43; // [rsp+90h] [rbp-40h]

  sub_1F40D10(
    (__int64)&v41,
    *a1,
    *(_QWORD *)(a1[1] + 48),
    **(unsigned __int8 **)(a2 + 40),
    *(_QWORD *)(*(_QWORD *)(a2 + 40) + 8LL));
  LOBYTE(v35) = v42;
  v36 = v43;
  if ( (_BYTE)v42 )
    v8 = sub_2127930(v42);
  else
    v8 = sub_1F58D40((__int64)&v35);
  v9 = *(_QWORD *)(a2 + 88);
  v10 = *(_QWORD *)(a2 + 72);
  v11 = *(_WORD *)(a2 + 24) > 258;
  v32 = (__int64 *)(v9 + 24);
  v12 = (*(_BYTE *)(a2 + 26) & 8) != 0;
  v37 = v10;
  if ( v10 )
  {
    v26 = v9;
    v28 = v11;
    v33 = v12;
    sub_1623A60((__int64)&v37, v10, 2);
    v9 = v26;
    v11 = v28;
    v12 = v33;
  }
  v13 = *(_DWORD *)(a2 + 64);
  v14 = a1[1];
  v27 = v9;
  v15 = v11;
  v38 = v13;
  v29 = v14;
  v34 = v12;
  sub_16A5A50((__int64)&v41, v32, v8);
  v16 = sub_1D38970(v29, (__int64)&v41, (__int64)&v37, v35, v36, v15, a5, a6, a7, v34);
  v17 = v42 <= 0x40;
  v18 = v27;
  *(_QWORD *)a3 = v16;
  *(_DWORD *)(a3 + 8) = v19;
  if ( !v17 && v41 )
  {
    j_j___libc_free_0_0(v41);
    v18 = v27;
  }
  v20 = *(_DWORD *)(v18 + 32);
  v21 = a1[1];
  v40 = v20;
  if ( v20 > 0x40 )
  {
    sub_16A4FD0((__int64)&v39, (const void **)v32);
    v20 = v40;
    if ( v40 > 0x40 )
    {
      sub_16A8110((__int64)&v39, v8);
      goto LABEL_12;
    }
  }
  else
  {
    v39 = *(_QWORD *)(v18 + 24);
  }
  if ( v20 == v8 )
    v39 = 0;
  else
    v39 >>= v8;
LABEL_12:
  sub_16A5A50((__int64)&v41, (__int64 *)&v39, v8);
  v22 = sub_1D38970(v21, (__int64)&v41, (__int64)&v37, v35, v36, v15, a5, a6, a7, v34);
  v17 = v42 <= 0x40;
  *(_QWORD *)a4 = v22;
  *(_DWORD *)(a4 + 8) = v23;
  result = v25;
  if ( !v17 && v41 )
    result = j_j___libc_free_0_0(v41);
  if ( v40 > 0x40 && v39 )
    result = j_j___libc_free_0_0(v39);
  if ( v37 )
    return sub_161E7C0((__int64)&v37, v37);
  return result;
}
