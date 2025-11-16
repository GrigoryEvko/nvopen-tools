// Function: sub_1F97690
// Address: 0x1f97690
//
__int64 *__fastcall sub_1F97690(
        __int64 *a1,
        __int64 a2,
        __int64 a3,
        unsigned int a4,
        const void **a5,
        __m128i a6,
        double a7,
        __m128i a8)
{
  __int64 v8; // r10
  __int64 v9; // rax
  __int64 v13; // rcx
  __int64 *result; // rax
  __int64 v15; // rsi
  unsigned __int64 v16; // rdi
  unsigned int v17; // r13d
  __int64 v18; // rax
  __int64 v19; // r14
  unsigned __int64 v20; // rdx
  unsigned __int64 v21; // r15
  __int64 v22; // rbx
  __int64 v23; // rcx
  __int64 v24; // r8
  __int64 v25; // r9
  __int64 *v26; // r12
  __int64 v27; // rax
  __int64 v28; // rdx
  __int64 v29; // rdi
  const void ***v30; // rdx
  __int128 v31; // [rsp-10h] [rbp-80h]
  const void **v32; // [rsp+8h] [rbp-68h]
  __int64 v33; // [rsp+18h] [rbp-58h]
  __int64 v34; // [rsp+18h] [rbp-58h]
  __int64 *v35; // [rsp+18h] [rbp-58h]
  char v36; // [rsp+2Fh] [rbp-41h] BYREF
  __int64 v37; // [rsp+30h] [rbp-40h] BYREF
  int v38; // [rsp+38h] [rbp-38h]

  v8 = a2;
  v9 = 1;
  v13 = a1[1];
  if ( (_BYTE)a4 != 1 )
  {
    if ( !(_BYTE)a4 )
      return 0;
    v9 = (unsigned __int8)a4;
    if ( !*(_QWORD *)(v13 + 8LL * (unsigned __int8)a4 + 120) )
      return 0;
  }
  if ( *(_BYTE *)(v13 + 259 * v9 + 2570) )
    return 0;
  v15 = *(_QWORD *)(a2 + 72);
  v16 = *(_QWORD *)(*(_QWORD *)(v8 + 40) + 16LL * (unsigned int)a3 + 8);
  v17 = *(unsigned __int8 *)(*(_QWORD *)(v8 + 40) + 16LL * (unsigned int)a3);
  v37 = v15;
  if ( v15 )
  {
    v32 = a5;
    v33 = v8;
    sub_1623A60((__int64)&v37, v15, 2);
    a5 = v32;
    v8 = v33;
  }
  v34 = v8;
  v38 = *(_DWORD *)(v8 + 64);
  v36 = 0;
  v18 = sub_1F973B0(a1, v8, a3, a4, a5, &v36, a6, a7, a8);
  v19 = v18;
  v21 = v20;
  v22 = v18;
  if ( v18 )
  {
    sub_1F81BC0((__int64)a1, v18);
    if ( v36 )
      sub_1F97190(a1, v34, v19, *(double *)a6.m128i_i64, a7, *(double *)a8.m128i_i64);
    v26 = (__int64 *)*a1;
    v27 = sub_1D2EF30(v26, v17, v16, v23, v24, v25);
    v29 = v28;
    v30 = (const void ***)(*(_QWORD *)(v22 + 40) + 16LL * (unsigned int)v21);
    *((_QWORD *)&v31 + 1) = v29;
    *(_QWORD *)&v31 = v27;
    result = sub_1D332F0(
               v26,
               148,
               (__int64)&v37,
               *(unsigned __int8 *)v30,
               v30[1],
               0,
               *(double *)a6.m128i_i64,
               a7,
               a8,
               v19,
               v21,
               v31);
  }
  else
  {
    result = 0;
  }
  if ( v37 )
  {
    v35 = result;
    sub_161E7C0((__int64)&v37, v37);
    return v35;
  }
  return result;
}
