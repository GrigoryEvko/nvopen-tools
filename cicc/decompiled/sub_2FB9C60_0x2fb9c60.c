// Function: sub_2FB9C60
// Address: 0x2fb9c60
//
unsigned __int64 __fastcall sub_2FB9C60(
        __int64 *a1,
        __int64 a2,
        __int32 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int64 *a7,
        char a8,
        int a9)
{
  __int32 v9; // r10d
  __int32 v11; // r14d
  __int64 v12; // r13
  _QWORD *v14; // rcx
  __int64 (*v15)(); // rax
  __int64 v16; // r13
  _QWORD *v17; // rax
  __int64 v18; // rdx
  unsigned __int64 v19; // r13
  unsigned __int64 result; // rax
  __int64 v21; // rax
  __int64 v22; // rdx
  __int64 v23; // r8
  __int64 v24; // r9
  __int64 v25; // rax
  __int64 v26; // rdi
  __int64 v27; // rsi
  char v28; // al
  __int32 v29; // r10d
  __int32 v30; // r13d
  unsigned __int64 v31; // rax
  int *v32; // rbx
  int v33; // r9d
  __int64 v34; // rbx
  __int64 *v35; // rsi
  unsigned int v36; // eax
  _QWORD *v37; // [rsp-10h] [rbp-E0h]
  __int64 v38; // [rsp+8h] [rbp-C8h]
  __int32 v39; // [rsp+18h] [rbp-B8h]
  __int64 v40; // [rsp+18h] [rbp-B8h]
  __int32 v41; // [rsp+20h] [rbp-B0h]
  __int64 v42; // [rsp+20h] [rbp-B0h]
  __int64 v44; // [rsp+28h] [rbp-A8h]
  _QWORD *v45; // [rsp+28h] [rbp-A8h]
  __int64 v48; // [rsp+48h] [rbp-88h]
  __int64 v49; // [rsp+50h] [rbp-80h] BYREF
  __int64 *v50; // [rsp+58h] [rbp-78h]
  __int64 (__fastcall *v51)(const __m128i **, const __m128i *, int); // [rsp+60h] [rbp-70h]
  unsigned __int64 (__fastcall *v52)(__int64, __int64, __int64, __int64, __int64, unsigned __int64); // [rsp+68h] [rbp-68h]
  __m128i v53; // [rsp+70h] [rbp-60h] BYREF
  _QWORD v54[10]; // [rsp+80h] [rbp-50h] BYREF

  v9 = a3;
  v11 = a2;
  v12 = -800;
  v14 = (_QWORD *)a1[5];
  v15 = *(__int64 (**)())(*v14 + 1344LL);
  if ( v15 != sub_2FB0290 )
  {
    v45 = (_QWORD *)a1[5];
    v36 = ((__int64 (__fastcall *)(_QWORD *, __int64, _QWORD))v15)(v14, a2, *(_QWORD *)(a6 + 32));
    v9 = a3;
    v14 = v45;
    v12 = -40LL * v36;
  }
  v16 = v14[1] + v12;
  v44 = *(_QWORD *)(a1[1] + 32);
  if ( (a4 & a5) == 0xFFFFFFFFFFFFFFFFLL || (v41 = v9, v21 = sub_2EBF1E0(a1[3], a2), v9 = v41, a4 == v21) && a5 == v22 )
  {
    v48 = 0;
    v49 = 0;
    v50 = 0;
    v51 = 0;
    v17 = sub_2F26260(a6, a7, &v49, v16, v9);
    v53.m128i_i64[0] = 0;
    v19 = v18;
    v53.m128i_i32[2] = a2;
    memset(v54, 0, 24);
    sub_2E8EAD0(v18, (__int64)v17, &v53);
    if ( v49 )
      sub_B91220((__int64)&v49, v49);
    return sub_2E192D0(v44, v19, a8) & 0xFFFFFFFFFFFFFFF8LL | 4;
  }
  else
  {
    v39 = v41;
    v25 = sub_2DF8570(
            a1[1],
            *(_DWORD *)(**(_QWORD **)(a1[9] + 16) + 4LL * (unsigned int)(*(_DWORD *)(a1[9] + 64) + a9)),
            **(_QWORD **)(a1[9] + 16),
            *(_QWORD *)(a1[9] + 16),
            v23,
            v24);
    v26 = a1[6];
    v42 = v25;
    v27 = *(_QWORD *)(*(_QWORD *)(a1[3] + 56) + 16 * (a2 & 0x7FFFFFFF));
    v53.m128i_i64[0] = (__int64)v54;
    v53.m128i_i64[1] = 0x800000000LL;
    v28 = sub_2FF7100(v26, v27 & 0xFFFFFFFFFFFFFFF8LL, a4, a5, &v53);
    v29 = v39;
    if ( !v28 )
      sub_C64ED0("Impossible to implement partial COPY", 1u);
    v40 = v53.m128i_i64[0] + 4LL * v53.m128i_u32[2];
    if ( v40 == v53.m128i_i64[0] )
    {
      v34 = 0;
    }
    else
    {
      v38 = v16;
      v30 = v29;
      v31 = 0;
      v32 = (int *)v53.m128i_i64[0];
      do
      {
        v33 = *v32++;
        v31 = sub_2FB9A10((__int64)a1, v11, v30, a6, (unsigned __int64 *)a7, v33, v42, a8, v31, v38);
      }
      while ( (int *)v40 != v32 );
      v34 = v31;
    }
    v37 = (_QWORD *)a1[6];
    v35 = (__int64 *)(a1[1] + 56);
    v49 = v34;
    v52 = sub_2FB02B0;
    v51 = sub_2FB02D0;
    v50 = v35;
    sub_2E0C490(v42, v35, a4, a5, (unsigned __int64)&v49, v44, v37, 0);
    if ( v51 )
      v51((const __m128i **)&v49, (const __m128i *)&v49, 3);
    result = v34;
    if ( (_QWORD *)v53.m128i_i64[0] != v54 )
    {
      _libc_free(v53.m128i_u64[0]);
      return v34;
    }
  }
  return result;
}
