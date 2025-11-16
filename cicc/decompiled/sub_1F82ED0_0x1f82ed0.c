// Function: sub_1F82ED0
// Address: 0x1f82ed0
//
__int64 *__fastcall sub_1F82ED0(
        __int64 *a1,
        unsigned int a2,
        __int64 a3,
        __int64 a4,
        unsigned __int64 a5,
        double a6,
        double a7,
        __m128i a8,
        __int64 a9,
        __int128 a10)
{
  __int64 v11; // r8
  unsigned __int64 v12; // r9
  int v13; // r10d
  unsigned __int8 *v15; // rax
  unsigned int v16; // r14d
  __int64 v19; // rax
  unsigned __int64 v20; // rax
  __int128 v21; // rax
  __int64 v22; // r13
  unsigned __int64 v23; // rax
  __int128 v24; // rax
  bool v25; // al
  __int64 v26; // rsi
  __int64 *v27; // r15
  __int64 v28; // rcx
  unsigned __int64 v29; // rdx
  __int64 *v30; // rdi
  __int64 v31; // rax
  bool v32; // al
  __int64 v33; // r8
  unsigned __int64 v34; // r9
  __int64 v35; // rsi
  __int64 *v36; // r13
  __int128 *v37; // rcx
  unsigned __int64 v38; // rdx
  __int64 v39; // [rsp+0h] [rbp-80h]
  int v40; // [rsp+18h] [rbp-68h]
  __int64 v41; // [rsp+18h] [rbp-68h]
  __int128 *v42; // [rsp+18h] [rbp-68h]
  __int64 v44; // [rsp+20h] [rbp-60h]
  __int64 *v45; // [rsp+20h] [rbp-60h]
  unsigned __int64 v46; // [rsp+28h] [rbp-58h]
  unsigned __int64 v47; // [rsp+28h] [rbp-58h]
  unsigned __int64 v48; // [rsp+28h] [rbp-58h]
  const void **v50; // [rsp+38h] [rbp-48h]
  __int64 v51; // [rsp+40h] [rbp-40h] BYREF
  int v52; // [rsp+48h] [rbp-38h]

  v11 = a4;
  v12 = a5;
  v13 = a5;
  v15 = (unsigned __int8 *)(*(_QWORD *)(a4 + 40) + 16LL * (unsigned int)a5);
  v16 = *v15;
  v50 = (const void **)*((_QWORD *)v15 + 1);
  if ( a2 == *(unsigned __int16 *)(a4 + 24) )
  {
    v46 = v12;
    v40 = v13;
    v19 = sub_1D23600(*a1, *(_QWORD *)(*(_QWORD *)(a4 + 32) + 40LL));
    v11 = a4;
    v12 = v46;
    v39 = v19;
    if ( v19 )
    {
      v20 = sub_1D23600(*a1, a10);
      if ( v20 )
      {
        *(_QWORD *)&v21 = sub_1D32920((_QWORD *)*a1, a2, a3, v16, (__int64)v50, v39, a6, a7, a8, v20);
        if ( (_QWORD)v21 )
          return sub_1D332F0(
                   (__int64 *)*a1,
                   a2,
                   a3,
                   v16,
                   v50,
                   0,
                   a6,
                   a7,
                   a8,
                   **(_QWORD **)(a4 + 32),
                   *(_QWORD *)(*(_QWORD *)(a4 + 32) + 8LL),
                   v21);
        return 0;
      }
      v25 = sub_1D18C00(a4, 1, v40);
      v11 = a4;
      v12 = v46;
      if ( v25 )
      {
        v26 = *(_QWORD *)(a4 + 72);
        v27 = (__int64 *)*a1;
        v28 = *(_QWORD *)(a4 + 32);
        v51 = v26;
        if ( v26 )
        {
          v41 = v28;
          sub_1623A60((__int64)&v51, v26, 2);
          v28 = v41;
        }
        v52 = *(_DWORD *)(a4 + 64);
        v45 = sub_1D332F0(v27, a2, (__int64)&v51, v16, v50, 0, a6, a7, a8, *(_QWORD *)v28, *(_QWORD *)(v28 + 8), a10);
        v48 = v29;
        if ( v51 )
          sub_161E7C0((__int64)&v51, v51);
        if ( !v45 )
          return 0;
        sub_1F81BC0((__int64)a1, (__int64)v45);
        v30 = (__int64 *)*a1;
        v31 = *(_QWORD *)(a4 + 32);
        return sub_1D332F0(v30, a2, a3, v16, v50, 0, a6, a7, a8, (__int64)v45, v48, *(_OWORD *)(v31 + 40));
      }
    }
  }
  if ( a2 != *(unsigned __int16 *)(a10 + 24) )
    return 0;
  v44 = v11;
  v47 = v12;
  v22 = sub_1D23600(*a1, *(_QWORD *)(*(_QWORD *)(a10 + 32) + 40LL));
  if ( !v22 )
    return 0;
  v23 = sub_1D23600(*a1, v44);
  if ( !v23 )
  {
    v32 = sub_1D18C00(a10, 1, SDWORD2(a10));
    v33 = v44;
    v34 = v47;
    if ( !v32 )
      return 0;
    v35 = *(_QWORD *)(a10 + 72);
    v36 = (__int64 *)*a1;
    v37 = *(__int128 **)(a10 + 32);
    v51 = v35;
    if ( v35 )
    {
      v42 = v37;
      sub_1623A60((__int64)&v51, v35, 2);
      v33 = v44;
      v34 = v47;
      v37 = v42;
    }
    v52 = *(_DWORD *)(a10 + 64);
    v45 = sub_1D332F0(v36, a2, (__int64)&v51, v16, v50, 0, a6, a7, a8, v33, v34, *v37);
    v48 = v38;
    if ( v51 )
      sub_161E7C0((__int64)&v51, v51);
    if ( !v45 )
      return 0;
    sub_1F81BC0((__int64)a1, (__int64)v45);
    v30 = (__int64 *)*a1;
    v31 = *(_QWORD *)(a10 + 32);
    return sub_1D332F0(v30, a2, a3, v16, v50, 0, a6, a7, a8, (__int64)v45, v48, *(_OWORD *)(v31 + 40));
  }
  *(_QWORD *)&v24 = sub_1D32920((_QWORD *)*a1, a2, a3, v16, (__int64)v50, v22, a6, a7, a8, v23);
  if ( !(_QWORD)v24 )
    return 0;
  return sub_1D332F0(
           (__int64 *)*a1,
           a2,
           a3,
           v16,
           v50,
           0,
           a6,
           a7,
           a8,
           **(_QWORD **)(a10 + 32),
           *(_QWORD *)(*(_QWORD *)(a10 + 32) + 8LL),
           v24);
}
