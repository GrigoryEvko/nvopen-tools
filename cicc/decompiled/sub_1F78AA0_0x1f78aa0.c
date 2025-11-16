// Function: sub_1F78AA0
// Address: 0x1f78aa0
//
__int64 *__fastcall sub_1F78AA0(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        unsigned __int64 a4,
        int a5,
        int a6,
        double a7,
        double a8,
        __m128i a9)
{
  unsigned int v11; // ebx
  __int64 v12; // rax
  __int64 v13; // rax
  __int64 v14; // r13
  void *v15; // rax
  void *v16; // rdx
  __int64 v17; // r13
  void *v18; // rax
  __int64 *v19; // r12
  __int64 v20; // rax
  __int64 v21; // rdx
  __int64 v22; // rdi
  bool v23; // al
  __int64 v24; // rdi
  __int64 v25; // r13
  __int64 v26; // r12
  __int64 v27; // rdx
  __int64 v28; // rbx
  __int64 v29; // r13
  void *v30; // [rsp+0h] [rbp-C0h]
  void *v31; // [rsp+8h] [rbp-B8h]
  unsigned __int16 v32; // [rsp+8h] [rbp-B8h]
  __int64 v33; // [rsp+10h] [rbp-B0h]
  __int16 *v34; // [rsp+20h] [rbp-A0h]
  void *v35; // [rsp+28h] [rbp-98h]
  bool v36; // [rsp+37h] [rbp-89h]
  unsigned __int128 v37; // [rsp+40h] [rbp-80h]
  __int64 v38[4]; // [rsp+50h] [rbp-70h] BYREF
  char v39[8]; // [rsp+70h] [rbp-50h] BYREF
  void *v40; // [rsp+78h] [rbp-48h] BYREF
  __int64 v41; // [rsp+80h] [rbp-40h]

  v37 = __PAIR128__(a4, a3);
  if ( *(_WORD *)(a2 + 24) != 76 )
    return 0;
  v11 = a5;
  if ( !**(_BYTE **)a1 )
  {
    v12 = *(_QWORD *)(a2 + 48);
    if ( !v12 || *(_QWORD *)(v12 + 32) )
      return 0;
  }
  v13 = sub_1D23470(*(_QWORD *)(*(_QWORD *)(a2 + 32) + 40LL), *(_QWORD *)(*(_QWORD *)(a2 + 32) + 48LL), a3, a4, a5, a6);
  v33 = v13;
  if ( !v13 )
    return 0;
  v14 = *(_QWORD *)(v13 + 88);
  v34 = (__int16 *)sub_1698280();
  sub_169D3F0((__int64)v38, 1.0);
  sub_169E320(&v40, v38, v34);
  sub_1698460((__int64)v38);
  sub_16A3360((__int64)v39, *(__int16 **)(v14 + 32), 0, (bool *)v38);
  v30 = v40;
  v31 = *(void **)(v14 + 32);
  v15 = sub_16982C0();
  v36 = 0;
  v16 = v30;
  v35 = v15;
  if ( v31 == v30 )
  {
    v22 = v14 + 32;
    if ( v15 == v30 )
      v23 = sub_169CB90(v22, (__int64)&v40);
    else
      v23 = sub_1698510(v22, (__int64)&v40);
    v16 = v40;
    v36 = v23;
  }
  if ( v35 == v16 )
  {
    v27 = v41;
    if ( v41 )
    {
      if ( v41 != v41 + 32LL * *(_QWORD *)(v41 - 8) )
      {
        v32 = v11;
        v28 = v41 + 32LL * *(_QWORD *)(v41 - 8);
        v29 = v41;
        do
        {
          v28 -= 32;
          sub_127D120((_QWORD *)(v28 + 8));
        }
        while ( v29 != v28 );
        v11 = v32;
        v27 = v29;
      }
      j_j_j___libc_free_0_0(v27 - 8);
    }
  }
  else
  {
    sub_1698460((__int64)&v40);
  }
  if ( !v36 )
  {
    v17 = *(_QWORD *)(v33 + 88);
    sub_169D3F0((__int64)v38, -1.0);
    sub_169E320(&v40, v38, v34);
    sub_1698460((__int64)v38);
    sub_16A3360((__int64)v39, *(__int16 **)(v17 + 32), 0, (bool *)v38);
    v18 = v40;
    if ( *(void **)(v17 + 32) == v40 )
    {
      v24 = v17 + 32;
      if ( v35 == v40 )
        v36 = sub_169CB90(v24, (__int64)&v40);
      else
        v36 = sub_1698510(v24, (__int64)&v40);
      v18 = v40;
    }
    if ( v35 == v18 )
    {
      v25 = v41;
      if ( v41 )
      {
        v26 = v41 + 32LL * *(_QWORD *)(v41 - 8);
        if ( v41 != v26 )
        {
          do
          {
            v26 -= 32;
            sub_127D120((_QWORD *)(v26 + 8));
          }
          while ( v25 != v26 );
        }
        j_j_j___libc_free_0_0(v25 - 8);
      }
    }
    else
    {
      sub_1698460((__int64)&v40);
    }
    if ( v36 )
    {
      v19 = **(__int64 ***)(a1 + 8);
      v20 = sub_1D309E0(
              v19,
              162,
              *(_QWORD *)(a1 + 24),
              **(unsigned int **)(a1 + 32),
              *(const void ***)(*(_QWORD *)(a1 + 32) + 8LL),
              0,
              -1.0,
              a8,
              *(double *)a9.m128i_i64,
              v37);
      return sub_1D3A900(
               v19,
               **(_DWORD **)(a1 + 16),
               *(_QWORD *)(a1 + 24),
               **(unsigned int **)(a1 + 32),
               *(const void ***)(*(_QWORD *)(a1 + 32) + 8LL),
               v11,
               (__m128)0xBFF0000000000000LL,
               a8,
               a9,
               **(_QWORD **)(a2 + 32),
               *(__int16 **)(*(_QWORD *)(a2 + 32) + 8LL),
               v37,
               v20,
               v21);
    }
    return 0;
  }
  return sub_1D3A900(
           **(__int64 ***)(a1 + 8),
           **(_DWORD **)(a1 + 16),
           *(_QWORD *)(a1 + 24),
           **(unsigned int **)(a1 + 32),
           *(const void ***)(*(_QWORD *)(a1 + 32) + 8LL),
           v11,
           (__m128)0x3FF0000000000000uLL,
           a8,
           a9,
           **(_QWORD **)(a2 + 32),
           *(__int16 **)(*(_QWORD *)(a2 + 32) + 8LL),
           v37,
           v37,
           *((__int64 *)&v37 + 1));
}
