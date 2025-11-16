// Function: sub_1BF3C70
// Address: 0x1bf3c70
//
__int64 __fastcall sub_1BF3C70(__int64 a1, __m128i a2, __m128i a3, __int64 a4, __int64 a5)
{
  __int64 v5; // rax
  __int64 v6; // rsi
  __int64 v8; // rax
  __int64 v9; // r15
  __int64 *v10; // r12
  __int64 v11; // rax
  char *v12; // rax
  __int64 v13; // rdi
  __int64 v14; // rsi
  __int64 v15; // rdx
  __int64 v16; // rcx
  __int64 v17; // r9
  char v18; // r10
  int v19; // r11d
  __int64 v20; // r13
  __int64 v21; // rax
  __int64 v22; // r15
  unsigned int v23; // edx
  __m128 *v24; // r14
  __int64 *v25; // rdi
  _QWORD *v26; // r14
  _QWORD *v27; // r12
  _QWORD *v28; // rdi
  unsigned int v29; // r12d
  __int64 v30; // r13
  __int64 v31; // rax
  __int64 v33; // r14
  _QWORD *v34; // r13
  char *v35; // rax
  _QWORD *v36; // rbx
  _QWORD *v37; // r12
  _QWORD *v38; // rdi
  __int64 v39; // rax
  __int64 v40; // rax
  void *v41; // [rsp+20h] [rbp-210h] BYREF
  int v42; // [rsp+28h] [rbp-208h]
  char v43; // [rsp+2Ch] [rbp-204h]
  __int64 v44; // [rsp+30h] [rbp-200h]
  __int64 v45; // [rsp+38h] [rbp-1F8h]
  __int64 v46; // [rsp+40h] [rbp-1F0h]
  __int64 v47; // [rsp+48h] [rbp-1E8h]
  char *v48; // [rsp+50h] [rbp-1E0h]
  __int64 v49; // [rsp+58h] [rbp-1D8h]
  __int64 v50; // [rsp+60h] [rbp-1D0h]
  char v51; // [rsp+70h] [rbp-1C0h]
  _BYTE *v52; // [rsp+78h] [rbp-1B8h] BYREF
  __int64 v53; // [rsp+80h] [rbp-1B0h]
  _BYTE v54[356]; // [rsp+88h] [rbp-1A8h] BYREF
  int v55; // [rsp+1ECh] [rbp-44h]
  __int64 v56; // [rsp+1F0h] [rbp-40h]

  v5 = *(_QWORD *)(a1 + 40);
  v6 = *(_QWORD *)a1;
  if ( !*(_QWORD *)(v5 + 16) )
    sub_4263D6(a1, v6, a5);
  v8 = (*(__int64 (__fastcall **)(_QWORD, __int64))(v5 + 24))(*(_QWORD *)(a1 + 40), v6);
  *(_QWORD *)(a1 + 48) = v8;
  v9 = *(_QWORD *)(v8 + 56);
  if ( v9 )
  {
    v10 = *(__int64 **)(a1 + 56);
    v11 = sub_15E0530(*v10);
    if ( sub_1602790(v11)
      || (v39 = sub_15E0530(*v10),
          v40 = sub_16033E0(v39),
          (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v40 + 48LL))(v40)) )
    {
      v12 = sub_1BF18B0(*(_QWORD *)(a1 + 464));
      v13 = *(_QWORD *)(v9 + 24);
      v14 = *(_QWORD *)(v9 + 32);
      v15 = *(_QWORD *)(v9 + 56);
      v16 = *(_QWORD *)(v9 + 40);
      v17 = *(_QWORD *)(v9 + 16);
      v18 = *(_BYTE *)(v9 + 12);
      v19 = *(_DWORD *)(v9 + 8);
      v50 = *(_QWORD *)(v9 + 64);
      v52 = v54;
      v53 = 0x400000000LL;
      v45 = v13;
      v46 = v14;
      v48 = v12;
      v49 = v15;
      v42 = v19;
      v43 = v18;
      v44 = v17;
      v47 = v16;
      v51 = 0;
      v54[352] = 0;
      v55 = -1;
      v41 = &unk_49F5BC8;
      v56 = *(_QWORD *)(v9 + 464);
      sub_15CAB20((__int64)&v41, "loop not vectorized: ", 0x15u);
      v20 = *(_QWORD *)(v9 + 88);
      v21 = 11LL * *(unsigned int *)(v9 + 96);
      v22 = *(unsigned int *)(v9 + 96);
      if ( v21 )
      {
        v23 = v53;
        do
        {
          if ( v23 >= HIDWORD(v53) )
          {
            sub_14B3F20((__int64)&v52, 0);
            v23 = v53;
          }
          v24 = (__m128 *)&v52[88 * v23];
          if ( v24 )
          {
            v25 = (__int64 *)&v52[88 * v23];
            v24->m128_u64[0] = (unsigned __int64)&v24[1];
            sub_1BF0E30(v25, *(_BYTE **)v20, *(_QWORD *)v20 + *(_QWORD *)(v20 + 8));
            v24[2].m128_u64[0] = (unsigned __int64)&v24[3];
            sub_1BF0E30((__int64 *)&v24[2], *(_BYTE **)(v20 + 32), *(_QWORD *)(v20 + 32) + *(_QWORD *)(v20 + 40));
            a2 = _mm_loadu_si128((const __m128i *)(v20 + 64));
            v24[4] = (__m128)a2;
            v24[5].m128_u64[0] = *(_QWORD *)(v20 + 80);
            v23 = v53;
          }
          ++v23;
          v20 += 88;
          LODWORD(v53) = v23;
          --v22;
        }
        while ( v22 );
      }
      v41 = &unk_49ECFF8;
      sub_143AA50(v10, (__int64)&v41);
      v26 = v52;
      v41 = &unk_49ECF68;
      v27 = &v52[88 * (unsigned int)v53];
      if ( v52 != (_BYTE *)v27 )
      {
        do
        {
          v27 -= 11;
          v28 = (_QWORD *)v27[4];
          if ( v28 != v27 + 6 )
            j_j___libc_free_0(v28, v27[6] + 1LL);
          if ( (_QWORD *)*v27 != v27 + 2 )
            j_j___libc_free_0(*v27, v27[2] + 1LL);
        }
        while ( v26 != v27 );
        v27 = v52;
      }
      if ( v27 != (_QWORD *)v54 )
        _libc_free((unsigned __int64)v27);
    }
    v8 = *(_QWORD *)(a1 + 48);
  }
  v29 = *(unsigned __int8 *)(v8 + 48);
  if ( (_BYTE)v29 )
  {
    if ( *(_BYTE *)(v8 + 49) )
    {
      v33 = *(_QWORD *)a1;
      v34 = *(_QWORD **)(a1 + 56);
      v35 = sub_1BF18B0(*(_QWORD *)(a1 + 464));
      sub_1BF1750((__int64)&v41, (__int64)v35, (__int64)"CantVectorizeStoreToLoopInvariantAddress", 40, v33, 0);
      sub_15CAB20((__int64)&v41, "write to a loop invariant address could not be vectorized", 0x39u);
      sub_143AA50(v34, (__int64)&v41);
      v36 = v52;
      v41 = &unk_49ECF68;
      v37 = &v52[88 * (unsigned int)v53];
      if ( v52 != (_BYTE *)v37 )
      {
        do
        {
          v37 -= 11;
          v38 = (_QWORD *)v37[4];
          if ( v38 != v37 + 6 )
            j_j___libc_free_0(v38, v37[6] + 1LL);
          if ( (_QWORD *)*v37 != v37 + 2 )
            j_j___libc_free_0(*v37, v37[2] + 1LL);
        }
        while ( v36 != v37 );
        v37 = v52;
      }
      if ( v37 != (_QWORD *)v54 )
        _libc_free((unsigned __int64)v37);
      return 0;
    }
    else
    {
      **(_DWORD **)(a1 + 456) = *(_DWORD *)(*(_QWORD *)(v8 + 8) + 280LL);
      v30 = *(_QWORD *)(a1 + 16);
      v31 = sub_1458800(**(_QWORD **)(a1 + 48));
      sub_1495190(v30, v31, a2, a3);
    }
  }
  return v29;
}
