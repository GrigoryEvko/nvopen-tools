// Function: sub_289EB20
// Address: 0x289eb20
//
void __fastcall sub_289EB20(__int64 a1, __int64 a2)
{
  __int64 v2; // r12
  __int64 *v3; // r14
  __int64 v4; // r8
  __int64 v5; // rcx
  __int64 v6; // r8
  int v7; // ebx
  bool v8; // al
  int v9; // r13d
  int v10; // r12d
  int v11; // r15d
  int v12; // ebx
  int v13; // esi
  __int64 **v14; // rax
  __int64 v15; // rax
  __int64 v16; // r8
  __int64 v17; // rdx
  unsigned __int64 v18; // r9
  int v19; // eax
  __int64 v20; // r8
  __int64 v21; // r9
  __m128i v22; // xmm0
  __int64 v23; // [rsp-8h] [rbp-408h]
  __int64 v24; // [rsp+8h] [rbp-3F8h]
  int v26[3]; // [rsp+68h] [rbp-398h] BYREF
  _BYTE v27[4]; // [rsp+74h] [rbp-38Ch] BYREF
  int v28; // [rsp+78h] [rbp-388h]
  unsigned int *v29[2]; // [rsp+80h] [rbp-380h] BYREF
  _BYTE v30[32]; // [rsp+90h] [rbp-370h] BYREF
  __int64 v31; // [rsp+B0h] [rbp-350h]
  __int64 v32; // [rsp+B8h] [rbp-348h]
  __int16 v33; // [rsp+C0h] [rbp-340h]
  __int64 v34; // [rsp+C8h] [rbp-338h]
  void **v35; // [rsp+D0h] [rbp-330h]
  void **v36; // [rsp+D8h] [rbp-328h]
  __int64 v37; // [rsp+E0h] [rbp-320h]
  int v38; // [rsp+E8h] [rbp-318h]
  __int16 v39; // [rsp+ECh] [rbp-314h]
  char v40; // [rsp+EEh] [rbp-312h]
  __int64 v41; // [rsp+F0h] [rbp-310h]
  __int64 v42; // [rsp+F8h] [rbp-308h]
  void *v43; // [rsp+100h] [rbp-300h] BYREF
  void *v44; // [rsp+108h] [rbp-2F8h] BYREF
  __int64 v45[2]; // [rsp+110h] [rbp-2F0h] BYREF
  char v46; // [rsp+120h] [rbp-2E0h] BYREF
  __int64 v47[2]; // [rsp+1C0h] [rbp-240h] BYREF
  char v48; // [rsp+1D0h] [rbp-230h] BYREF
  _BYTE *v49; // [rsp+270h] [rbp-190h] BYREF
  __int64 v50; // [rsp+278h] [rbp-188h]
  _BYTE v51[128]; // [rsp+280h] [rbp-180h] BYREF
  __m128i v52; // [rsp+300h] [rbp-100h] BYREF
  bool v53; // [rsp+310h] [rbp-F0h]
  __m128i v54; // [rsp+320h] [rbp-E0h] BYREF
  _BYTE v55[128]; // [rsp+330h] [rbp-D0h] BYREF
  __m128i v56; // [rsp+3B0h] [rbp-50h]
  bool v57; // [rsp+3C0h] [rbp-40h]

  v2 = a2;
  v34 = sub_BD5C60(a2);
  v35 = &v43;
  v36 = &v44;
  v39 = 512;
  v33 = 0;
  v43 = &unk_49DA100;
  v29[0] = (unsigned int *)v30;
  v29[1] = (unsigned int *)0x200000000LL;
  v44 = &unk_49DA0B0;
  v37 = 0;
  v38 = 0;
  v40 = 7;
  v41 = 0;
  v42 = 0;
  v31 = 0;
  v32 = 0;
  sub_D5F1F0((__int64)v29, a2);
  v3 = *(__int64 **)(*(_QWORD *)(a2 + 8) + 24LL);
  sub_28940A0(
    (__int64)v26,
    *(_QWORD *)(v2 + 32 * (2LL - (*(_DWORD *)(v2 + 4) & 0x7FFFFFF))),
    *(_QWORD *)(v2 + 32 * (3LL - (*(_DWORD *)(v2 + 4) & 0x7FFFFFF))));
  sub_28940A0(
    (__int64)v27,
    *(_QWORD *)(v2 + 32 * (v4 - (*(_DWORD *)(v2 + 4) & 0x7FFFFFF))),
    *(_QWORD *)(v2 + 32 * (4LL - (*(_DWORD *)(v2 + 4) & 0x7FFFFFF))));
  sub_2895860((__int64)v45, a1, *(_QWORD *)(v2 - 32LL * (*(_DWORD *)(v2 + 4) & 0x7FFFFFF)), v5, v6);
  sub_2895860(
    (__int64)v47,
    a1,
    *(_QWORD *)(v2 + 32 * (1LL - (*(_DWORD *)(v2 + 4) & 0x7FFFFFF))),
    (__int64)v27,
    (__int64)v29);
  v7 = v28;
  v49 = v51;
  if ( dword_5003CC8 )
    v7 = v26[0];
  v50 = 0x1000000000LL;
  v8 = dword_5003CC8 == 0;
  v52 = 0u;
  v53 = dword_5003CC8 == 0;
  if ( v7 )
  {
    v9 = v7;
    v10 = v26[0];
    v11 = 0;
    v12 = v28;
    while ( 1 )
    {
      v13 = v12;
      if ( v8 )
        v13 = v10;
      v14 = (__int64 **)sub_BCDA70(v3, v13);
      v15 = sub_ACADE0(v14);
      v17 = (unsigned int)v50;
      v18 = (unsigned int)v50 + 1LL;
      if ( v18 > HIDWORD(v50) )
      {
        v24 = v15;
        sub_C8D5F0((__int64)&v49, v51, (unsigned int)v50 + 1LL, 8u, v16, v18);
        v17 = (unsigned int)v50;
        v15 = v24;
      }
      ++v11;
      *(_QWORD *)&v49[8 * v17] = v15;
      LODWORD(v50) = v50 + 1;
      if ( v9 == v11 )
        break;
      v8 = v53;
    }
    v2 = a2;
  }
  v19 = sub_28956B0(v2);
  sub_2899430(a1, (__int64 *)&v49, v45, v47, (__int64)v29, 0, 0, v19);
  v54.m128i_i64[0] = (__int64)v55;
  v54.m128i_i64[1] = 0x1000000000LL;
  if ( (_DWORD)v50 )
    sub_2894AD0((__int64)&v54, (__int64)&v49, v23, (unsigned int)v50, v20, v21);
  v22 = _mm_loadu_si128(&v52);
  v57 = v53;
  v56 = v22;
  sub_289E450(a1, v2, &v54, v29, v20, v21);
  if ( (_BYTE *)v54.m128i_i64[0] != v55 )
    _libc_free(v54.m128i_u64[0]);
  if ( v49 != v51 )
    _libc_free((unsigned __int64)v49);
  if ( (char *)v47[0] != &v48 )
    _libc_free(v47[0]);
  if ( (char *)v45[0] != &v46 )
    _libc_free(v45[0]);
  nullsub_61();
  v43 = &unk_49DA100;
  nullsub_63();
  if ( (_BYTE *)v29[0] != v30 )
    _libc_free((unsigned __int64)v29[0]);
}
