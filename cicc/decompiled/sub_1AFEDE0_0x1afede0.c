// Function: sub_1AFEDE0
// Address: 0x1afede0
//
void __fastcall sub_1AFEDE0(__int64 *a1, __int64 *a2, unsigned int *a3)
{
  __int64 v5; // rax
  __int64 v6; // rsi
  __int64 v7; // r12
  __m128i v8; // xmm0
  __int64 v9; // rax
  __m128i v10; // xmm1
  __int64 v11; // rax
  int v12; // eax
  char *v13; // r12
  char *v14; // rbx
  char *v15; // rdi
  _QWORD *v16; // rbx
  _QWORD *v17; // r12
  _QWORD *v18; // rdi
  __int64 v19; // rax
  __int64 v20; // rax
  __int64 v21; // [rsp+8h] [rbp-488h]
  __int64 v22; // [rsp+18h] [rbp-478h] BYREF
  __m128i v23[2]; // [rsp+20h] [rbp-470h] BYREF
  _QWORD v24[2]; // [rsp+40h] [rbp-450h] BYREF
  __int64 v25; // [rsp+50h] [rbp-440h] BYREF
  __int64 *v26; // [rsp+60h] [rbp-430h]
  __int64 v27; // [rsp+70h] [rbp-420h] BYREF
  void *v28; // [rsp+A0h] [rbp-3F0h] BYREF
  int v29; // [rsp+A8h] [rbp-3E8h]
  char v30; // [rsp+ACh] [rbp-3E4h]
  __int64 v31; // [rsp+B0h] [rbp-3E0h]
  __m128i v32; // [rsp+B8h] [rbp-3D8h]
  __int64 v33; // [rsp+C8h] [rbp-3C8h]
  __int64 v34; // [rsp+D0h] [rbp-3C0h]
  __m128i v35; // [rsp+D8h] [rbp-3B8h]
  __int64 v36; // [rsp+E8h] [rbp-3A8h]
  _BYTE *v38; // [rsp+F8h] [rbp-398h] BYREF
  __int64 v39; // [rsp+100h] [rbp-390h]
  _BYTE v40[356]; // [rsp+108h] [rbp-388h] BYREF
  int v41; // [rsp+26Ch] [rbp-224h]
  __int64 v42; // [rsp+270h] [rbp-220h]
  _QWORD v43[11]; // [rsp+280h] [rbp-210h] BYREF
  char *v44; // [rsp+2D8h] [rbp-1B8h]
  unsigned int v45; // [rsp+2E0h] [rbp-1B0h]
  char v46; // [rsp+2E8h] [rbp-1A8h] BYREF

  v5 = sub_15E0530(*a1);
  if ( sub_1602790(v5)
    || (v19 = sub_15E0530(*a1),
        v20 = sub_16033E0(v19),
        (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v20 + 48LL))(v20)) )
  {
    v6 = *a2;
    v21 = **(_QWORD **)(v6 + 32);
    sub_13FD840(&v22, v6);
    sub_15C9090((__int64)v23, &v22);
    sub_15CA330((__int64)v43, (__int64)"loop-unroll", (__int64)"FullyUnrolled", 13, v23, v21);
    sub_15CAB20((__int64)v43, "completely unrolled loop with ", 0x1Eu);
    sub_15C9C50((__int64)v24, "UnrollCount", 11, *a3);
    v7 = sub_17C2270((__int64)v43, (__int64)v24);
    sub_15CAB20(v7, " iterations", 0xBu);
    v8 = _mm_loadu_si128((const __m128i *)(v7 + 24));
    v29 = *(_DWORD *)(v7 + 8);
    v30 = *(_BYTE *)(v7 + 12);
    v9 = *(_QWORD *)(v7 + 16);
    v32 = v8;
    v31 = v9;
    v33 = *(_QWORD *)(v7 + 40);
    v10 = _mm_loadu_si128((const __m128i *)(v7 + 56));
    v28 = &unk_49ECF68;
    v11 = *(_QWORD *)(v7 + 48);
    v35 = v10;
    v34 = v11;
    if ( *(_BYTE *)(v7 + 80) )
      v36 = *(_QWORD *)(v7 + 72);
    v39 = 0x400000000LL;
    v12 = *(_DWORD *)(v7 + 96);
    v38 = v40;
    if ( v12 )
      sub_1AFDB00((__int64)&v38, v7 + 88);
    v40[352] = *(_BYTE *)(v7 + 456);
    v41 = *(_DWORD *)(v7 + 460);
    v42 = *(_QWORD *)(v7 + 464);
    v28 = &unk_49ECF98;
    if ( v26 != &v27 )
      j_j___libc_free_0(v26, v27 + 1);
    if ( (__int64 *)v24[0] != &v25 )
      j_j___libc_free_0(v24[0], v25 + 1);
    v13 = v44;
    v43[0] = &unk_49ECF68;
    v14 = &v44[88 * v45];
    if ( v44 != v14 )
    {
      do
      {
        v14 -= 88;
        v15 = (char *)*((_QWORD *)v14 + 4);
        if ( v15 != v14 + 48 )
          j_j___libc_free_0(v15, *((_QWORD *)v14 + 6) + 1LL);
        if ( *(char **)v14 != v14 + 16 )
          j_j___libc_free_0(*(_QWORD *)v14, *((_QWORD *)v14 + 2) + 1LL);
      }
      while ( v13 != v14 );
      v14 = v44;
    }
    if ( v14 != &v46 )
      _libc_free((unsigned __int64)v14);
    if ( v22 )
      sub_161E7C0((__int64)&v22, v22);
    sub_143AA50(a1, (__int64)&v28);
    v16 = v38;
    v28 = &unk_49ECF68;
    v17 = &v38[88 * (unsigned int)v39];
    if ( v38 != (_BYTE *)v17 )
    {
      do
      {
        v17 -= 11;
        v18 = (_QWORD *)v17[4];
        if ( v18 != v17 + 6 )
          j_j___libc_free_0(v18, v17[6] + 1LL);
        if ( (_QWORD *)*v17 != v17 + 2 )
          j_j___libc_free_0(*v17, v17[2] + 1LL);
      }
      while ( v16 != v17 );
      v17 = v38;
    }
    if ( v17 != (_QWORD *)v40 )
      _libc_free((unsigned __int64)v17);
  }
}
