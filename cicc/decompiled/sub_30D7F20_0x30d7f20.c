// Function: sub_30D7F20
// Address: 0x30d7f20
//
void __fastcall sub_30D7F20(__int64 *a1, __int64 a2, char **a3)
{
  __int64 v5; // r14
  __int64 v6; // rax
  __int64 v7; // r14
  char *v8; // r12
  size_t v9; // r8
  __int64 v10; // r12
  __int64 v11; // rdx
  __int64 v12; // rcx
  __int64 v13; // r8
  __int64 v14; // r9
  __m128i v15; // xmm0
  __m128i v16; // xmm1
  __m128i v17; // xmm2
  __int64 v18; // rax
  __int64 v19; // rax
  unsigned __int64 *v20; // r14
  unsigned __int64 *v21; // r12
  unsigned __int64 v22; // rdi
  unsigned __int64 *v23; // r13
  unsigned __int64 *v24; // r12
  unsigned __int64 v25; // rdi
  __int64 v26; // rax
  __int64 v27; // rax
  unsigned __int64 v28[2]; // [rsp+0h] [rbp-430h] BYREF
  __int64 v29; // [rsp+10h] [rbp-420h] BYREF
  __int64 *v30; // [rsp+20h] [rbp-410h]
  __int64 v31; // [rsp+30h] [rbp-400h] BYREF
  unsigned __int64 v32[2]; // [rsp+50h] [rbp-3E0h] BYREF
  __int64 v33; // [rsp+60h] [rbp-3D0h] BYREF
  __int64 *v34; // [rsp+70h] [rbp-3C0h]
  __int64 v35; // [rsp+80h] [rbp-3B0h] BYREF
  void *v36; // [rsp+A0h] [rbp-390h] BYREF
  int v37; // [rsp+A8h] [rbp-388h]
  char v38; // [rsp+ACh] [rbp-384h]
  __int64 v39; // [rsp+B0h] [rbp-380h]
  __m128i v40; // [rsp+B8h] [rbp-378h]
  __int64 v41; // [rsp+C8h] [rbp-368h]
  __m128i v42; // [rsp+D0h] [rbp-360h]
  __m128i v43; // [rsp+E0h] [rbp-350h]
  unsigned __int64 *v44; // [rsp+F0h] [rbp-340h] BYREF
  __int64 v45; // [rsp+F8h] [rbp-338h]
  _BYTE v46[324]; // [rsp+100h] [rbp-330h] BYREF
  int v47; // [rsp+244h] [rbp-1ECh]
  __int64 v48; // [rsp+248h] [rbp-1E8h]
  _QWORD v49[10]; // [rsp+250h] [rbp-1E0h] BYREF
  unsigned __int64 *v50; // [rsp+2A0h] [rbp-190h]
  unsigned int v51; // [rsp+2A8h] [rbp-188h]
  char v52; // [rsp+2B0h] [rbp-180h] BYREF

  v5 = *a1;
  v6 = sub_B2BE50(*a1);
  if ( sub_B6EA50(v6)
    || (v26 = sub_B2BE50(v5),
        v27 = sub_B6F970(v26),
        (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v27 + 48LL))(v27)) )
  {
    sub_B176B0((__int64)v49, (__int64)"inline-cost", (__int64)"NeverInline", 11, *(_QWORD *)(a2 + 96));
    sub_B16080((__int64)v32, "Callee", 6, *(unsigned __int8 **)(a2 + 72));
    v7 = sub_2820EB0((__int64)v49, (__int64)v32);
    sub_B18290(v7, " is ", 4u);
    v8 = *a3;
    v9 = 0;
    if ( v8 )
      v9 = strlen(v8);
    sub_B16430((__int64)v28, "InlineResult", 0xCu, v8, v9);
    v10 = sub_2445430(v7, (__int64)v28);
    sub_B18290(v10, ". Cost is not fully computed", 0x1Cu);
    v15 = _mm_loadu_si128((const __m128i *)(v10 + 24));
    v16 = _mm_loadu_si128((const __m128i *)(v10 + 48));
    v37 = *(_DWORD *)(v10 + 8);
    v17 = _mm_loadu_si128((const __m128i *)(v10 + 64));
    v38 = *(_BYTE *)(v10 + 12);
    v18 = *(_QWORD *)(v10 + 16);
    v40 = v15;
    v39 = v18;
    v36 = &unk_49D9D40;
    v19 = *(_QWORD *)(v10 + 40);
    v44 = (unsigned __int64 *)v46;
    v41 = v19;
    v45 = 0x400000000LL;
    LODWORD(v19) = *(_DWORD *)(v10 + 88);
    v42 = v16;
    v43 = v17;
    if ( (_DWORD)v19 )
      sub_30D78E0((__int64)&v44, v10 + 80, v11, v12, v13, v14);
    v46[320] = *(_BYTE *)(v10 + 416);
    v47 = *(_DWORD *)(v10 + 420);
    v48 = *(_QWORD *)(v10 + 424);
    v36 = &unk_49D9DB0;
    if ( v30 != &v31 )
      j_j___libc_free_0((unsigned __int64)v30);
    if ( (__int64 *)v28[0] != &v29 )
      j_j___libc_free_0(v28[0]);
    if ( v34 != &v35 )
      j_j___libc_free_0((unsigned __int64)v34);
    if ( (__int64 *)v32[0] != &v33 )
      j_j___libc_free_0(v32[0]);
    v20 = v50;
    v49[0] = &unk_49D9D40;
    v21 = &v50[10 * v51];
    if ( v50 != v21 )
    {
      do
      {
        v21 -= 10;
        v22 = v21[4];
        if ( (unsigned __int64 *)v22 != v21 + 6 )
          j_j___libc_free_0(v22);
        if ( (unsigned __int64 *)*v21 != v21 + 2 )
          j_j___libc_free_0(*v21);
      }
      while ( v20 != v21 );
      v21 = v50;
    }
    if ( v21 != (unsigned __int64 *)&v52 )
      _libc_free((unsigned __int64)v21);
    sub_1049740(a1, (__int64)&v36);
    v23 = v44;
    v36 = &unk_49D9D40;
    v24 = &v44[10 * (unsigned int)v45];
    if ( v44 != v24 )
    {
      do
      {
        v24 -= 10;
        v25 = v24[4];
        if ( (unsigned __int64 *)v25 != v24 + 6 )
          j_j___libc_free_0(v25);
        if ( (unsigned __int64 *)*v24 != v24 + 2 )
          j_j___libc_free_0(*v24);
      }
      while ( v23 != v24 );
      v24 = v44;
    }
    if ( v24 != (unsigned __int64 *)v46 )
      _libc_free((unsigned __int64)v24);
  }
}
