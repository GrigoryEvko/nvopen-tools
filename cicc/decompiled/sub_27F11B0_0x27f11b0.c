// Function: sub_27F11B0
// Address: 0x27f11b0
//
void __fastcall sub_27F11B0(__int64 *a1, unsigned __int8 *a2)
{
  __int64 v3; // r14
  __int64 v4; // rax
  __int64 v5; // rdx
  __int64 v6; // rcx
  __int64 v7; // r8
  __int64 v8; // r9
  __int64 v9; // r12
  __m128i v10; // xmm0
  __m128i v11; // xmm1
  __m128i v12; // xmm2
  __int64 v13; // rax
  __int64 v14; // rax
  unsigned __int64 *v15; // r14
  unsigned __int64 *v16; // r12
  unsigned __int64 v17; // rdi
  unsigned __int64 *v18; // r13
  unsigned __int64 *v19; // r12
  unsigned __int64 v20; // rdi
  __int64 v21; // rax
  __int64 v22; // rax
  unsigned __int64 v23[2]; // [rsp+0h] [rbp-3E0h] BYREF
  __int64 v24; // [rsp+10h] [rbp-3D0h] BYREF
  __int64 *v25; // [rsp+20h] [rbp-3C0h]
  __int64 v26; // [rsp+30h] [rbp-3B0h] BYREF
  void *v27; // [rsp+50h] [rbp-390h] BYREF
  int v28; // [rsp+58h] [rbp-388h]
  char v29; // [rsp+5Ch] [rbp-384h]
  __int64 v30; // [rsp+60h] [rbp-380h]
  __m128i v31; // [rsp+68h] [rbp-378h]
  __int64 v32; // [rsp+78h] [rbp-368h]
  __m128i v33; // [rsp+80h] [rbp-360h]
  __m128i v34; // [rsp+90h] [rbp-350h]
  unsigned __int64 *v35; // [rsp+A0h] [rbp-340h] BYREF
  __int64 v36; // [rsp+A8h] [rbp-338h]
  _BYTE v37[324]; // [rsp+B0h] [rbp-330h] BYREF
  int v38; // [rsp+1F4h] [rbp-1ECh]
  __int64 v39; // [rsp+1F8h] [rbp-1E8h]
  _QWORD v40[10]; // [rsp+200h] [rbp-1E0h] BYREF
  unsigned __int64 *v41; // [rsp+250h] [rbp-190h]
  unsigned int v42; // [rsp+258h] [rbp-188h]
  char v43; // [rsp+260h] [rbp-180h] BYREF

  v3 = *a1;
  v4 = sub_B2BE50(*a1);
  if ( sub_B6EA50(v4)
    || (v21 = sub_B2BE50(v3),
        v22 = sub_B6F970(v21),
        (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v22 + 48LL))(v22)) )
  {
    sub_B174A0((__int64)v40, (__int64)"licm", (__int64)"Hoisted", 7, (__int64)a2);
    sub_B18290((__int64)v40, "hoisting ", 9u);
    sub_B16080((__int64)v23, "Inst", 4, a2);
    v9 = sub_23FD640((__int64)v40, (__int64)v23);
    v10 = _mm_loadu_si128((const __m128i *)(v9 + 24));
    v11 = _mm_loadu_si128((const __m128i *)(v9 + 48));
    v28 = *(_DWORD *)(v9 + 8);
    v12 = _mm_loadu_si128((const __m128i *)(v9 + 64));
    v29 = *(_BYTE *)(v9 + 12);
    v13 = *(_QWORD *)(v9 + 16);
    v31 = v10;
    v30 = v13;
    v27 = &unk_49D9D40;
    v14 = *(_QWORD *)(v9 + 40);
    v35 = (unsigned __int64 *)v37;
    v32 = v14;
    v36 = 0x400000000LL;
    LODWORD(v14) = *(_DWORD *)(v9 + 88);
    v33 = v11;
    v34 = v12;
    if ( (_DWORD)v14 )
      sub_27EFAF0((__int64)&v35, v9 + 80, v5, v6, v7, v8);
    v37[320] = *(_BYTE *)(v9 + 416);
    v38 = *(_DWORD *)(v9 + 420);
    v39 = *(_QWORD *)(v9 + 424);
    v27 = &unk_49D9D78;
    if ( v25 != &v26 )
      j_j___libc_free_0((unsigned __int64)v25);
    if ( (__int64 *)v23[0] != &v24 )
      j_j___libc_free_0(v23[0]);
    v15 = v41;
    v40[0] = &unk_49D9D40;
    v16 = &v41[10 * v42];
    if ( v41 != v16 )
    {
      do
      {
        v16 -= 10;
        v17 = v16[4];
        if ( (unsigned __int64 *)v17 != v16 + 6 )
          j_j___libc_free_0(v17);
        if ( (unsigned __int64 *)*v16 != v16 + 2 )
          j_j___libc_free_0(*v16);
      }
      while ( v15 != v16 );
      v16 = v41;
    }
    if ( v16 != (unsigned __int64 *)&v43 )
      _libc_free((unsigned __int64)v16);
    sub_1049740(a1, (__int64)&v27);
    v18 = v35;
    v27 = &unk_49D9D40;
    v19 = &v35[10 * (unsigned int)v36];
    if ( v35 != v19 )
    {
      do
      {
        v19 -= 10;
        v20 = v19[4];
        if ( (unsigned __int64 *)v20 != v19 + 6 )
          j_j___libc_free_0(v20);
        if ( (unsigned __int64 *)*v19 != v19 + 2 )
          j_j___libc_free_0(*v19);
      }
      while ( v18 != v19 );
      v19 = v35;
    }
    if ( v19 != (unsigned __int64 *)v37 )
      _libc_free((unsigned __int64)v19);
  }
}
