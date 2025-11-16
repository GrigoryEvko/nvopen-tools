// Function: sub_2FCBF60
// Address: 0x2fcbf60
//
void __fastcall sub_2FCBF60(__int64 *a1, __int64 a2, unsigned __int8 **a3)
{
  __int64 v5; // r12
  __int64 v6; // rax
  __int64 v7; // r12
  __int64 v8; // rdx
  __int64 v9; // rcx
  __int64 v10; // r8
  __int64 v11; // r9
  __m128i v12; // xmm0
  __m128i v13; // xmm1
  __m128i v14; // xmm2
  __int64 v15; // rax
  __int64 v16; // rax
  unsigned __int64 *v17; // r14
  unsigned __int64 *v18; // r12
  unsigned __int64 v19; // rdi
  unsigned __int64 *v20; // r13
  unsigned __int64 *v21; // r12
  unsigned __int64 v22; // rdi
  __int64 v23; // rax
  __int64 v24; // rax
  unsigned __int64 v25[2]; // [rsp+0h] [rbp-3E0h] BYREF
  __int64 v26; // [rsp+10h] [rbp-3D0h] BYREF
  __int64 *v27; // [rsp+20h] [rbp-3C0h]
  __int64 v28; // [rsp+30h] [rbp-3B0h] BYREF
  void *v29; // [rsp+50h] [rbp-390h] BYREF
  int v30; // [rsp+58h] [rbp-388h]
  char v31; // [rsp+5Ch] [rbp-384h]
  __int64 v32; // [rsp+60h] [rbp-380h]
  __m128i v33; // [rsp+68h] [rbp-378h]
  __int64 v34; // [rsp+78h] [rbp-368h]
  __m128i v35; // [rsp+80h] [rbp-360h]
  __m128i v36; // [rsp+90h] [rbp-350h]
  unsigned __int64 *v37; // [rsp+A0h] [rbp-340h] BYREF
  __int64 v38; // [rsp+A8h] [rbp-338h]
  _BYTE v39[324]; // [rsp+B0h] [rbp-330h] BYREF
  int v40; // [rsp+1F4h] [rbp-1ECh]
  __int64 v41; // [rsp+1F8h] [rbp-1E8h]
  _QWORD v42[10]; // [rsp+200h] [rbp-1E0h] BYREF
  unsigned __int64 *v43; // [rsp+250h] [rbp-190h]
  unsigned int v44; // [rsp+258h] [rbp-188h]
  char v45; // [rsp+260h] [rbp-180h] BYREF

  v5 = *a1;
  v6 = sub_B2BE50(*a1);
  if ( sub_B6EA50(v6)
    || (v23 = sub_B2BE50(v5),
        v24 = sub_B6F970(v23),
        (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v24 + 48LL))(v24)) )
  {
    sub_B174A0((__int64)v42, (__int64)"stack-protector", (__int64)"StackProtectorAllocaOrArray", 27, a2);
    sub_B18290((__int64)v42, "Stack protection applied to function ", 0x25u);
    sub_B16080((__int64)v25, "Function", 8, *a3);
    v7 = sub_23FD640((__int64)v42, (__int64)v25);
    sub_B18290(v7, " due to a call to alloca or use of a variable length array", 0x3Au);
    v12 = _mm_loadu_si128((const __m128i *)(v7 + 24));
    v13 = _mm_loadu_si128((const __m128i *)(v7 + 48));
    v30 = *(_DWORD *)(v7 + 8);
    v14 = _mm_loadu_si128((const __m128i *)(v7 + 64));
    v31 = *(_BYTE *)(v7 + 12);
    v15 = *(_QWORD *)(v7 + 16);
    v33 = v12;
    v32 = v15;
    v29 = &unk_49D9D40;
    v16 = *(_QWORD *)(v7 + 40);
    v37 = (unsigned __int64 *)v39;
    v34 = v16;
    v38 = 0x400000000LL;
    LODWORD(v16) = *(_DWORD *)(v7 + 88);
    v35 = v13;
    v36 = v14;
    if ( (_DWORD)v16 )
      sub_2FCBCE0((__int64)&v37, v7 + 80, v8, v9, v10, v11);
    v39[320] = *(_BYTE *)(v7 + 416);
    v40 = *(_DWORD *)(v7 + 420);
    v41 = *(_QWORD *)(v7 + 424);
    v29 = &unk_49D9D78;
    if ( v27 != &v28 )
      j_j___libc_free_0((unsigned __int64)v27);
    if ( (__int64 *)v25[0] != &v26 )
      j_j___libc_free_0(v25[0]);
    v17 = v43;
    v42[0] = &unk_49D9D40;
    v18 = &v43[10 * v44];
    if ( v43 != v18 )
    {
      do
      {
        v18 -= 10;
        v19 = v18[4];
        if ( (unsigned __int64 *)v19 != v18 + 6 )
          j_j___libc_free_0(v19);
        if ( (unsigned __int64 *)*v18 != v18 + 2 )
          j_j___libc_free_0(*v18);
      }
      while ( v17 != v18 );
      v18 = v43;
    }
    if ( v18 != (unsigned __int64 *)&v45 )
      _libc_free((unsigned __int64)v18);
    sub_1049740(a1, (__int64)&v29);
    v20 = v37;
    v29 = &unk_49D9D40;
    v21 = &v37[10 * (unsigned int)v38];
    if ( v37 != v21 )
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
      v21 = v37;
    }
    if ( v21 != (unsigned __int64 *)v39 )
      _libc_free((unsigned __int64)v21);
  }
}
