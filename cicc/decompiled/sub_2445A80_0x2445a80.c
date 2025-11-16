// Function: sub_2445A80
// Address: 0x2445a80
//
void __fastcall sub_2445A80(
        __int64 *a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int64 a7,
        unsigned __int8 **a8,
        unsigned __int64 *a9,
        unsigned __int64 *a10)
{
  __int64 v11; // r12
  __int64 v12; // rax
  __int64 v13; // r12
  __int64 v14; // r12
  __int64 v15; // rdx
  __int64 v16; // rcx
  __int64 v17; // r8
  __int64 v18; // r9
  __int64 v19; // r12
  __m128i v20; // xmm0
  __m128i v21; // xmm1
  __m128i v22; // xmm2
  __int64 v23; // rax
  __int64 v24; // rax
  unsigned __int64 *v25; // r14
  unsigned __int64 *v26; // r12
  unsigned __int64 v27; // rdi
  unsigned __int64 *v28; // r13
  unsigned __int64 *v29; // r12
  unsigned __int64 v30; // rdi
  __int64 v31; // rax
  __int64 v32; // rax
  __int64 v33[2]; // [rsp+0h] [rbp-480h] BYREF
  __int64 v34; // [rsp+10h] [rbp-470h] BYREF
  __int64 *v35; // [rsp+20h] [rbp-460h]
  __int64 v36; // [rsp+30h] [rbp-450h] BYREF
  __int64 v37[2]; // [rsp+50h] [rbp-430h] BYREF
  __int64 v38; // [rsp+60h] [rbp-420h] BYREF
  __int64 *v39; // [rsp+70h] [rbp-410h]
  __int64 v40; // [rsp+80h] [rbp-400h] BYREF
  unsigned __int64 v41[2]; // [rsp+A0h] [rbp-3E0h] BYREF
  __int64 v42; // [rsp+B0h] [rbp-3D0h] BYREF
  __int64 *v43; // [rsp+C0h] [rbp-3C0h]
  __int64 v44; // [rsp+D0h] [rbp-3B0h] BYREF
  void *v45; // [rsp+F0h] [rbp-390h] BYREF
  int v46; // [rsp+F8h] [rbp-388h]
  char v47; // [rsp+FCh] [rbp-384h]
  __int64 v48; // [rsp+100h] [rbp-380h]
  __m128i v49; // [rsp+108h] [rbp-378h]
  __int64 v50; // [rsp+118h] [rbp-368h]
  __m128i v51; // [rsp+120h] [rbp-360h]
  __m128i v52; // [rsp+130h] [rbp-350h]
  unsigned __int64 *v53; // [rsp+140h] [rbp-340h] BYREF
  __int64 v54; // [rsp+148h] [rbp-338h]
  _BYTE v55[324]; // [rsp+150h] [rbp-330h] BYREF
  int v56; // [rsp+294h] [rbp-1ECh]
  __int64 v57; // [rsp+298h] [rbp-1E8h]
  _QWORD v58[10]; // [rsp+2A0h] [rbp-1E0h] BYREF
  unsigned __int64 *v59; // [rsp+2F0h] [rbp-190h]
  unsigned int v60; // [rsp+2F8h] [rbp-188h]
  char v61; // [rsp+300h] [rbp-180h] BYREF

  v11 = *a1;
  v12 = sub_B2BE50(*a1);
  if ( sub_B6EA50(v12)
    || (v31 = sub_B2BE50(v11),
        v32 = sub_B6F970(v31),
        (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v32 + 48LL))(v32)) )
  {
    sub_B174A0((__int64)v58, (__int64)"pgo-icall-prom", (__int64)"Promoted", 8, a7);
    sub_B18290((__int64)v58, "Promote indirect call to ", 0x19u);
    sub_B16080((__int64)v41, "DirectCallee", 12, *a8);
    v13 = sub_23FD640((__int64)v58, (__int64)v41);
    sub_B18290(v13, " with count ", 0xCu);
    sub_B16B10(v37, "Count", 5, *a9);
    v14 = sub_23FD640(v13, (__int64)v37);
    sub_B18290(v14, " out of ", 8u);
    sub_B16B10(v33, "TotalCount", 10, *a10);
    v19 = sub_23FD640(v14, (__int64)v33);
    v20 = _mm_loadu_si128((const __m128i *)(v19 + 24));
    v21 = _mm_loadu_si128((const __m128i *)(v19 + 48));
    v46 = *(_DWORD *)(v19 + 8);
    v22 = _mm_loadu_si128((const __m128i *)(v19 + 64));
    v47 = *(_BYTE *)(v19 + 12);
    v23 = *(_QWORD *)(v19 + 16);
    v49 = v20;
    v48 = v23;
    v45 = &unk_49D9D40;
    v24 = *(_QWORD *)(v19 + 40);
    v53 = (unsigned __int64 *)v55;
    v50 = v24;
    v54 = 0x400000000LL;
    LODWORD(v24) = *(_DWORD *)(v19 + 88);
    v51 = v21;
    v52 = v22;
    if ( (_DWORD)v24 )
      sub_2445800((__int64)&v53, v19 + 80, v15, v16, v17, v18);
    v55[320] = *(_BYTE *)(v19 + 416);
    v56 = *(_DWORD *)(v19 + 420);
    v57 = *(_QWORD *)(v19 + 424);
    v45 = &unk_49D9D78;
    if ( v35 != &v36 )
      j_j___libc_free_0((unsigned __int64)v35);
    if ( (__int64 *)v33[0] != &v34 )
      j_j___libc_free_0(v33[0]);
    if ( v39 != &v40 )
      j_j___libc_free_0((unsigned __int64)v39);
    if ( (__int64 *)v37[0] != &v38 )
      j_j___libc_free_0(v37[0]);
    if ( v43 != &v44 )
      j_j___libc_free_0((unsigned __int64)v43);
    if ( (__int64 *)v41[0] != &v42 )
      j_j___libc_free_0(v41[0]);
    v25 = v59;
    v58[0] = &unk_49D9D40;
    v26 = &v59[10 * v60];
    if ( v59 != v26 )
    {
      do
      {
        v26 -= 10;
        v27 = v26[4];
        if ( (unsigned __int64 *)v27 != v26 + 6 )
          j_j___libc_free_0(v27);
        if ( (unsigned __int64 *)*v26 != v26 + 2 )
          j_j___libc_free_0(*v26);
      }
      while ( v25 != v26 );
      v26 = v59;
    }
    if ( v26 != (unsigned __int64 *)&v61 )
      _libc_free((unsigned __int64)v26);
    sub_1049740(a1, (__int64)&v45);
    v28 = v53;
    v45 = &unk_49D9D40;
    v29 = &v53[10 * (unsigned int)v54];
    if ( v53 != v29 )
    {
      do
      {
        v29 -= 10;
        v30 = v29[4];
        if ( (unsigned __int64 *)v30 != v29 + 6 )
          j_j___libc_free_0(v30);
        if ( (unsigned __int64 *)*v29 != v29 + 2 )
          j_j___libc_free_0(*v29);
      }
      while ( v28 != v29 );
      v29 = v53;
    }
    if ( v29 != (unsigned __int64 *)v55 )
      _libc_free((unsigned __int64)v29);
  }
}
