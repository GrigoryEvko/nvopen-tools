// Function: sub_2822A00
// Address: 0x2822a00
//
void __fastcall sub_2822A00(__int64 *a1, __int64 *a2)
{
  __int64 v3; // r12
  __int64 v4; // rax
  __int64 v5; // r12
  unsigned __int8 *v6; // rax
  __int64 v7; // r12
  __int64 v8; // rdx
  __int64 v9; // rcx
  __int64 v10; // r8
  __int64 v11; // r9
  __int64 v12; // r12
  __m128i v13; // xmm0
  __m128i v14; // xmm1
  __m128i v15; // xmm2
  __int64 v16; // rax
  __int64 v17; // rax
  unsigned __int64 *v18; // r14
  unsigned __int64 *v19; // r12
  unsigned __int64 v20; // rdi
  unsigned __int64 *v21; // r13
  unsigned __int64 *v22; // r12
  unsigned __int64 v23; // rdi
  __int64 v24; // rax
  __int64 v25; // rax
  unsigned __int64 v26[2]; // [rsp+0h] [rbp-480h] BYREF
  __int64 v27; // [rsp+10h] [rbp-470h] BYREF
  __int64 *v28; // [rsp+20h] [rbp-460h]
  __int64 v29; // [rsp+30h] [rbp-450h] BYREF
  unsigned __int64 v30[2]; // [rsp+50h] [rbp-430h] BYREF
  __int64 v31; // [rsp+60h] [rbp-420h] BYREF
  __int64 *v32; // [rsp+70h] [rbp-410h]
  __int64 v33; // [rsp+80h] [rbp-400h] BYREF
  unsigned __int64 v34[2]; // [rsp+A0h] [rbp-3E0h] BYREF
  __int64 v35; // [rsp+B0h] [rbp-3D0h] BYREF
  __int64 *v36; // [rsp+C0h] [rbp-3C0h]
  __int64 v37; // [rsp+D0h] [rbp-3B0h] BYREF
  void *v38; // [rsp+F0h] [rbp-390h] BYREF
  int v39; // [rsp+F8h] [rbp-388h]
  char v40; // [rsp+FCh] [rbp-384h]
  __int64 v41; // [rsp+100h] [rbp-380h]
  __m128i v42; // [rsp+108h] [rbp-378h]
  __int64 v43; // [rsp+118h] [rbp-368h]
  __m128i v44; // [rsp+120h] [rbp-360h]
  __m128i v45; // [rsp+130h] [rbp-350h]
  unsigned __int64 *v46; // [rsp+140h] [rbp-340h] BYREF
  __int64 v47; // [rsp+148h] [rbp-338h]
  _BYTE v48[324]; // [rsp+150h] [rbp-330h] BYREF
  int v49; // [rsp+294h] [rbp-1ECh]
  __int64 v50; // [rsp+298h] [rbp-1E8h]
  _QWORD v51[10]; // [rsp+2A0h] [rbp-1E0h] BYREF
  unsigned __int64 *v52; // [rsp+2F0h] [rbp-190h]
  unsigned int v53; // [rsp+2F8h] [rbp-188h]
  char v54; // [rsp+300h] [rbp-180h] BYREF

  v3 = *a1;
  v4 = sub_B2BE50(*a1);
  if ( sub_B6EA50(v4)
    || (v24 = sub_B2BE50(v3),
        v25 = sub_B6F970(v24),
        (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v25 + 48LL))(v25)) )
  {
    sub_B176B0((__int64)v51, (__int64)"loop-idiom", (__int64)"SizeStrideUnequal", 17, *a2);
    sub_B16430((__int64)v34, "Inst", 4u, "memcpy", 6);
    v5 = sub_2820EB0((__int64)v51, (__int64)v34);
    sub_B18290(v5, " in ", 4u);
    v6 = (unsigned __int8 *)sub_B43CB0(*a2);
    sub_B16080((__int64)v30, "Function", 8, v6);
    v7 = sub_2445430(v5, (__int64)v30);
    sub_B18290(v7, " function will not be hoisted: ", 0x1Fu);
    sub_B16430((__int64)v26, "Reason", 6u, "memcpy size is not equal to stride", 34);
    v12 = sub_2445430(v7, (__int64)v26);
    v13 = _mm_loadu_si128((const __m128i *)(v12 + 24));
    v14 = _mm_loadu_si128((const __m128i *)(v12 + 48));
    v39 = *(_DWORD *)(v12 + 8);
    v15 = _mm_loadu_si128((const __m128i *)(v12 + 64));
    v40 = *(_BYTE *)(v12 + 12);
    v16 = *(_QWORD *)(v12 + 16);
    v42 = v13;
    v41 = v16;
    v38 = &unk_49D9D40;
    v17 = *(_QWORD *)(v12 + 40);
    v46 = (unsigned __int64 *)v48;
    v43 = v17;
    v47 = 0x400000000LL;
    LODWORD(v17) = *(_DWORD *)(v12 + 88);
    v44 = v14;
    v45 = v15;
    if ( (_DWORD)v17 )
      sub_2822780((__int64)&v46, v12 + 80, v8, v9, v10, v11);
    v48[320] = *(_BYTE *)(v12 + 416);
    v49 = *(_DWORD *)(v12 + 420);
    v50 = *(_QWORD *)(v12 + 424);
    v38 = &unk_49D9DB0;
    if ( v28 != &v29 )
      j_j___libc_free_0((unsigned __int64)v28);
    if ( (__int64 *)v26[0] != &v27 )
      j_j___libc_free_0(v26[0]);
    if ( v32 != &v33 )
      j_j___libc_free_0((unsigned __int64)v32);
    if ( (__int64 *)v30[0] != &v31 )
      j_j___libc_free_0(v30[0]);
    if ( v36 != &v37 )
      j_j___libc_free_0((unsigned __int64)v36);
    if ( (__int64 *)v34[0] != &v35 )
      j_j___libc_free_0(v34[0]);
    v18 = v52;
    v51[0] = &unk_49D9D40;
    v19 = &v52[10 * v53];
    if ( v52 != v19 )
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
      v19 = v52;
    }
    if ( v19 != (unsigned __int64 *)&v54 )
      _libc_free((unsigned __int64)v19);
    sub_1049740(a1, (__int64)&v38);
    v21 = v46;
    v38 = &unk_49D9D40;
    v22 = &v46[10 * (unsigned int)v47];
    if ( v46 != v22 )
    {
      do
      {
        v22 -= 10;
        v23 = v22[4];
        if ( (unsigned __int64 *)v23 != v22 + 6 )
          j_j___libc_free_0(v23);
        if ( (unsigned __int64 *)*v22 != v22 + 2 )
          j_j___libc_free_0(*v22);
      }
      while ( v21 != v22 );
      v22 = v46;
    }
    if ( v22 != (unsigned __int64 *)v48 )
      _libc_free((unsigned __int64)v22);
  }
}
