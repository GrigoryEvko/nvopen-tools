// Function: sub_2AC2B40
// Address: 0x2ac2b40
//
void __fastcall sub_2AC2B40(__int64 *a1, __int64 a2, unsigned int a3, __int64 a4, __int64 a5, __int64 a6, __int64 a7)
{
  size_t v7; // r14
  __int8 *v8; // r13
  __int64 v10; // r15
  __int64 v11; // rax
  __int64 v12; // rdx
  __int64 v13; // rcx
  __int64 v14; // r8
  __int64 v15; // r13
  __int64 v16; // rbx
  __int64 v17; // rdx
  __int64 v18; // rcx
  __int64 v19; // r8
  __int64 v20; // r9
  __m128i v21; // xmm0
  __m128i v22; // xmm1
  __m128i v23; // xmm2
  __int64 v24; // rax
  __int64 v25; // rax
  __int64 v26; // rax
  __int64 v27; // rax
  __int64 v28; // [rsp+10h] [rbp-460h]
  __int64 v29; // [rsp+28h] [rbp-448h] BYREF
  __m128i v30; // [rsp+30h] [rbp-440h] BYREF
  __int64 v31[2]; // [rsp+40h] [rbp-430h] BYREF
  __int64 v32; // [rsp+50h] [rbp-420h] BYREF
  __int64 *v33; // [rsp+60h] [rbp-410h]
  __int64 v34; // [rsp+70h] [rbp-400h] BYREF
  unsigned __int64 v35[2]; // [rsp+90h] [rbp-3E0h] BYREF
  __int64 v36; // [rsp+A0h] [rbp-3D0h] BYREF
  __int64 *v37; // [rsp+B0h] [rbp-3C0h]
  __int64 v38; // [rsp+C0h] [rbp-3B0h] BYREF
  void *v39; // [rsp+E0h] [rbp-390h] BYREF
  int v40; // [rsp+E8h] [rbp-388h]
  char v41; // [rsp+ECh] [rbp-384h]
  __int64 v42; // [rsp+F0h] [rbp-380h]
  __m128i v43; // [rsp+F8h] [rbp-378h]
  __int64 v44; // [rsp+108h] [rbp-368h]
  __m128i v45; // [rsp+110h] [rbp-360h]
  __m128i v46; // [rsp+120h] [rbp-350h]
  _QWORD v47[2]; // [rsp+130h] [rbp-340h] BYREF
  _BYTE v48[324]; // [rsp+140h] [rbp-330h] BYREF
  int v49; // [rsp+284h] [rbp-1ECh]
  __int64 v50; // [rsp+288h] [rbp-1E8h]
  _QWORD v51[10]; // [rsp+290h] [rbp-1E0h] BYREF
  _BYTE v52[400]; // [rsp+2E0h] [rbp-190h] BYREF

  v7 = 0;
  v8 = (__int8 *)byte_3F871B3;
  v10 = *a1;
  if ( *(_QWORD *)(a2 + 16) != *(_QWORD *)(a2 + 8) )
  {
    v7 = 6;
    v8 = "outer ";
  }
  v11 = sub_B2BE50(v10);
  if ( sub_B6EA50(v11)
    || (v26 = sub_B2BE50(v10),
        v27 = sub_B6F970(v26),
        (*(unsigned __int8 (__fastcall **)(__int64, __int64))(*(_QWORD *)v27 + 48LL))(v27, a2)) )
  {
    v28 = **(_QWORD **)(a2 + 32);
    sub_D4BD20(&v29, a2, v12, v13, v14, v28);
    sub_B157E0((__int64)&v30, &v29);
    sub_B17430((__int64)v51, (__int64)"loop-vectorize", (__int64)"Vectorized", 10, &v30, v28);
    sub_B18290((__int64)v51, "vectorized ", 0xBu);
    sub_B18290((__int64)v51, v8, v7);
    sub_B18290((__int64)v51, "loop (vectorization width: ", 0x1Bu);
    sub_B16C30((__int64)v35, "VectorizationFactor", 19, a7);
    v15 = sub_23FD640((__int64)v51, (__int64)v35);
    sub_B18290(v15, ", interleaved count: ", 0x15u);
    sub_B169E0(v31, "InterleaveCount", 15, a3);
    v16 = sub_23FD640(v15, (__int64)v31);
    sub_B18290(v16, ")", 1u);
    v21 = _mm_loadu_si128((const __m128i *)(v16 + 24));
    v22 = _mm_loadu_si128((const __m128i *)(v16 + 48));
    v40 = *(_DWORD *)(v16 + 8);
    v23 = _mm_loadu_si128((const __m128i *)(v16 + 64));
    v41 = *(_BYTE *)(v16 + 12);
    v24 = *(_QWORD *)(v16 + 16);
    v43 = v21;
    v42 = v24;
    v39 = &unk_49D9D40;
    v25 = *(_QWORD *)(v16 + 40);
    v45 = v22;
    v44 = v25;
    v47[0] = v48;
    v47[1] = 0x400000000LL;
    LODWORD(v25) = *(_DWORD *)(v16 + 88);
    v46 = v23;
    if ( (_DWORD)v25 )
      sub_2AC2270((__int64)v47, v16 + 80, v17, v18, v19, v20);
    v48[320] = *(_BYTE *)(v16 + 416);
    v49 = *(_DWORD *)(v16 + 420);
    v50 = *(_QWORD *)(v16 + 424);
    v39 = &unk_49D9D78;
    if ( v33 != &v34 )
      j_j___libc_free_0((unsigned __int64)v33);
    if ( (__int64 *)v31[0] != &v32 )
      j_j___libc_free_0(v31[0]);
    if ( v37 != &v38 )
      j_j___libc_free_0((unsigned __int64)v37);
    if ( (__int64 *)v35[0] != &v36 )
      j_j___libc_free_0(v35[0]);
    v51[0] = &unk_49D9D40;
    sub_23FD590((__int64)v52);
    sub_9C6650(&v29);
    sub_1049740(a1, (__int64)&v39);
    v39 = &unk_49D9D40;
    sub_23FD590((__int64)v47);
  }
}
