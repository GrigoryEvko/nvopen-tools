// Function: sub_267B7C0
// Address: 0x267b7c0
//
void __fastcall sub_267B7C0(__int64 a1, __int64 a2, __int8 *a3, size_t a4, __int64 a5)
{
  __int64 *v7; // rax
  __int64 *v8; // r14
  int v9; // eax
  __int64 v10; // r15
  __int64 v11; // rax
  __int64 v12; // rbx
  __int64 v13; // rcx
  __int64 v14; // r8
  __int64 v15; // r9
  __m128i v16; // xmm0
  __m128i v17; // xmm2
  __int64 v18; // rdx
  __int64 v19; // rcx
  __int64 v20; // r8
  __int64 v21; // r9
  __int64 v22; // rax
  __int64 v23; // rbx
  __int64 v24; // rdx
  __int64 v25; // rcx
  __int64 v26; // r8
  __int64 v27; // r9
  __int64 v28; // rax
  __int64 v29; // rax
  __int64 v30; // rax
  __int64 v31; // rax
  unsigned __int64 v33[2]; // [rsp+10h] [rbp-540h] BYREF
  _QWORD v34[2]; // [rsp+20h] [rbp-530h] BYREF
  _QWORD *v35; // [rsp+30h] [rbp-520h]
  _QWORD v36[4]; // [rsp+40h] [rbp-510h] BYREF
  _BYTE v37[344]; // [rsp+60h] [rbp-4F0h] BYREF
  __int64 v38; // [rsp+1B8h] [rbp-398h]
  _QWORD v39[10]; // [rsp+1C0h] [rbp-390h] BYREF
  _BYTE v40[344]; // [rsp+210h] [rbp-340h] BYREF
  __int64 v41; // [rsp+368h] [rbp-1E8h]
  void *v42; // [rsp+370h] [rbp-1E0h] BYREF
  int v43; // [rsp+378h] [rbp-1D8h]
  char v44; // [rsp+37Ch] [rbp-1D4h]
  __int64 v45; // [rsp+380h] [rbp-1D0h]
  __m128i v46; // [rsp+388h] [rbp-1C8h]
  __int64 v47; // [rsp+398h] [rbp-1B8h]
  __m128i v48; // [rsp+3A0h] [rbp-1B0h]
  __m128i v49; // [rsp+3B0h] [rbp-1A0h]
  _QWORD v50[2]; // [rsp+3C0h] [rbp-190h] BYREF
  _BYTE v51[324]; // [rsp+3D0h] [rbp-180h] BYREF
  int v52; // [rsp+514h] [rbp-3Ch]
  __int64 v53; // [rsp+518h] [rbp-38h]

  v7 = (__int64 *)(*(__int64 (__fastcall **)(_QWORD))(a1 + 56))(*(_QWORD *)(a1 + 64));
  v8 = v7;
  if ( a4 <= 2 )
  {
    v10 = *v7;
  }
  else
  {
    if ( *(_WORD *)a3 != 19791 || (v9 = 0, a3[2] != 80) )
      v9 = 1;
    v10 = *v8;
    if ( !v9 )
    {
      v11 = sub_B2BE50(*v8);
      if ( sub_B6EA50(v11)
        || (v30 = sub_B2BE50(v10),
            v31 = sub_B6F970(v30),
            (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v31 + 48LL))(v31)) )
      {
        sub_B17560((__int64)v39, (__int64)"openmp-opt", (__int64)a3, a4, a2);
        sub_B18290((__int64)v39, "OpenMP runtime call ", 0x14u);
        sub_B16430(
          (__int64)v33,
          "OpenMPOptRuntime",
          0x10u,
          *(_BYTE **)(*(_QWORD *)a5 + 8LL),
          *(_QWORD *)(*(_QWORD *)a5 + 16LL));
        v12 = sub_23FD640((__int64)v39, (__int64)v33);
        sub_B18290(v12, " deduplicated.", 0xEu);
        v43 = *(_DWORD *)(v12 + 8);
        v44 = *(_BYTE *)(v12 + 12);
        v45 = *(_QWORD *)(v12 + 16);
        v16 = _mm_loadu_si128((const __m128i *)(v12 + 24));
        v42 = &unk_49D9D40;
        v46 = v16;
        v47 = *(_QWORD *)(v12 + 40);
        v48 = _mm_loadu_si128((const __m128i *)(v12 + 48));
        v17 = _mm_loadu_si128((const __m128i *)(v12 + 64));
        v50[0] = v51;
        v50[1] = 0x400000000LL;
        v49 = v17;
        if ( *(_DWORD *)(v12 + 88) )
          sub_26781A0((__int64)v50, v12 + 80, (__int64)v51, v13, v14, v15);
        v51[320] = *(_BYTE *)(v12 + 416);
        v52 = *(_DWORD *)(v12 + 420);
        v53 = *(_QWORD *)(v12 + 424);
        v42 = &unk_49D9D78;
        if ( v35 != v36 )
          j_j___libc_free_0((unsigned __int64)v35);
        if ( (_QWORD *)v33[0] != v34 )
          j_j___libc_free_0(v33[0]);
        sub_B18290((__int64)&v42, " [", 2u);
        sub_B18290((__int64)&v42, a3, a4);
        sub_B18290((__int64)&v42, "]", 1u);
        sub_23FE290((__int64)v33, (__int64)&v42, v18, v19, v20, v21);
        v38 = v53;
        v33[0] = (unsigned __int64)&unk_49D9D78;
        v42 = &unk_49D9D40;
        sub_23FD590((__int64)v50);
        v39[0] = &unk_49D9D40;
        sub_23FD590((__int64)v40);
        sub_1049740(v8, (__int64)v33);
        v33[0] = (unsigned __int64)&unk_49D9D40;
        sub_23FD590((__int64)v37);
      }
      return;
    }
  }
  v22 = sub_B2BE50(v10);
  if ( sub_B6EA50(v22)
    || (v28 = sub_B2BE50(v10),
        v29 = sub_B6F970(v28),
        (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v29 + 48LL))(v29)) )
  {
    sub_B17560((__int64)&v42, (__int64)"openmp-opt", (__int64)a3, a4, a2);
    sub_B18290((__int64)&v42, "OpenMP runtime call ", 0x14u);
    sub_B16430(
      (__int64)v33,
      "OpenMPOptRuntime",
      0x10u,
      *(_BYTE **)(*(_QWORD *)a5 + 8LL),
      *(_QWORD *)(*(_QWORD *)a5 + 16LL));
    v23 = sub_23FD640((__int64)&v42, (__int64)v33);
    sub_B18290(v23, " deduplicated.", 0xEu);
    sub_23FE290((__int64)v39, v23, v24, v25, v26, v27);
    v41 = *(_QWORD *)(v23 + 424);
    v39[0] = &unk_49D9D78;
    if ( v35 != v36 )
      j_j___libc_free_0((unsigned __int64)v35);
    if ( (_QWORD *)v33[0] != v34 )
      j_j___libc_free_0(v33[0]);
    v42 = &unk_49D9D40;
    sub_23FD590((__int64)v50);
    sub_1049740(v8, (__int64)v39);
    v39[0] = &unk_49D9D40;
    sub_23FD590((__int64)v40);
  }
}
