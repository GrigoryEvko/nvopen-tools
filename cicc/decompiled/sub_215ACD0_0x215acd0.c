// Function: sub_215ACD0
// Address: 0x215acd0
//
__int64 __fastcall sub_215ACD0(__int64 a1, __int64 a2)
{
  _QWORD *v4; // r8
  _BYTE *v5; // rsi
  _BYTE *v6; // r15
  _QWORD *v7; // r9
  __int64 v8; // rax
  __int64 v9; // rdx
  __int64 v10; // rax
  __int64 v11; // rax
  __int64 v12; // rax
  __int64 v13; // rax
  __int64 v14; // rax
  __int64 v15; // rdi
  __int64 v16; // rax
  __int64 v17; // rdi
  __int64 v18; // rdx
  __int64 v19; // rax
  __int64 v20; // rdi
  void (*v21)(); // rcx
  void (*v22)(void); // rdx
  __int64 v23; // rdx
  __int64 *v24; // rdi
  __int64 v25; // rax
  void (*v26)(); // rdx
  void (*v27)(); // rax
  void (*v28)(); // rax
  __int64 v29; // rbx
  __int64 v30; // rdi
  char *v31; // rdi
  __int64 v32; // rdx
  char *v33; // rbx
  _QWORD *v34; // r12
  __int64 v36; // rax
  _QWORD *v37; // rdi
  int v38; // r8d
  int v39; // r9d
  __int64 v40; // rax
  __m128i *v41; // rax
  __int64 v42; // [rsp+0h] [rbp-142F0h]
  _QWORD *v43; // [rsp+0h] [rbp-142F0h]
  _QWORD *v44; // [rsp+8h] [rbp-142E8h]
  _QWORD *v45; // [rsp+8h] [rbp-142E8h]
  _QWORD *v46; // [rsp+10h] [rbp-142E0h]
  _QWORD *v47; // [rsp+18h] [rbp-142D8h]
  __int64 v48; // [rsp+20h] [rbp-142D0h] BYREF
  __int64 v49; // [rsp+28h] [rbp-142C8h]
  __m128i v50; // [rsp+30h] [rbp-142C0h] BYREF
  __m128i v51; // [rsp+40h] [rbp-142B0h] BYREF
  const char *v52; // [rsp+50h] [rbp-142A0h]
  _QWORD *v53; // [rsp+60h] [rbp-14290h] BYREF
  __int64 v54; // [rsp+68h] [rbp-14288h]
  _QWORD v55[2]; // [rsp+70h] [rbp-14280h] BYREF
  int v56; // [rsp+80h] [rbp-14270h]
  __int64 *v57; // [rsp+88h] [rbp-14268h]
  _QWORD *v58; // [rsp+90h] [rbp-14260h] BYREF
  __int64 v59; // [rsp+98h] [rbp-14258h]
  _QWORD v60[16]; // [rsp+A0h] [rbp-14250h] BYREF
  _QWORD v61[27]; // [rsp+120h] [rbp-141D0h] BYREF
  __int64 *v62; // [rsp+1F8h] [rbp-140F8h]
  __int64 v63; // [rsp+208h] [rbp-140E8h] BYREF
  void *v64; // [rsp+228h] [rbp-140C8h]
  void *v65; // [rsp+260h] [rbp-14090h] BYREF
  char *v66; // [rsp+388h] [rbp-13F68h]
  unsigned int v67; // [rsp+390h] [rbp-13F60h]
  char v68; // [rsp+398h] [rbp-13F58h] BYREF
  void *v69; // [rsp+3D8h] [rbp-13F18h]
  __int64 v70; // [rsp+3F8h] [rbp-13EF8h]
  __int64 v71; // [rsp+12528h] [rbp-1DC8h]
  char v72[8]; // [rsp+14278h] [rbp-78h] BYREF
  void *v73; // [rsp+14280h] [rbp-70h] BYREF

  v4 = *(_QWORD **)(a1 + 232);
  v5 = (_BYTE *)v4[70];
  v6 = (_BYTE *)v4[66];
  v7 = v4 + 59;
  v8 = v4[67];
  if ( v5 )
  {
    v42 = v4[67];
    v44 = v4 + 59;
    v9 = (__int64)&v5[v4[71]];
    v46 = *(_QWORD **)(a1 + 232);
    v58 = v60;
    sub_214AFD0((__int64 *)&v58, v5, v9);
    v8 = v42;
    v7 = v44;
    v4 = v46;
  }
  else
  {
    v59 = 0;
    v58 = v60;
    LOBYTE(v60[0]) = 0;
  }
  if ( v6 )
  {
    v43 = v7;
    v45 = v4;
    v53 = v55;
    sub_214AFD0((__int64 *)&v53, v6, (__int64)&v6[v8]);
    v7 = v43;
    v4 = v45;
  }
  else
  {
    LOBYTE(v55[0]) = 0;
    v53 = v55;
    v54 = 0;
  }
  sub_21651F0(v61, v7, &v53, &v58, v4);
  if ( v53 != v55 )
    j_j___libc_free_0(v53, v55[0] + 1LL);
  if ( v58 != v60 )
    j_j___libc_free_0(v58, v60[0] + 1LL);
  v10 = sub_16321C0(a2, (__int64)"llvm.global_ctors", 17, 1);
  if ( v10 )
  {
    v11 = *(_QWORD *)(v10 - 24);
    if ( *(_BYTE *)(v11 + 16) == 6 && (*(_DWORD *)(v11 + 20) & 0xFFFFFFF) != 0 )
      sub_16BD130("Module has a nontrivial global ctor, which NVPTX does not support.", 1u);
  }
  v12 = sub_16321C0(a2, (__int64)"llvm.global_dtors", 17, 1);
  if ( v12 )
  {
    v13 = *(_QWORD *)(v12 - 24);
    if ( *(_BYTE *)(v13 + 16) == 6 && (*(_DWORD *)(v13 + 20) & 0xFFFFFFF) != 0 )
      sub_16BD130("Module has a nontrivial global dtor, which NVPTX does not support.", 1u);
  }
  v59 = 0x8000000000LL;
  v58 = v60;
  v56 = 1;
  v53 = &unk_49EFC48;
  v55[1] = 0;
  v57 = (__int64 *)&v58;
  v55[0] = 0;
  v54 = 0;
  sub_16E7A40((__int64)&v53, 0, 0, 0);
  v14 = sub_160F9A0(*(_QWORD *)(a1 + 8), (__int64)&unk_4FC6A0E, 1u);
  v15 = v14;
  if ( v14 )
    v15 = (*(__int64 (__fastcall **)(__int64, void *))(*(_QWORD *)v14 + 104LL))(v14, &unk_4FC6A0E);
  *(_QWORD *)(a1 + 272) = v15;
  v16 = sub_396DD80(a1);
  (*(void (__fastcall **)(__int64, _QWORD, _QWORD))(*(_QWORD *)v16 + 16LL))(
    v16,
    *(_QWORD *)(a1 + 248),
    *(_QWORD *)(a1 + 232));
  (*(void (__fastcall **)(_QWORD, _QWORD))(**(_QWORD **)(a1 + 256) + 168LL))(*(_QWORD *)(a1 + 256), 0);
  if ( *(_BYTE *)(*(_QWORD *)(a1 + 240) + 344LL) )
  {
    v36 = sub_22077B0(6736);
    v37 = (_QWORD *)v36;
    if ( v36 )
    {
      v47 = (_QWORD *)v36;
      sub_3988FF0(v36, a1, a2);
      v37 = v47;
      *v47 = &unk_4A018B8;
    }
    *(_QWORD *)(a1 + 504) = v37;
    sub_399B1E0();
    v50.m128i_i64[0] = *(_QWORD *)(a1 + 504);
    v50.m128i_i64[1] = (__int64)"NVPTX DWARF Debug Writer";
    v51.m128i_i64[0] = (__int64)"NVPTX Debug Info Emission";
    v51.m128i_i64[1] = (__int64)"NVPTX DWARF Emission";
    v52 = "NVPTX DWARF Emission";
    v40 = *(unsigned int *)(a1 + 432);
    if ( (unsigned int)v40 >= *(_DWORD *)(a1 + 436) )
    {
      sub_16CD150(a1 + 424, (const void *)(a1 + 440), 0, 40, v38, v39);
      v40 = *(unsigned int *)(a1 + 432);
    }
    v41 = (__m128i *)(*(_QWORD *)(a1 + 424) + 40 * v40);
    *v41 = _mm_loadu_si128(&v50);
    v41[1] = _mm_loadu_si128(&v51);
    v41[2].m128i_i64[0] = (__int64)v52;
    ++*(_DWORD *)(a1 + 432);
  }
  *(_DWORD *)(a1 + 912) = 0;
  sub_214F370(a1, a2, (__int64)&v53, (__int64)v61);
  v17 = *(_QWORD *)(a1 + 256);
  v18 = *((unsigned int *)v57 + 2);
  v19 = *v57;
  v51.m128i_i16[0] = 261;
  v48 = v19;
  v49 = v18;
  v50.m128i_i64[0] = (__int64)&v48;
  sub_38DD5A0(v17, &v50);
  if ( *(_QWORD *)(a2 + 96) )
  {
    v20 = *(_QWORD *)(a1 + 256);
    v21 = *(void (**)())(*(_QWORD *)v20 + 104LL);
    v50.m128i_i64[0] = (__int64)"Start of file scope inline assembly";
    v51.m128i_i16[0] = 259;
    if ( v21 != nullsub_580 )
    {
      ((void (__fastcall *)(__int64, __m128i *, __int64))v21)(v20, &v50, 1);
      v20 = *(_QWORD *)(a1 + 256);
    }
    v22 = *(void (**)(void))(*(_QWORD *)v20 + 144LL);
    if ( v22 != nullsub_581 )
    {
      v22();
      v20 = *(_QWORD *)(a1 + 256);
    }
    v23 = *(_QWORD *)(a2 + 88);
    v50.m128i_i64[0] = (__int64)&v48;
    v48 = v23;
    v49 = *(_QWORD *)(a2 + 96);
    v51.m128i_i16[0] = 261;
    sub_38DD5A0(v20, &v50);
    v24 = *(__int64 **)(a1 + 256);
    v25 = *v24;
    v26 = *(void (**)())(*v24 + 144);
    if ( v26 != nullsub_581 )
    {
      ((void (__fastcall *)(__int64 *, __m128i *))v26)(v24, &v50);
      v24 = *(__int64 **)(a1 + 256);
      v25 = *v24;
    }
    v27 = *(void (**)())(v25 + 104);
    v50.m128i_i64[0] = (__int64)"End of file scope inline assembly";
    v51.m128i_i16[0] = 259;
    if ( v27 != nullsub_580 )
    {
      ((void (__fastcall *)(__int64 *, __m128i *, __int64))v27)(v24, &v50, 1);
      v24 = *(__int64 **)(a1 + 256);
    }
    v28 = *(void (**)())(*v24 + 144);
    if ( v28 != nullsub_581 )
      ((void (__fastcall *)(__int64 *, __m128i *))v28)(v24, &v50);
  }
  *(_BYTE *)(a1 + 784) = 0;
  *(_QWORD *)(a1 + 792) = 0;
  v53 = &unk_49EFD28;
  sub_16E7960((__int64)&v53);
  if ( v58 != v60 )
    _libc_free((unsigned __int64)v58);
  v61[0] = &unk_4A02928;
  v73 = &unk_4A01970;
  nullsub_1993(&v73);
  nullsub_1991(v72);
  v29 = v71;
  v69 = &unk_49FEE48;
  while ( v29 )
  {
    sub_214B3D0(*(_QWORD *)(v29 + 24));
    v30 = v29;
    v29 = *(_QWORD *)(v29 + 16);
    j_j___libc_free_0(v30, 48);
  }
  j___libc_free_0(v70);
  v31 = v66;
  v32 = v67;
  v33 = v66;
  v64 = &unk_4A01B58;
  v65 = &unk_4A02228;
  if ( v67 )
  {
    do
    {
      v34 = *(_QWORD **)v33;
      if ( *(_QWORD *)v33 )
      {
        if ( (_QWORD *)*v34 != v34 + 2 )
          j_j___libc_free_0(*v34, v34[2] + 1LL);
        j_j___libc_free_0(v34, 32);
        v31 = v66;
        v32 = v67;
      }
      v33 += 8;
    }
    while ( v33 != &v31[8 * v32] );
  }
  if ( v31 != &v68 )
    _libc_free((unsigned __int64)v31);
  v65 = &unk_4A02068;
  sub_1F4A9C0(&v65);
  v64 = &unk_4A012A0;
  nullsub_759();
  if ( v62 != &v63 )
    j_j___libc_free_0(v62, v63 + 1);
  v61[0] = &unk_4A027E0;
  sub_39BA210(v61);
  return 0;
}
