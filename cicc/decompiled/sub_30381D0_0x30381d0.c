// Function: sub_30381D0
// Address: 0x30381d0
//
__int64 __fastcall sub_30381D0(__int64 a1, __int64 a2, unsigned int a3, __int64 a4)
{
  __int64 v7; // rax
  __int64 v8; // rax
  void *v9; // r9
  int v10; // r8d
  __int64 v11; // r10
  __int64 v12; // rcx
  __int64 v13; // rax
  _QWORD *v14; // rbx
  __int64 v15; // rsi
  bool v16; // cf
  __int16 v17; // r13
  __int64 v18; // rax
  __int64 v19; // rdx
  int v20; // edx
  int v21; // r9d
  __int64 v22; // rax
  __int64 v23; // rsi
  __int64 v24; // r12
  __int64 v26; // rsi
  __int64 v27; // rax
  __m128i v28; // xmm0
  unsigned __int16 *v29; // rbx
  __int64 v30; // r8
  int v31; // ecx
  __m128i v32; // rax
  __m128i v33; // xmm1
  __int64 v34; // rax
  __int128 v35; // [rsp-20h] [rbp-F0h]
  __int64 v36; // [rsp+0h] [rbp-D0h]
  __int64 v37; // [rsp+8h] [rbp-C8h]
  void *v38; // [rsp+10h] [rbp-C0h]
  int v39; // [rsp+18h] [rbp-B8h]
  __int64 v40; // [rsp+18h] [rbp-B8h]
  __int64 v41; // [rsp+20h] [rbp-B0h] BYREF
  int v42; // [rsp+28h] [rbp-A8h]
  __m128i v43; // [rsp+30h] [rbp-A0h] BYREF
  __m128i v44; // [rsp+40h] [rbp-90h]
  void *v45; // [rsp+50h] [rbp-80h] BYREF
  __int64 v46; // [rsp+58h] [rbp-78h]
  __int64 v47; // [rsp+60h] [rbp-70h]
  __m128i v48; // [rsp+68h] [rbp-68h]
  const char *v49; // [rsp+78h] [rbp-58h]
  __int16 v50; // [rsp+98h] [rbp-38h]

  v7 = *(_QWORD *)(a1 + 537016);
  if ( *(_DWORD *)(v7 + 336) <= 0x48u || *(_DWORD *)(v7 + 340) <= 0x207u )
  {
    v26 = *(_QWORD *)(a2 + 80);
    v27 = **(_QWORD **)(a4 + 40);
    v41 = v26;
    v40 = v27;
    if ( v26 )
      sub_B96E90((__int64)&v41, v26, 1);
    v42 = *(_DWORD *)(a2 + 72);
    sub_B157E0((__int64)&v43, &v41);
    v28 = _mm_loadu_si128(&v43);
    v50 = 259;
    v47 = v40;
    v46 = 24;
    v48 = v28;
    v45 = &unk_49D9E88;
    v49 = "Support for dynamic alloca introduced in PTX ISA version 7.3 and requires target sm_52.";
    if ( v41 )
      sub_B91220((__int64)&v41, v41);
    sub_B6EB20(*(_QWORD *)(a4 + 64), (__int64)&v45);
    v29 = (unsigned __int16 *)(*(_QWORD *)(a2 + 48) + 16LL * a3);
    v30 = *((_QWORD *)v29 + 1);
    v31 = *v29;
    v41 = 0;
    v42 = 0;
    v32.m128i_i64[0] = sub_3400BD0(a4, 0, (unsigned int)&v41, v31, v30, 0, 0);
    v43 = v32;
    if ( v41 )
      sub_B91220((__int64)&v41, v41);
    v33 = _mm_loadu_si128((const __m128i *)*(_QWORD *)(a2 + 40));
    v41 = 0;
    v42 = 0;
    v44 = v33;
    v34 = sub_3411660(a4, &v43, 2, &v41);
    v23 = v41;
    v24 = v34;
    if ( v41 )
      goto LABEL_8;
  }
  else
  {
    v8 = *(_QWORD *)(a2 + 40);
    v9 = *(void **)v8;
    v10 = *(_DWORD *)(v8 + 8);
    v11 = *(_QWORD *)(v8 + 40);
    v12 = *(_QWORD *)(v8 + 48);
    v13 = *(_QWORD *)(*(_QWORD *)(v8 + 80) + 96LL);
    v14 = *(_QWORD **)(v13 + 24);
    if ( *(_DWORD *)(v13 + 32) > 0x40u )
      v14 = (_QWORD *)*v14;
    v15 = *(_QWORD *)(a2 + 80);
    v41 = v15;
    if ( v15 )
    {
      v36 = v12;
      v37 = v11;
      v38 = v9;
      v39 = v10;
      sub_B96E90((__int64)&v41, v15, 1);
      v12 = v36;
      v11 = v37;
      v9 = v38;
      v10 = v39;
    }
    v42 = *(_DWORD *)(a2 + 72);
    v16 = *(_BYTE *)(*(_QWORD *)(a1 + 537008) + 1264LL) == 0;
    LODWORD(v46) = v10;
    v17 = 7 - (v16 - 1);
    v45 = v9;
    v18 = sub_33FB310(a4, v11, v12, &v41, (7 - (v16 - 1)) & 0xF, 0);
    v48.m128i_i64[0] = v19;
    v47 = v18;
    v48.m128i_i64[1] = sub_3400BD0(a4, (_DWORD)v14, (unsigned int)&v41, 7, 0, 1, 0);
    LODWORD(v49) = v20;
    v44.m128i_i16[0] = 1;
    *((_QWORD *)&v35 + 1) = 3;
    *(_QWORD *)&v35 = &v45;
    v43.m128i_i16[0] = v17;
    v43.m128i_i64[1] = 0;
    v44.m128i_i64[1] = 0;
    v22 = sub_3411BE0(a4, 541, (unsigned int)&v41, (unsigned int)&v43, 2, v21, v35);
    v23 = v41;
    v24 = v22;
    if ( v41 )
LABEL_8:
      sub_B91220((__int64)&v41, v23);
  }
  return v24;
}
