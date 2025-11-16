// Function: sub_28574C0
// Address: 0x28574c0
//
__int64 __fastcall sub_28574C0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rdx
  __m128i v7; // xmm0
  __int64 v8; // rcx
  __int16 v9; // dx
  __m128i v10; // xmm1
  __m128i v11; // xmm2
  __int64 v12; // rdx
  __int64 v13; // rbx
  __int64 v14; // rdx
  __int64 v15; // r12
  __int64 v16; // rcx
  __int64 v17; // rdx
  __int64 v18; // rcx
  __int64 v19; // rdx
  __int64 v20; // r8
  __int64 v21; // rdx
  __int64 v22; // rcx
  __int64 v23; // r8
  __int64 v24; // r9
  __int64 v25; // rdx
  __int64 v26; // rax
  __int64 v27; // rdx
  __int64 v28; // rcx
  __int64 v29; // r8
  __int64 v30; // rdx
  __int64 v31; // rcx
  __int64 v32; // r8
  __int64 v33; // r9
  unsigned __int64 *v35; // r14
  unsigned __int64 *v36; // r12
  __int64 v37; // rbx
  unsigned __int64 *v38; // r12
  __int64 v39; // [rsp+30h] [rbp-8D0h]
  __int64 v40; // [rsp+38h] [rbp-8C8h]
  __int64 v41; // [rsp+40h] [rbp-8C0h] BYREF
  __int64 v42; // [rsp+48h] [rbp-8B8h]
  __int64 v43; // [rsp+50h] [rbp-8B0h]
  int v44; // [rsp+58h] [rbp-8A8h]
  int v45; // [rsp+60h] [rbp-8A0h]
  __m128i v46; // [rsp+68h] [rbp-898h]
  _QWORD v47[2]; // [rsp+78h] [rbp-888h] BYREF
  _BYTE v48[640]; // [rsp+88h] [rbp-878h] BYREF
  __m128i v49; // [rsp+308h] [rbp-5F8h]
  __m128i v50; // [rsp+318h] [rbp-5E8h]
  __int16 v51; // [rsp+328h] [rbp-5D8h]
  __int64 v52; // [rsp+330h] [rbp-5D0h]
  _QWORD v53[2]; // [rsp+338h] [rbp-5C8h] BYREF
  char v54; // [rsp+348h] [rbp-5B8h] BYREF
  _BYTE v55[32]; // [rsp+888h] [rbp-78h] BYREF
  _BYTE v56[88]; // [rsp+8A8h] [rbp-58h] BYREF

  v6 = *(_QWORD *)(a1 + 8);
  v7 = _mm_loadu_si128((const __m128i *)(a1 + 40));
  v47[1] = 0x800000000LL;
  v8 = *(unsigned int *)(a1 + 64);
  ++*(_QWORD *)a1;
  v42 = v6;
  v40 = a1 + 56;
  LODWORD(v43) = *(_DWORD *)(a1 + 16);
  LODWORD(v6) = *(_DWORD *)(a1 + 20);
  v41 = 1;
  HIDWORD(v43) = v6;
  LODWORD(v6) = *(_DWORD *)(a1 + 24);
  *(_QWORD *)(a1 + 8) = 0;
  v44 = v6;
  LODWORD(v6) = *(_DWORD *)(a1 + 32);
  *(_QWORD *)(a1 + 16) = 0;
  v45 = v6;
  *(_DWORD *)(a1 + 24) = 0;
  v47[0] = v48;
  v46 = v7;
  if ( (_DWORD)v8 )
    sub_2852510((__int64)v47, v40, (__int64)v48, v8, a5);
  v9 = *(_WORD *)(a1 + 744);
  v10 = _mm_loadu_si128((const __m128i *)(a1 + 712));
  v11 = _mm_loadu_si128((const __m128i *)(a1 + 728));
  v53[1] = 0xC00000000LL;
  v51 = v9;
  v39 = a1 + 760;
  v52 = *(_QWORD *)(a1 + 752);
  v53[0] = &v54;
  v12 = *(unsigned int *)(a1 + 768);
  v49 = v10;
  v50 = v11;
  if ( (_DWORD)v12 )
    sub_28515F0((__int64)v53, v39, v12, v8, a5, a6);
  v13 = a1 + 2120;
  sub_C8CF70((__int64)v55, v56, 4, a1 + 2152, a1 + 2120);
  v14 = *(unsigned int *)(a1 + 24);
  if ( (_DWORD)v14 )
  {
    v35 = *(unsigned __int64 **)(a1 + 8);
    v36 = &v35[6 * v14];
    do
    {
      if ( (unsigned __int64 *)*v35 != v35 + 2 )
        _libc_free(*v35);
      v35 += 6;
    }
    while ( v36 != v35 );
    v14 = *(unsigned int *)(a1 + 24);
  }
  v15 = a2 + 2120;
  sub_C7D6A0(*(_QWORD *)(a1 + 8), 48 * v14, 8);
  *(_QWORD *)(a1 + 16) = 0;
  *(_QWORD *)(a1 + 8) = 0;
  *(_DWORD *)(a1 + 24) = 0;
  ++*(_QWORD *)a1;
  v16 = *(_QWORD *)(a2 + 8);
  ++*(_QWORD *)a2;
  v17 = *(_QWORD *)(a1 + 8);
  *(_QWORD *)(a1 + 8) = v16;
  LODWORD(v16) = *(_DWORD *)(a2 + 16);
  *(_QWORD *)(a2 + 8) = v17;
  LODWORD(v17) = *(_DWORD *)(a1 + 16);
  *(_DWORD *)(a1 + 16) = v16;
  LODWORD(v16) = *(_DWORD *)(a2 + 20);
  *(_DWORD *)(a2 + 16) = v17;
  LODWORD(v17) = *(_DWORD *)(a1 + 20);
  *(_DWORD *)(a1 + 20) = v16;
  v18 = *(unsigned int *)(a2 + 24);
  *(_DWORD *)(a2 + 20) = v17;
  LODWORD(v17) = *(_DWORD *)(a1 + 24);
  *(_DWORD *)(a1 + 24) = v18;
  *(_DWORD *)(a2 + 24) = v17;
  *(_DWORD *)(a1 + 32) = *(_DWORD *)(a2 + 32);
  *(_QWORD *)(a1 + 40) = *(_QWORD *)(a2 + 40);
  v19 = *(unsigned int *)(a2 + 48);
  *(_DWORD *)(a1 + 48) = v19;
  sub_2852510(v40, a2 + 56, v19, v18, v20);
  *(_QWORD *)(a1 + 712) = *(_QWORD *)(a2 + 712);
  *(_BYTE *)(a1 + 720) = *(_BYTE *)(a2 + 720);
  *(_QWORD *)(a1 + 728) = *(_QWORD *)(a2 + 728);
  *(_BYTE *)(a1 + 736) = *(_BYTE *)(a2 + 736);
  *(_BYTE *)(a1 + 744) = *(_BYTE *)(a2 + 744);
  *(_BYTE *)(a1 + 745) = *(_BYTE *)(a2 + 745);
  v21 = *(_QWORD *)(a2 + 752);
  *(_QWORD *)(a1 + 752) = v21;
  sub_28515F0(v39, a2 + 760, v21, v22, v23, v24);
  if ( a2 + 2120 != v13 )
    sub_C8CF80(v13, (void *)(a1 + 2152), 4, a2 + 2152, a2 + 2120);
  v25 = *(unsigned int *)(a2 + 24);
  if ( (_DWORD)v25 )
  {
    v37 = *(_QWORD *)(a2 + 8) + 48 * v25;
    v38 = *(unsigned __int64 **)(a2 + 8);
    do
    {
      if ( (unsigned __int64 *)*v38 != v38 + 2 )
        _libc_free(*v38);
      v38 += 6;
    }
    while ( (unsigned __int64 *)v37 != v38 );
    v15 = a2 + 2120;
    v25 = *(unsigned int *)(a2 + 24);
  }
  sub_C7D6A0(*(_QWORD *)(a2 + 8), 48 * v25, 8);
  v26 = v42;
  ++*(_QWORD *)a2;
  ++v41;
  *(_QWORD *)(a2 + 8) = v26;
  v42 = 0;
  *(_QWORD *)(a2 + 16) = v43;
  v43 = 0;
  *(_DWORD *)(a2 + 24) = v44;
  v44 = 0;
  *(_DWORD *)(a2 + 32) = v45;
  *(_QWORD *)(a2 + 40) = v46.m128i_i64[0];
  *(_DWORD *)(a2 + 48) = v46.m128i_i32[2];
  sub_2852510(a2 + 56, (__int64)v47, v27, v28, v29);
  *(_QWORD *)(a2 + 712) = v49.m128i_i64[0];
  *(_BYTE *)(a2 + 720) = v49.m128i_i8[8];
  *(_QWORD *)(a2 + 728) = v50.m128i_i64[0];
  *(_BYTE *)(a2 + 736) = v50.m128i_i8[8];
  *(_WORD *)(a2 + 744) = v51;
  *(_QWORD *)(a2 + 752) = v52;
  sub_28515F0(a2 + 760, (__int64)v53, v30, v31, v32, v33);
  sub_C8CF80(v15, (void *)(a2 + 2152), 4, (__int64)v56, (__int64)v55);
  return sub_2855330((__int64)&v41);
}
