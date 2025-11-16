// Function: sub_1997A20
// Address: 0x1997a20
//
__int64 __fastcall sub_1997A20(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, int a6)
{
  __int64 v6; // rdx
  __m128i v7; // xmm0
  int v8; // ecx
  __int64 v9; // rdx
  __int64 v10; // rcx
  __int64 v11; // rdx
  __int64 v12; // rdx
  __int64 v13; // rbx
  __int64 v14; // rcx
  __int64 v15; // rcx
  __int64 v16; // rdx
  __int64 v17; // rdx
  __int64 v18; // rcx
  __int64 v19; // r8
  int v20; // r9d
  __int64 v21; // rdx
  __int64 v22; // rax
  __int64 v23; // rdx
  __int64 v24; // rcx
  __int64 v25; // r8
  int v26; // r9d
  unsigned __int64 *v28; // r14
  unsigned __int64 *v29; // r15
  unsigned __int64 *v30; // rax
  unsigned __int64 *v31; // rbx
  __int64 v32; // [rsp+20h] [rbp-800h]
  __int64 *v33; // [rsp+28h] [rbp-7F8h]
  unsigned __int64 *v34; // [rsp+28h] [rbp-7F8h]
  __int64 v35; // [rsp+30h] [rbp-7F0h] BYREF
  __int64 v36; // [rsp+38h] [rbp-7E8h]
  __int64 v37; // [rsp+40h] [rbp-7E0h]
  int v38; // [rsp+48h] [rbp-7D8h]
  int v39; // [rsp+50h] [rbp-7D0h]
  __m128i v40; // [rsp+58h] [rbp-7C8h]
  _QWORD v41[2]; // [rsp+68h] [rbp-7B8h] BYREF
  char v42; // [rsp+78h] [rbp-7A8h] BYREF
  __int64 v43; // [rsp+2F8h] [rbp-528h]
  __int64 v44; // [rsp+300h] [rbp-520h]
  __int16 v45; // [rsp+308h] [rbp-518h]
  __int64 v46; // [rsp+310h] [rbp-510h]
  __int64 v47[2]; // [rsp+318h] [rbp-508h] BYREF
  char v48; // [rsp+328h] [rbp-4F8h] BYREF
  _QWORD v49[5]; // [rsp+7A8h] [rbp-78h] BYREF
  _BYTE v50[80]; // [rsp+7D0h] [rbp-50h] BYREF

  v6 = *(_QWORD *)(a1 + 8);
  v7 = _mm_loadu_si128((const __m128i *)(a1 + 40));
  v41[1] = 0x800000000LL;
  v8 = *(_DWORD *)(a1 + 64);
  ++*(_QWORD *)a1;
  v36 = v6;
  v32 = a1 + 56;
  LODWORD(v37) = *(_DWORD *)(a1 + 16);
  LODWORD(v6) = *(_DWORD *)(a1 + 20);
  v35 = 1;
  HIDWORD(v37) = v6;
  LODWORD(v6) = *(_DWORD *)(a1 + 24);
  *(_QWORD *)(a1 + 8) = 0;
  v38 = v6;
  LODWORD(v6) = *(_DWORD *)(a1 + 32);
  *(_QWORD *)(a1 + 16) = 0;
  v39 = v6;
  *(_DWORD *)(a1 + 24) = 0;
  v41[0] = &v42;
  v40 = v7;
  if ( v8 )
    sub_1995960((__int64)v41, v32);
  v9 = *(_QWORD *)(a1 + 720);
  v10 = *(_QWORD *)(a1 + 712);
  v47[1] = 0xC00000000LL;
  v44 = v9;
  v33 = (__int64 *)(a1 + 744);
  v45 = *(_WORD *)(a1 + 728);
  v11 = *(_QWORD *)(a1 + 736);
  v43 = v10;
  v46 = v11;
  v47[0] = (__int64)&v48;
  v12 = *(unsigned int *)(a1 + 752);
  if ( (_DWORD)v12 )
    sub_1996030((__int64)v47, v33, v12, v10, a5, a6);
  v13 = a1 + 1912;
  sub_16CCEE0(v49, (__int64)v50, 4, a1 + 1912);
  v14 = *(unsigned int *)(a1 + 24);
  if ( (_DWORD)v14 )
  {
    v28 = *(unsigned __int64 **)(a1 + 8);
    v29 = &v28[6 * v14];
    do
    {
      if ( (unsigned __int64 *)*v28 != v28 + 2 )
        _libc_free(*v28);
      v28 += 6;
    }
    while ( v29 != v28 );
  }
  j___libc_free_0(*(_QWORD *)(a1 + 8));
  ++*(_QWORD *)a1;
  *(_QWORD *)(a1 + 16) = 0;
  *(_QWORD *)(a1 + 8) = 0;
  *(_DWORD *)(a1 + 24) = 0;
  v15 = *(_QWORD *)(a2 + 8);
  ++*(_QWORD *)a2;
  v16 = *(_QWORD *)(a1 + 8);
  *(_QWORD *)(a1 + 8) = v15;
  LODWORD(v15) = *(_DWORD *)(a2 + 16);
  *(_QWORD *)(a2 + 8) = v16;
  LODWORD(v16) = *(_DWORD *)(a1 + 16);
  *(_DWORD *)(a1 + 16) = v15;
  LODWORD(v15) = *(_DWORD *)(a2 + 20);
  *(_DWORD *)(a2 + 16) = v16;
  LODWORD(v16) = *(_DWORD *)(a1 + 20);
  *(_DWORD *)(a1 + 20) = v15;
  LODWORD(v15) = *(_DWORD *)(a2 + 24);
  *(_DWORD *)(a2 + 20) = v16;
  LODWORD(v16) = *(_DWORD *)(a1 + 24);
  *(_DWORD *)(a1 + 24) = v15;
  *(_DWORD *)(a2 + 24) = v16;
  *(_DWORD *)(a1 + 32) = *(_DWORD *)(a2 + 32);
  *(_QWORD *)(a1 + 40) = *(_QWORD *)(a2 + 40);
  *(_DWORD *)(a1 + 48) = *(_DWORD *)(a2 + 48);
  sub_1995960(v32, a2 + 56);
  *(_QWORD *)(a1 + 712) = *(_QWORD *)(a2 + 712);
  *(_QWORD *)(a1 + 720) = *(_QWORD *)(a2 + 720);
  *(_BYTE *)(a1 + 728) = *(_BYTE *)(a2 + 728);
  *(_BYTE *)(a1 + 729) = *(_BYTE *)(a2 + 729);
  v17 = *(_QWORD *)(a2 + 736);
  *(_QWORD *)(a1 + 736) = v17;
  sub_1996030((__int64)v33, (__int64 *)(a2 + 744), v17, v18, v19, v20);
  if ( a2 + 1912 != v13 )
    sub_16CCF00(v13, 4, a2 + 1912);
  v21 = *(unsigned int *)(a2 + 24);
  if ( (_DWORD)v21 )
  {
    v30 = *(unsigned __int64 **)(a2 + 8);
    v31 = &v30[6 * v21];
    do
    {
      if ( (unsigned __int64 *)*v30 != v30 + 2 )
      {
        v34 = v30;
        _libc_free(*v30);
        v30 = v34;
      }
      v30 += 6;
    }
    while ( v31 != v30 );
  }
  j___libc_free_0(*(_QWORD *)(a2 + 8));
  v22 = v36;
  ++*(_QWORD *)a2;
  ++v35;
  *(_QWORD *)(a2 + 8) = v22;
  v36 = 0;
  *(_QWORD *)(a2 + 16) = v37;
  v37 = 0;
  *(_DWORD *)(a2 + 24) = v38;
  v38 = 0;
  *(_DWORD *)(a2 + 32) = v39;
  *(_QWORD *)(a2 + 40) = v40.m128i_i64[0];
  *(_DWORD *)(a2 + 48) = v40.m128i_i32[2];
  sub_1995960(a2 + 56, (__int64)v41);
  *(_QWORD *)(a2 + 712) = v43;
  *(_QWORD *)(a2 + 720) = v44;
  *(_WORD *)(a2 + 728) = v45;
  *(_QWORD *)(a2 + 736) = v46;
  sub_1996030(a2 + 744, v47, v23, v24, v25, v26);
  sub_16CCF00(a2 + 1912, 4, (__int64)v49);
  return sub_1996B30((__int64)&v35);
}
