// Function: sub_2D460D0
// Address: 0x2d460d0
//
__int64 __fastcall sub_2D460D0(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        unsigned __int8 a4,
        unsigned int a5,
        unsigned __int8 a6,
        __int64 (__fastcall *a7)(__int64, __int64, __int64),
        __int64 a8,
        void (__fastcall *a9)(__int64, __int64, __int64, __int64, __int64, _QWORD, _QWORD, _QWORD, _QWORD *, __int64 *, __int64),
        __int64 a10,
        __int64 a11)
{
  __int64 v12; // r13
  _QWORD *v13; // rdi
  __int64 *v14; // rsi
  __int64 v15; // rdx
  __int64 v16; // r14
  __int64 v17; // rax
  _QWORD *v18; // rdi
  _QWORD *v19; // rax
  __int64 v20; // r9
  __int64 v21; // rbx
  __int64 v22; // r14
  __int64 v23; // r13
  __int64 v24; // rdx
  unsigned int v25; // esi
  _QWORD *v26; // rax
  __int64 v27; // r14
  __int64 v28; // r13
  __int64 v29; // rbx
  __int64 v30; // rdx
  unsigned int v31; // esi
  __int64 v32; // r13
  int v33; // eax
  int v34; // eax
  unsigned int v35; // edx
  __int64 v36; // rax
  __int64 v37; // rdx
  __int64 v38; // rdx
  __int64 v39; // r8
  unsigned int v40; // eax
  __int64 v41; // rbx
  int v42; // eax
  int v43; // eax
  unsigned int v44; // edx
  __int64 v45; // rax
  __int64 v46; // rdx
  __int64 v47; // rdx
  _QWORD *v48; // rax
  __int64 v49; // r9
  __int64 v50; // rbx
  __int64 v51; // r13
  __int64 v52; // r14
  __int64 v53; // rdx
  unsigned int v54; // esi
  __int64 v56; // [rsp-8h] [rbp-118h]
  __int64 v57; // [rsp+8h] [rbp-108h]
  _QWORD *v63; // [rsp+58h] [rbp-B8h]
  __int64 v64; // [rsp+58h] [rbp-B8h]
  __int64 v65; // [rsp+60h] [rbp-B0h]
  __int64 v66; // [rsp+68h] [rbp-A8h]
  __int64 v67; // [rsp+78h] [rbp-98h] BYREF
  _QWORD v68[4]; // [rsp+80h] [rbp-90h] BYREF
  __int16 v69; // [rsp+A0h] [rbp-70h]
  _QWORD v70[4]; // [rsp+B0h] [rbp-60h] BYREF
  __int16 v71; // [rsp+D0h] [rbp-40h]

  v12 = *(_QWORD *)(a1 + 72);
  v13 = *(_QWORD **)(a1 + 48);
  v14 = *(__int64 **)(a1 + 56);
  v15 = *(unsigned __int16 *)(a1 + 64);
  v16 = v13[9];
  v63 = v13;
  v70[0] = "atomicrmw.end";
  v71 = 259;
  v65 = sub_AA8550(v13, v14, v15, (__int64)v70, 0);
  v70[0] = "atomicrmw.start";
  v71 = 259;
  v17 = sub_22077B0(0x50u);
  v66 = v17;
  if ( v17 )
    sub_AA4D50(v17, v12, (__int64)v70, v16, v65);
  v18 = (_QWORD *)((v13[6] & 0xFFFFFFFFFFFFFFF8LL) - 24);
  if ( (v63[6] & 0xFFFFFFFFFFFFFFF8LL) == 0 )
    v18 = 0;
  sub_B43D60(v18);
  *(_QWORD *)(a1 + 56) = v63 + 6;
  *(_QWORD *)(a1 + 48) = v63;
  v71 = 257;
  *(_WORD *)(a1 + 64) = 0;
  v69 = 257;
  v19 = sub_BD2C40(80, 1u);
  v21 = (__int64)v19;
  if ( v19 )
  {
    sub_B4D190((__int64)v19, a2, a3, (__int64)v70, 0, a4, 0, 0);
    v20 = v56;
  }
  (*(void (__fastcall **)(_QWORD, __int64, _QWORD *, _QWORD, _QWORD, __int64))(**(_QWORD **)(a1 + 88) + 16LL))(
    *(_QWORD *)(a1 + 88),
    v21,
    v68,
    *(_QWORD *)(a1 + 56),
    *(_QWORD *)(a1 + 64),
    v20);
  v22 = *(_QWORD *)a1;
  v23 = *(_QWORD *)a1 + 16LL * *(unsigned int *)(a1 + 8);
  if ( *(_QWORD *)a1 != v23 )
  {
    do
    {
      v24 = *(_QWORD *)(v22 + 8);
      v25 = *(_DWORD *)v22;
      v22 += 16;
      sub_B99FD0(v21, v25, v24);
    }
    while ( v23 != v22 );
  }
  v71 = 257;
  v26 = sub_BD2C40(72, 1u);
  v27 = (__int64)v26;
  if ( v26 )
    sub_B4C8F0((__int64)v26, v66, 1u, 0, 0);
  (*(void (__fastcall **)(_QWORD, __int64, _QWORD *, _QWORD, _QWORD))(**(_QWORD **)(a1 + 88) + 16LL))(
    *(_QWORD *)(a1 + 88),
    v27,
    v70,
    *(_QWORD *)(a1 + 56),
    *(_QWORD *)(a1 + 64));
  if ( *(_QWORD *)a1 != *(_QWORD *)a1 + 16LL * *(unsigned int *)(a1 + 8) )
  {
    v57 = v21;
    v28 = *(_QWORD *)a1 + 16LL * *(unsigned int *)(a1 + 8);
    v29 = *(_QWORD *)a1;
    do
    {
      v30 = *(_QWORD *)(v29 + 8);
      v31 = *(_DWORD *)v29;
      v29 += 16;
      sub_B99FD0(v27, v31, v30);
    }
    while ( v28 != v29 );
    v21 = v57;
  }
  *(_WORD *)(a1 + 64) = 0;
  *(_QWORD *)(a1 + 48) = v66;
  *(_QWORD *)(a1 + 56) = v66 + 48;
  v70[0] = "loaded";
  v71 = 259;
  v32 = sub_D5C860((__int64 *)a1, a2, 2, (__int64)v70);
  v33 = *(_DWORD *)(v32 + 4) & 0x7FFFFFF;
  if ( v33 == *(_DWORD *)(v32 + 72) )
  {
    sub_B48D90(v32);
    v33 = *(_DWORD *)(v32 + 4) & 0x7FFFFFF;
  }
  v34 = (v33 + 1) & 0x7FFFFFF;
  v35 = v34 | *(_DWORD *)(v32 + 4) & 0xF8000000;
  v36 = *(_QWORD *)(v32 - 8) + 32LL * (unsigned int)(v34 - 1);
  *(_DWORD *)(v32 + 4) = v35;
  if ( *(_QWORD *)v36 )
  {
    v37 = *(_QWORD *)(v36 + 8);
    **(_QWORD **)(v36 + 16) = v37;
    if ( v37 )
      *(_QWORD *)(v37 + 16) = *(_QWORD *)(v36 + 16);
  }
  *(_QWORD *)v36 = v21;
  if ( v21 )
  {
    v38 = *(_QWORD *)(v21 + 16);
    *(_QWORD *)(v36 + 8) = v38;
    if ( v38 )
      *(_QWORD *)(v38 + 16) = v36 + 8;
    *(_QWORD *)(v36 + 16) = v21 + 16;
    *(_QWORD *)(v21 + 16) = v36;
  }
  *(_QWORD *)(*(_QWORD *)(v32 - 8) + 32LL * *(unsigned int *)(v32 + 72)
                                   + 8LL * ((*(_DWORD *)(v32 + 4) & 0x7FFFFFFu) - 1)) = v63;
  v67 = 0;
  v39 = a7(a8, a1, v32);
  v40 = 2;
  v68[0] = 0;
  if ( a5 != 1 )
    v40 = a5;
  a9(a10, a1, a3, v32, v39, a4, v40, a6, v68, &v67, a11);
  v41 = v67;
  v42 = *(_DWORD *)(v32 + 4) & 0x7FFFFFF;
  if ( v42 == *(_DWORD *)(v32 + 72) )
  {
    sub_B48D90(v32);
    v42 = *(_DWORD *)(v32 + 4) & 0x7FFFFFF;
  }
  v43 = (v42 + 1) & 0x7FFFFFF;
  v44 = v43 | *(_DWORD *)(v32 + 4) & 0xF8000000;
  v45 = *(_QWORD *)(v32 - 8) + 32LL * (unsigned int)(v43 - 1);
  *(_DWORD *)(v32 + 4) = v44;
  if ( *(_QWORD *)v45 )
  {
    v46 = *(_QWORD *)(v45 + 8);
    **(_QWORD **)(v45 + 16) = v46;
    if ( v46 )
      *(_QWORD *)(v46 + 16) = *(_QWORD *)(v45 + 16);
  }
  *(_QWORD *)v45 = v41;
  if ( v41 )
  {
    v47 = *(_QWORD *)(v41 + 16);
    *(_QWORD *)(v45 + 8) = v47;
    if ( v47 )
      *(_QWORD *)(v47 + 16) = v45 + 8;
    *(_QWORD *)(v45 + 16) = v41 + 16;
    *(_QWORD *)(v41 + 16) = v45;
  }
  *(_QWORD *)(*(_QWORD *)(v32 - 8) + 32LL * *(unsigned int *)(v32 + 72)
                                   + 8LL * ((*(_DWORD *)(v32 + 4) & 0x7FFFFFFu) - 1)) = v66;
  v64 = v68[0];
  v71 = 257;
  v48 = sub_BD2C40(72, 3u);
  v50 = (__int64)v48;
  if ( v48 )
    sub_B4C9A0((__int64)v48, v65, v66, v64, 3u, v49, 0, 0);
  (*(void (__fastcall **)(_QWORD, __int64, _QWORD *, _QWORD, _QWORD))(**(_QWORD **)(a1 + 88) + 16LL))(
    *(_QWORD *)(a1 + 88),
    v50,
    v70,
    *(_QWORD *)(a1 + 56),
    *(_QWORD *)(a1 + 64));
  v51 = *(_QWORD *)a1;
  v52 = *(_QWORD *)a1 + 16LL * *(unsigned int *)(a1 + 8);
  if ( *(_QWORD *)a1 != v52 )
  {
    do
    {
      v53 = *(_QWORD *)(v51 + 8);
      v54 = *(_DWORD *)v51;
      v51 += 16;
      sub_B99FD0(v50, v54, v53);
    }
    while ( v52 != v51 );
  }
  sub_A88F30(a1, v65, *(_QWORD *)(v65 + 56), 1);
  return v67;
}
