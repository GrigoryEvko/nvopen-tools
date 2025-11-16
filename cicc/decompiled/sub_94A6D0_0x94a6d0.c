// Function: sub_94A6D0
// Address: 0x94a6d0
//
__int64 __fastcall sub_94A6D0(__int64 a1, __int64 a2, int a3, unsigned __int64 *a4)
{
  __int64 v7; // rsi
  __int64 v8; // rbx
  __m128i *v9; // rdx
  unsigned __int64 i; // rax
  __int64 v11; // rsi
  unsigned int **v12; // r14
  __int64 v13; // r12
  __int64 v14; // rax
  __m128i *v15; // rax
  __int64 v16; // rdi
  __int64 v17; // rbx
  char v18; // al
  int v19; // eax
  __int64 v20; // rbx
  __int64 v21; // rax
  __int64 v22; // rax
  __int64 v23; // rdx
  unsigned __int64 v24; // rax
  char v25; // r8
  __int64 v26; // rax
  __int64 v27; // r12
  unsigned int *v28; // rax
  __int64 v29; // rcx
  unsigned int *v30; // rbx
  __int64 v31; // rdx
  __int64 v32; // rsi
  __int64 v33; // r13
  __int64 v34; // rdi
  unsigned int v35; // ebx
  int v36; // ebx
  unsigned int v37; // esi
  __int64 v38; // rax
  unsigned int v40; // [rsp+8h] [rbp-98h]
  int v41; // [rsp+8h] [rbp-98h]
  unsigned __int8 v42; // [rsp+8h] [rbp-98h]
  _DWORD *v45; // [rsp+18h] [rbp-88h]
  __int64 v47; // [rsp+28h] [rbp-78h]
  int v48; // [rsp+28h] [rbp-78h]
  unsigned int *v49; // [rsp+28h] [rbp-78h]
  unsigned int v50; // [rsp+38h] [rbp-68h]
  _QWORD v51[4]; // [rsp+40h] [rbp-60h] BYREF
  __int16 v52; // [rsp+60h] [rbp-40h]

  v7 = *(_QWORD *)(a4[9] + 16);
  v8 = *(_QWORD *)(v7 + 16);
  v9 = sub_92F410(a2, v7);
  for ( i = *a4; *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
    ;
  v47 = (__int64)v9;
  v11 = *(_DWORD *)(v9->m128i_i64[1] + 8) >> 8;
  v12 = (unsigned int **)(a2 + 48);
  v13 = sub_BCCE00(*(_QWORD *)(a2 + 40), 8LL * *(_QWORD *)(i + 128));
  v14 = sub_BCE770(v13, v11);
  v52 = 257;
  v48 = sub_949E90((unsigned int **)(a2 + 48), 0x31u, v47, v14, (__int64)v51, 0, v50, 0);
  v45 = (_DWORD *)a4 + 9;
  v15 = sub_92F410(a2, v8);
  v16 = v15->m128i_i64[1];
  v17 = (__int64)v15;
  v18 = *(_BYTE *)(v16 + 8);
  if ( v18 == 14 )
  {
    v20 = sub_92C9E0(a2, v17, 0, v13, 0, 0, v45);
  }
  else
  {
    if ( v18 != 12 )
      sub_91B8A0("unexpected: a non-integer and non-pointer type was used with atomic builtin!", v45, 1);
    v40 = sub_BCB060(v16);
    if ( v40 > (unsigned int)sub_BCB060(v13) )
      sub_91B8A0("unexpected: Integer type too small!", v45, 1);
    v52 = 257;
    v41 = sub_BCB060(*(_QWORD *)(v17 + 8));
    v19 = sub_BCB060(v13);
    v20 = sub_949E90((unsigned int **)(a2 + 48), 9 * (unsigned int)(v41 == v19) + 40, v17, v13, (__int64)v51, 0, v50, 0);
  }
  v21 = sub_AA4E30(*(_QWORD *)(a2 + 96));
  v22 = sub_9208B0(v21, *(_QWORD *)(v20 + 8));
  v51[1] = v23;
  v51[0] = (unsigned __int64)(v22 + 7) >> 3;
  v24 = sub_CA1930(v51);
  v25 = -1;
  if ( v24 )
  {
    _BitScanReverse64(&v24, v24);
    v25 = 63 - (v24 ^ 0x3F);
  }
  v42 = v25;
  v52 = 257;
  v26 = sub_BD2C40(80, unk_3F148C0);
  v27 = v26;
  if ( v26 )
    sub_B4D750(v26, a3, v48, v20, v42, 2, 1, 0, 0);
  (*(void (__fastcall **)(_QWORD, __int64, _QWORD *, _QWORD, _QWORD))(**(_QWORD **)(a2 + 136) + 16LL))(
    *(_QWORD *)(a2 + 136),
    v27,
    v51,
    *(_QWORD *)(a2 + 104),
    *(_QWORD *)(a2 + 112));
  v28 = *(unsigned int **)(a2 + 48);
  v29 = (__int64)&v28[4 * *(unsigned int *)(a2 + 56)];
  v30 = v28;
  v49 = (unsigned int *)v29;
  while ( v49 != v30 )
  {
    v31 = *((_QWORD *)v30 + 1);
    v32 = *v30;
    v30 += 4;
    sub_B99FD0(v27, v32, v31);
  }
  v33 = sub_91A390(*(_QWORD *)(a2 + 32) + 8LL, *a4, 0, v29);
  if ( *(_BYTE *)(v33 + 8) == 14 )
  {
    v52 = 257;
    v38 = sub_949E90(v12, 0x30u, v27, v33, (__int64)v51, 0, v50, 0);
  }
  else
  {
    v34 = *(_QWORD *)(v27 + 8);
    if ( *(_BYTE *)(v34 + 8) != 12 )
      sub_91B8A0("unexpected: a non-integer and non-pointer type was used with atomic builtin!", v45, 1);
    v35 = sub_BCB060(v34);
    if ( v35 < (unsigned int)sub_BCB060(v33) )
      sub_91B8A0("unexpected: Integer type too small!", v45, 1);
    v52 = 257;
    v36 = sub_BCB060(*(_QWORD *)(v27 + 8));
    v37 = 49;
    if ( v36 != (unsigned int)sub_BCB060(v33) )
      v37 = 38;
    v38 = sub_949E90(v12, v37, v27, v33, (__int64)v51, 0, v50, 0);
  }
  *(_QWORD *)a1 = v38;
  *(_BYTE *)(a1 + 12) &= ~1u;
  *(_DWORD *)(a1 + 8) = 0;
  *(_DWORD *)(a1 + 16) = 0;
  return a1;
}
