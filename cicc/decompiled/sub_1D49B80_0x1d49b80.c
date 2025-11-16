// Function: sub_1D49B80
// Address: 0x1d49b80
//
void __fastcall sub_1D49B80(__int64 a1, __int64 a2, __m128 a3, double a4, __m128i a5)
{
  __int64 v7; // rsi
  __int64 v8; // rax
  __int64 v9; // rdx
  __int64 v10; // r13
  __int64 v11; // r15
  _BYTE *v12; // rdi
  __int64 (__fastcall *v13)(__int64, __int64, _QWORD, __int64, __int64); // r14
  unsigned __int8 *v14; // rax
  __int64 v15; // rax
  int v16; // eax
  __int64 *v17; // r13
  int v18; // esi
  __int64 *v19; // rax
  __int64 v20; // r14
  __int64 v21; // r15
  __int128 v22; // rax
  __int64 *v23; // rax
  __int64 v24; // r13
  __int64 v25; // rdx
  __int64 v26; // rcx
  __int64 v27; // r8
  __int64 v28; // r9
  __int64 v29; // [rsp+8h] [rbp-58h]
  unsigned int v30; // [rsp+10h] [rbp-50h]
  unsigned __int64 v31; // [rsp+10h] [rbp-50h]
  __int16 *v32; // [rsp+18h] [rbp-48h]
  __int64 v33; // [rsp+20h] [rbp-40h] BYREF
  int v34; // [rsp+28h] [rbp-38h]

  v7 = *(_QWORD *)(a2 + 72);
  v33 = v7;
  if ( v7 )
    sub_1623A60((__int64)&v33, v7, 2);
  v34 = *(_DWORD *)(a2 + 64);
  v8 = *(_QWORD *)(a2 + 32);
  v9 = *(_QWORD *)(v8 + 40);
  if ( *(_WORD *)(v9 + 24) != 209 )
    BUG();
  v10 = *(_QWORD *)(a1 + 320);
  v11 = *(_QWORD *)(a1 + 272);
  v12 = *(_BYTE **)(*(_QWORD *)(v9 + 88) - 8LL * *(unsigned int *)(*(_QWORD *)(v9 + 88) + 8LL));
  if ( *v12 )
    v12 = 0;
  v13 = *(__int64 (__fastcall **)(__int64, __int64, _QWORD, __int64, __int64))(*(_QWORD *)v10 + 1256LL);
  v14 = (unsigned __int8 *)(*(_QWORD *)(*(_QWORD *)(v8 + 80) + 40LL) + 16LL * *(unsigned int *)(v8 + 88));
  v29 = *((_QWORD *)v14 + 1);
  v30 = *v14;
  v15 = sub_161E970((__int64)v12);
  v16 = v13(v10, v15, v30, v29, v11);
  v17 = *(__int64 **)(a1 + 272);
  v18 = v16;
  v19 = *(__int64 **)(a2 + 32);
  v20 = v19[10];
  v21 = v19[11];
  v31 = *v19;
  v32 = (__int16 *)v19[1];
  *(_QWORD *)&v22 = sub_1D2A660(
                      v17,
                      v18,
                      *(unsigned __int8 *)(*(_QWORD *)(v20 + 40) + 16LL * *((unsigned int *)v19 + 22)),
                      *(_QWORD *)(*(_QWORD *)(v20 + 40) + 16LL * *((unsigned int *)v19 + 22) + 8),
                      *v19,
                      (__int64)v32);
  v23 = sub_1D3A900(v17, 0x2Eu, (__int64)&v33, 1u, 0, 0, a3, a4, a5, v31, v32, v22, v20, v21);
  *((_DWORD *)v23 + 7) = -1;
  v24 = (__int64)v23;
  sub_1D444E0(*(_QWORD *)(a1 + 272), a2, (__int64)v23);
  sub_1D49010(v24);
  sub_1D2DC70(*(const __m128i **)(a1 + 272), a2, v25, v26, v27, v28);
  if ( v33 )
    sub_161E7C0((__int64)&v33, v33);
}
