// Function: sub_1D499C0
// Address: 0x1d499c0
//
unsigned __int64 __fastcall sub_1D499C0(__int64 a1, __int64 a2, double a3, double a4, __m128i a5)
{
  __int64 v7; // rsi
  __int64 v8; // rax
  __int64 v9; // r13
  _BYTE *v10; // rdi
  __int64 (__fastcall *v11)(__int64, __int64, _QWORD, __int64, __int64); // r15
  unsigned int v12; // r14d
  __int64 v13; // rax
  int v14; // eax
  __int64 *v15; // rdx
  int v16; // r14d
  __int64 *v17; // r13
  __int64 v18; // rax
  int v19; // edx
  __int64 v20; // r8
  __int64 v21; // rdx
  __int64 v22; // r9
  __int64 *v23; // rax
  __int64 v24; // r13
  __int64 v25; // rdx
  __int64 v26; // rcx
  __int64 v27; // r8
  __int64 v28; // r9
  unsigned __int64 result; // rax
  __int128 v30; // [rsp-10h] [rbp-A0h]
  unsigned __int64 v31; // [rsp-10h] [rbp-A0h]
  __int64 v32; // [rsp+8h] [rbp-88h]
  int v33; // [rsp+14h] [rbp-7Ch]
  __int64 v34; // [rsp+18h] [rbp-78h]
  __int64 v35; // [rsp+20h] [rbp-70h]
  const void ***v36; // [rsp+20h] [rbp-70h]
  __int64 v37; // [rsp+28h] [rbp-68h]
  __int64 v38; // [rsp+30h] [rbp-60h] BYREF
  int v39; // [rsp+38h] [rbp-58h]
  __int64 v40; // [rsp+40h] [rbp-50h] BYREF
  int v41; // [rsp+48h] [rbp-48h]
  _QWORD *v42; // [rsp+50h] [rbp-40h]
  __int64 v43; // [rsp+58h] [rbp-38h]

  v7 = *(_QWORD *)(a2 + 72);
  v38 = v7;
  if ( v7 )
    sub_1623A60((__int64)&v38, v7, 2);
  v39 = *(_DWORD *)(a2 + 64);
  v8 = *(_QWORD *)(*(_QWORD *)(a2 + 32) + 40LL);
  if ( *(_WORD *)(v8 + 24) != 209 )
    BUG();
  v9 = *(_QWORD *)(a1 + 320);
  v35 = *(_QWORD *)(a1 + 272);
  v10 = *(_BYTE **)(*(_QWORD *)(v8 + 88) - 8LL * *(unsigned int *)(*(_QWORD *)(v8 + 88) + 8LL));
  if ( *v10 )
    v10 = 0;
  v11 = *(__int64 (__fastcall **)(__int64, __int64, _QWORD, __int64, __int64))(*(_QWORD *)v9 + 1256LL);
  v12 = **(unsigned __int8 **)(a2 + 40);
  v37 = *(_QWORD *)(*(_QWORD *)(a2 + 40) + 8LL);
  v13 = sub_161E970((__int64)v10);
  v14 = v11(v9, v13, v12, v37, v35);
  v15 = *(__int64 **)(a2 + 32);
  v16 = v14;
  v17 = *(__int64 **)(a1 + 272);
  LODWORD(v11) = **(unsigned __int8 **)(a2 + 40);
  v32 = *v15;
  v33 = *((_DWORD *)v15 + 2);
  v34 = *(_QWORD *)(*(_QWORD *)(a2 + 40) + 8LL);
  v18 = sub_1D252B0((__int64)v17, **(unsigned __int8 **)(a2 + 40), v34, 1, 0);
  LODWORD(v37) = v19;
  v41 = v33;
  v40 = v32;
  v36 = (const void ***)v18;
  v42 = sub_1D2A660(v17, v16, (unsigned int)v11, v34, v20, v34);
  v43 = v21;
  *((_QWORD *)&v30 + 1) = 2;
  *(_QWORD *)&v30 = &v40;
  v23 = sub_1D36D80(v17, 47, (__int64)&v38, v36, v37, a3, a4, a5, v22, v30);
  *((_DWORD *)v23 + 7) = -1;
  v24 = (__int64)v23;
  sub_1D444E0(*(_QWORD *)(a1 + 272), a2, (__int64)v23);
  sub_1D49010(v24);
  sub_1D2DC70(*(const __m128i **)(a1 + 272), a2, v25, v26, v27, v28);
  result = v31;
  if ( v38 )
    return sub_161E7C0((__int64)&v38, v38);
  return result;
}
