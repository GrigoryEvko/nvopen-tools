// Function: sub_1C7DCE0
// Address: 0x1c7dce0
//
__int64 __fastcall sub_1C7DCE0(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __m128 a4,
        double a5,
        double a6,
        double a7,
        double a8,
        double a9,
        double a10,
        __m128 a11)
{
  __int64 *v12; // rdx
  __int64 v13; // rax
  __int64 v14; // rdx
  __int64 v15; // rax
  __int64 *v16; // rdx
  __int64 v17; // r12
  __int64 v18; // rax
  __int64 v19; // rdx
  __int64 v20; // rax
  unsigned int v21; // r12d
  double v22; // xmm4_8
  double v23; // xmm5_8
  __int64 v25[11]; // [rsp+0h] [rbp-130h] BYREF
  _BYTE *v26; // [rsp+58h] [rbp-D8h]
  _BYTE *v27; // [rsp+60h] [rbp-D0h]
  __int64 v28; // [rsp+68h] [rbp-C8h]
  int v29; // [rsp+70h] [rbp-C0h]
  _BYTE v30[32]; // [rsp+78h] [rbp-B8h] BYREF
  __int64 v31; // [rsp+98h] [rbp-98h]
  __int64 v32; // [rsp+A0h] [rbp-90h]
  __int64 v33; // [rsp+A8h] [rbp-88h]
  __int64 v34; // [rsp+B0h] [rbp-80h]
  __int64 v35; // [rsp+B8h] [rbp-78h]
  __int64 v36; // [rsp+C0h] [rbp-70h]
  __int64 v37; // [rsp+C8h] [rbp-68h]
  __int64 v38; // [rsp+D0h] [rbp-60h]
  _BYTE *v39; // [rsp+D8h] [rbp-58h]
  __int64 v40; // [rsp+E0h] [rbp-50h]
  _BYTE v41[32]; // [rsp+E8h] [rbp-48h] BYREF
  __int16 v42; // [rsp+108h] [rbp-28h]

  v12 = *(__int64 **)(a1 + 8);
  v13 = *v12;
  v14 = v12[1];
  if ( v13 == v14 )
LABEL_20:
    BUG();
  while ( *(_UNKNOWN **)v13 != &unk_4F9E06C )
  {
    v13 += 16;
    if ( v14 == v13 )
      goto LABEL_20;
  }
  v15 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v13 + 8) + 104LL))(*(_QWORD *)(v13 + 8), &unk_4F9E06C);
  v16 = *(__int64 **)(a1 + 8);
  v17 = v15 + 160;
  v18 = *v16;
  v19 = v16[1];
  if ( v18 == v19 )
LABEL_19:
    BUG();
  while ( *(_UNKNOWN **)v18 != &unk_4F9920C )
  {
    v18 += 16;
    if ( v19 == v18 )
      goto LABEL_19;
  }
  v20 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v18 + 8) + 104LL))(*(_QWORD *)(v18 + 8), &unk_4F9920C);
  v25[1] = a3;
  v39 = v41;
  v25[2] = v20 + 160;
  v26 = v30;
  v27 = v30;
  v40 = 0x400000000LL;
  v25[3] = v17;
  memset(&v25[4], 0, 56);
  v28 = 4;
  v29 = 0;
  v31 = 0;
  v32 = 0;
  v33 = 0;
  v34 = 0;
  v35 = 0;
  v36 = 0;
  v37 = 0;
  v38 = 0;
  v42 = 0;
  v25[0] = a2;
  v21 = sub_13FCBF0(a2);
  if ( (_BYTE)v21 )
    v21 = sub_1C7B2C0(v25, a4, a5, a6, a7, v22, v23, a10, a11);
  if ( v39 != v41 )
    _libc_free((unsigned __int64)v39);
  if ( v27 != v26 )
    _libc_free((unsigned __int64)v27);
  return v21;
}
