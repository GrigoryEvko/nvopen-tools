// Function: sub_19FCAE0
// Address: 0x19fcae0
//
__int64 __fastcall sub_19FCAE0(
        __int64 a1,
        __int64 a2,
        __m128 a3,
        __m128i a4,
        __m128 a5,
        __m128i a6,
        double a7,
        double a8,
        double a9,
        __m128 a10)
{
  __int64 v10; // rax
  __int64 *v11; // rcx
  __int64 v12; // rbx
  __int64 v13; // rax
  __int64 v14; // rcx
  __int64 v15; // rax
  __int64 *v16; // rcx
  __int64 v17; // rax
  __int64 v18; // rcx
  __int64 v19; // rax
  __int64 *v20; // rcx
  __int64 v21; // r13
  __int64 v22; // rax
  __int64 v23; // rcx
  __int64 v24; // rax
  __int64 *v25; // rcx
  __int64 v26; // r15
  __int64 v27; // rax
  __int64 v28; // rcx
  __int64 v29; // rax
  __int64 v30; // rax
  __int64 *v31; // rdx
  __int64 v32; // rcx
  __int64 v33; // rax
  __int64 v34; // rdx
  __int64 v35; // rax
  double v36; // xmm4_8
  double v37; // xmm5_8
  unsigned int v38; // r12d
  __int64 v40; // [rsp+0h] [rbp-B40h]
  __int64 v41; // [rsp+8h] [rbp-B38h]
  _BYTE v42[2864]; // [rsp+10h] [rbp-B30h] BYREF

  v10 = sub_1632FA0(*(_QWORD *)(a2 + 40));
  v11 = *(__int64 **)(a1 + 8);
  v12 = v10;
  v13 = *v11;
  v14 = v11[1];
  if ( v13 == v14 )
LABEL_35:
    BUG();
  while ( *(_UNKNOWN **)v13 != &unk_4F99768 )
  {
    v13 += 16;
    if ( v14 == v13 )
      goto LABEL_35;
  }
  v15 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v13 + 8) + 104LL))(*(_QWORD *)(v13 + 8), &unk_4F99768);
  v16 = *(__int64 **)(a1 + 8);
  v40 = *(_QWORD *)(v15 + 160);
  v17 = *v16;
  v18 = v16[1];
  if ( v17 == v18 )
LABEL_31:
    BUG();
  while ( *(_UNKNOWN **)v17 != &unk_4F96DB4 )
  {
    v17 += 16;
    if ( v18 == v17 )
      goto LABEL_31;
  }
  v19 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v17 + 8) + 104LL))(*(_QWORD *)(v17 + 8), &unk_4F96DB4);
  v20 = *(__int64 **)(a1 + 8);
  v21 = *(_QWORD *)(v19 + 160);
  v22 = *v20;
  v23 = v20[1];
  if ( v22 == v23 )
LABEL_32:
    BUG();
  while ( *(_UNKNOWN **)v22 != &unk_4F9B6E8 )
  {
    v22 += 16;
    if ( v23 == v22 )
      goto LABEL_32;
  }
  v24 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v22 + 8) + 104LL))(*(_QWORD *)(v22 + 8), &unk_4F9B6E8);
  v25 = *(__int64 **)(a1 + 8);
  v26 = v24 + 360;
  v27 = *v25;
  v28 = v25[1];
  if ( v27 == v28 )
LABEL_33:
    BUG();
  while ( *(_UNKNOWN **)v27 != &unk_4F9D764 )
  {
    v27 += 16;
    if ( v28 == v27 )
      goto LABEL_33;
  }
  v29 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v27 + 8) + 104LL))(*(_QWORD *)(v27 + 8), &unk_4F9D764);
  v30 = sub_14CF090(v29, a2);
  v31 = *(__int64 **)(a1 + 8);
  v32 = v30;
  v33 = *v31;
  v34 = v31[1];
  if ( v33 == v34 )
LABEL_34:
    BUG();
  while ( *(_UNKNOWN **)v33 != &unk_4F9E06C )
  {
    v33 += 16;
    if ( v34 == v33 )
      goto LABEL_34;
  }
  v41 = v32;
  v35 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v33 + 8) + 104LL))(*(_QWORD *)(v33 + 8), &unk_4F9E06C);
  sub_19E61F0((__int64)v42, a2, v35 + 160, v41, v26, v21, v40, v12);
  v38 = sub_19F99A0((__int64)v42, a3, a4, a5, a6, v36, v37, a9, a10);
  sub_19E28F0((__int64)v42);
  return v38;
}
