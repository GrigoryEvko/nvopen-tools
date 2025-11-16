// Function: sub_134AE90
// Address: 0x134ae90
//
_QWORD *__fastcall sub_134AE90(__int64 a1, __int64 a2, _QWORD *a3, __int64 *a4, _QWORD *a5, _QWORD *a6, _QWORD *a7)
{
  __int64 v9; // rbx
  __int64 v10; // rbx
  __int64 v11; // rbx
  unsigned int v12; // ebx
  __int64 v13; // r15
  __int64 *v14; // rcx
  __int64 v15; // r12
  __int64 v16; // r12
  __int64 v17; // r12
  __int64 v18; // r14
  __int64 v19; // r14
  __int64 v20; // r13
  __int64 v21; // r13
  __int64 v22; // rax
  unsigned int v23; // esi
  __int64 v24; // r12
  __int64 v25; // rax
  _QWORD *result; // rax
  __int64 v30; // [rsp+28h] [rbp-78h]
  __int64 v31; // [rsp+30h] [rbp-70h]
  __int64 *v32; // [rsp+38h] [rbp-68h]
  __int64 v33; // [rsp+40h] [rbp-60h]
  __int64 v34; // [rsp+48h] [rbp-58h]
  __int64 v35; // [rsp+50h] [rbp-50h]
  __int64 v36; // [rsp+58h] [rbp-48h]
  __int64 v37; // [rsp+60h] [rbp-40h]
  __int64 v38; // [rsp+68h] [rbp-38h]

  v37 = a2 + 39072;
  v9 = sub_13427E0(a2 + 39072);
  v36 = a2 + 48728;
  a3[7] += (sub_13427E0(a2 + 48728) + v9) << 12;
  *a3 += *(_QWORD *)(a2 + 68112);
  v10 = *(_QWORD *)(a2 + 8);
  v31 = a2 + 192;
  v11 = sub_13427E0(a2 + 192) + v10;
  v38 = a2 + 9848;
  *a7 += (sub_13427E0(a2 + 9848) + v11) << 12;
  v12 = 0;
  v13 = a2 + 29288;
  a3[1] += **(_QWORD **)(a2 + 62248);
  a3[2] += *(_QWORD *)(*(_QWORD *)(a2 + 62248) + 8LL);
  a3[3] += *(_QWORD *)(*(_QWORD *)(a2 + 62248) + 16LL);
  a3[4] += *(_QWORD *)(*(_QWORD *)(a2 + 62248) + 24LL);
  a3[5] += *(_QWORD *)(*(_QWORD *)(a2 + 62248) + 32LL);
  a3[6] += *(_QWORD *)(*(_QWORD *)(a2 + 62248) + 40LL);
  a3[9] += *(_QWORD *)(*(_QWORD *)(a2 + 62248) + 64LL);
  v14 = a4;
  v30 = a2 + 19632;
  do
  {
    v32 = v14;
    v15 = sub_13427F0(v31, v12);
    v33 = v15 + sub_13427F0(v38, v12);
    v16 = sub_13427F0(v30, v12);
    v34 = v16 + sub_13427F0(v13, v12);
    v17 = sub_13427F0(v37, v12);
    v35 = v17 + sub_13427F0(v36, v12);
    v18 = sub_1342810(v31, v12);
    v19 = sub_1342810(v38, v12) + v18;
    v20 = sub_1342810(v30, v12);
    v21 = sub_1342810(v13, v12) + v20;
    v22 = sub_1342810(v37, v12);
    v23 = v12++;
    v24 = v22;
    v25 = sub_1342810(v36, v23);
    *v32 = v33;
    v14 = v32 + 6;
    v32[1] = v19;
    v32[2] = v34;
    v32[3] = v21;
    v32[4] = v35;
    v32[5] = v25 + v24;
  }
  while ( v12 != 199 );
  result = (_QWORD *)a2;
  if ( *(_BYTE *)(a2 + 17) )
  {
    sub_1348660(a1, a2 + 62384, a5);
    return sub_130EDA0(a1, a2 + 62264, a6);
  }
  return result;
}
