// Function: sub_330AAF0
// Address: 0x330aaf0
//
__int64 __fastcall sub_330AAF0(_QWORD *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  const __m128i *v7; // rax
  __int64 v8; // rsi
  __int128 v9; // xmm0
  __int64 v10; // r12
  __int64 v11; // r13
  unsigned __int16 *v12; // rax
  __int64 v13; // rcx
  int v14; // r15d
  int v15; // r9d
  int v16; // eax
  __int64 v17; // r9
  int v18; // edx
  int v19; // eax
  __int64 result; // rax
  int v21; // r8d
  __int64 v22; // rax
  int v23; // edx
  int v24; // r9d
  unsigned __int32 v25; // edx
  __int128 v26; // [rsp-20h] [rbp-B0h]
  __int128 v27; // [rsp-10h] [rbp-A0h]
  unsigned __int32 v28; // [rsp+0h] [rbp-90h]
  int v29; // [rsp+8h] [rbp-88h]
  int v30; // [rsp+10h] [rbp-80h]
  __int64 v31; // [rsp+18h] [rbp-78h]
  __int64 v32; // [rsp+18h] [rbp-78h]
  __int64 v33; // [rsp+20h] [rbp-70h]
  __int64 v34; // [rsp+30h] [rbp-60h] BYREF
  int v35; // [rsp+38h] [rbp-58h]
  __int64 v36; // [rsp+40h] [rbp-50h] BYREF
  unsigned __int32 v37; // [rsp+48h] [rbp-48h]
  __int64 v38; // [rsp+50h] [rbp-40h]
  int v39; // [rsp+58h] [rbp-38h]

  v7 = *(const __m128i **)(a2 + 40);
  v8 = *(_QWORD *)(a2 + 80);
  v9 = (__int128)_mm_loadu_si128(v7);
  v10 = v7[2].m128i_i64[1];
  v11 = v7[3].m128i_i64[0];
  v31 = v7->m128i_i64[0];
  v28 = v7->m128i_u32[2];
  v12 = (unsigned __int16 *)(*(_QWORD *)(v7->m128i_i64[0] + 48) + 16LL * v28);
  v13 = *((_QWORD *)v12 + 1);
  v14 = *v12;
  v34 = v8;
  v30 = v13;
  if ( v8 )
    sub_B96E90((__int64)&v34, v8, 1);
  v35 = *(_DWORD *)(a2 + 72);
  if ( !(unsigned __int8)sub_33CF8A0(a2, 1, a3, v13, a5, a6) )
  {
LABEL_15:
    v22 = sub_33F17F0(*a1, 67, &v34, 262, 0);
    *((_QWORD *)&v27 + 1) = v11;
    *(_QWORD *)&v27 = v10;
    v29 = v23;
    v32 = v22;
    v36 = sub_3406EB0(*a1, 56, (unsigned int)&v34, v14, v30, v24, v9, v27);
    v37 = v25;
    v38 = v32;
    v39 = v29;
    goto LABEL_16;
  }
  v16 = *(_DWORD *)(v31 + 24);
  if ( v16 == 35 || v16 == 11 )
  {
    v19 = *(_DWORD *)(v10 + 24);
    if ( v19 != 35 && v19 != 11 )
    {
      *((_QWORD *)&v26 + 1) = v11;
      *(_QWORD *)&v26 = v10;
      result = sub_3411F20(*a1, 68, (unsigned int)&v34, *(_QWORD *)(a2 + 48), *(_DWORD *)(a2 + 68), v15, v26, v9);
      goto LABEL_11;
    }
  }
  if ( !(unsigned __int8)sub_33CF170(v10, v11) )
  {
    v21 = sub_33DD440(*a1, v9, *((_QWORD *)&v9 + 1), v10, v11, v17);
    result = 0;
    if ( v21 )
      goto LABEL_11;
    goto LABEL_15;
  }
  v38 = sub_33F17F0(*a1, 67, &v34, 262, 0);
  v36 = v31;
  v39 = v18;
  v37 = v28;
LABEL_16:
  result = sub_32EB790((__int64)a1, a2, &v36, 2, 1);
LABEL_11:
  if ( v34 )
  {
    v33 = result;
    sub_B91220((__int64)&v34, v34);
    return v33;
  }
  return result;
}
