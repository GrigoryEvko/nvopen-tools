// Function: sub_326B270
// Address: 0x326b270
//
__int64 __fastcall sub_326B270(__int64 a1, int a2, int a3)
{
  __int64 *v3; // rax
  __int64 v4; // r12
  __int64 v5; // r13
  int v6; // ebx
  __int64 v7; // r14
  unsigned __int16 *v9; // rax
  unsigned __int16 v10; // cx
  __int64 v11; // r8
  __int64 v12; // rax
  int v13; // edx
  const __m128i *v14; // rdx
  __int64 v15; // r15
  __int64 v16; // rbx
  __int64 v17; // rax
  __int64 v18; // rdi
  __int64 v19; // rsi
  int v20; // eax
  __int64 v21; // rax
  __int128 v22; // rax
  int v23; // r9d
  __int128 v24; // rax
  int v25; // r9d
  int v26; // eax
  __int64 v27; // rdx
  __int128 v28; // rax
  int v29; // r9d
  __int128 v30; // rax
  int v31; // r9d
  __int128 v32; // [rsp-10h] [rbp-80h]
  __int128 v33; // [rsp-10h] [rbp-80h]
  int v34; // [rsp+8h] [rbp-68h]
  int v35; // [rsp+8h] [rbp-68h]
  int v36; // [rsp+10h] [rbp-60h]
  int v37; // [rsp+10h] [rbp-60h]
  __int64 v39; // [rsp+20h] [rbp-50h]
  __int128 v40; // [rsp+20h] [rbp-50h]
  __int64 v42; // [rsp+38h] [rbp-38h]

  v3 = *(__int64 **)(a1 + 40);
  v4 = v3[5];
  v5 = v3[6];
  v6 = *((_DWORD *)v3 + 2);
  v7 = v3[10];
  v39 = *v3;
  v42 = v3[11];
  if ( !(unsigned __int8)sub_326A930(v4, v5, 0) )
    return 0;
  if ( !(unsigned __int8)sub_326A930(v7, v42, 0) )
    return 0;
  v9 = *(unsigned __int16 **)(a1 + 48);
  v10 = *v9;
  v11 = *((_QWORD *)v9 + 1);
  if ( *(_DWORD *)(v39 + 24) != 208 )
    return 0;
  v12 = *(_QWORD *)(v39 + 56);
  if ( !v12 )
    return 0;
  v13 = 1;
  do
  {
    if ( *(_DWORD *)(v12 + 8) == v6 )
    {
      if ( !v13 )
        return 0;
      v12 = *(_QWORD *)(v12 + 32);
      if ( !v12 )
        goto LABEL_14;
      if ( *(_DWORD *)(v12 + 8) == v6 )
        return 0;
      v13 = 0;
    }
    v12 = *(_QWORD *)(v12 + 32);
  }
  while ( v12 );
  if ( v13 == 1 )
    return 0;
LABEL_14:
  v14 = *(const __m128i **)(v39 + 40);
  v15 = v14->m128i_i64[0];
  v16 = v14->m128i_u32[2];
  v17 = *(_QWORD *)(v14->m128i_i64[0] + 48) + 16 * v16;
  if ( *(_WORD *)v17 != v10 || *(_QWORD *)(v17 + 8) != v11 && !v10 )
    return 0;
  v18 = v14[2].m128i_i64[1];
  v19 = v14[3].m128i_i64[0];
  v20 = *(_DWORD *)(v14[5].m128i_i64[0] + 96);
  v40 = (__int128)_mm_loadu_si128(v14);
  if ( v20 != 18 )
  {
    if ( v20 == 20 )
    {
      v34 = v10;
      v36 = v11;
      if ( (unsigned __int8)sub_33E0720(v18, v19, 0) )
      {
        if ( (unsigned __int8)sub_33E0720(v7, v42, 0) )
        {
          v21 = sub_3263630(v15, v16);
          *(_QWORD *)&v22 = sub_3400BD0(a3, (int)v21 - 1, a2, v34, v36, 0, 0, v21);
          *(_QWORD *)&v24 = sub_3406EB0(a3, 191, a2, v34, v36, v23, v40, v22);
          *((_QWORD *)&v32 + 1) = v5;
          *(_QWORD *)&v32 = v4;
          return sub_3406EB0(a3, 186, a2, v34, v36, v25, v24, v32);
        }
      }
    }
    return 0;
  }
  v35 = v10;
  v37 = v11;
  if ( !(unsigned __int8)sub_33E07E0(v18, v19, 0) || !(unsigned __int8)sub_33E07E0(v7, v42, 0) )
    return 0;
  v26 = sub_3263630(v15, v16);
  *(_QWORD *)&v28 = sub_3400BD0(a3, v26 - 1, a2, v35, v37, 0, 0, v27);
  *(_QWORD *)&v30 = sub_3406EB0(a3, 191, a2, v35, v37, v29, v40, v28);
  *((_QWORD *)&v33 + 1) = v5;
  *(_QWORD *)&v33 = v4;
  return sub_3406EB0(a3, 187, a2, v35, v37, v31, v30, v33);
}
