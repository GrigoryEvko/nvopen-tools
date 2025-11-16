// Function: sub_3269A00
// Address: 0x3269a00
//
__int64 __fastcall sub_3269A00(_QWORD *a1, __int64 a2)
{
  __int64 v2; // r9
  const __m128i *v3; // rax
  __int64 v4; // r13
  __int64 v5; // rdx
  __int64 v6; // r10
  __int64 v7; // r11
  __int64 v8; // r14
  __int64 v9; // r15
  unsigned __int16 *v10; // rax
  __int64 v11; // r8
  int v12; // ebx
  __int64 v13; // rsi
  __int64 v14; // rax
  __int64 v15; // rsi
  __int64 v16; // r14
  char v18; // al
  int v19; // r8d
  __int64 v20; // r9
  char v21; // al
  __int64 v22; // r13
  __int64 v23; // r10
  __int64 v24; // r11
  int v25; // esi
  __int64 v26; // rax
  char v27; // al
  __int128 v28; // [rsp-30h] [rbp-B0h]
  __int128 v29; // [rsp-20h] [rbp-A0h]
  __int64 v30; // [rsp+10h] [rbp-70h]
  __int64 v31; // [rsp+20h] [rbp-60h]
  __int64 v32; // [rsp+20h] [rbp-60h]
  __int64 v33; // [rsp+28h] [rbp-58h]
  __int128 v34; // [rsp+30h] [rbp-50h]
  int v35; // [rsp+30h] [rbp-50h]
  __int64 v36; // [rsp+40h] [rbp-40h] BYREF
  int v37; // [rsp+48h] [rbp-38h]

  v2 = a2;
  v3 = *(const __m128i **)(a2 + 40);
  v4 = *a1;
  v5 = v3->m128i_i64[0];
  v6 = v3->m128i_i64[0];
  v7 = v3->m128i_i64[1];
  v8 = v3[2].m128i_i64[1];
  v9 = v3[3].m128i_i64[0];
  v34 = (__int128)_mm_loadu_si128(v3 + 5);
  v10 = (unsigned __int16 *)(*(_QWORD *)(v3->m128i_i64[0] + 48) + 16LL * v3->m128i_u32[2]);
  v11 = *((_QWORD *)v10 + 1);
  v12 = *v10;
  if ( *(_DWORD *)(v5 + 24) == 51 || *(_DWORD *)(v8 + 24) == 51 )
  {
    v13 = *(_QWORD *)(a2 + 80);
    v36 = v13;
    if ( !v13 )
      goto LABEL_6;
    v31 = v2;
    v35 = v11;
    goto LABEL_5;
  }
  v30 = *((_QWORD *)v10 + 1);
  v32 = v6;
  v33 = v7;
  v18 = sub_33E2390(v4, v6, v7, 1);
  v19 = v30;
  v20 = a2;
  if ( v18 )
  {
    v21 = sub_33E2390(*a1, v8, v9, 1);
    v19 = v30;
    v20 = a2;
    if ( !v21 )
    {
      v22 = *a1;
      v23 = v32;
      v24 = v33;
      v36 = *(_QWORD *)(a2 + 80);
      if ( v36 )
      {
        sub_B96E90((__int64)&v36, v36, 1);
        v20 = a2;
        v23 = v32;
        v24 = v33;
        v19 = v30;
      }
      v25 = *(_DWORD *)(v20 + 24);
      *((_QWORD *)&v29 + 1) = v24;
      *(_QWORD *)&v29 = v23;
      *((_QWORD *)&v28 + 1) = v9;
      *(_QWORD *)&v28 = v8;
      v37 = *(_DWORD *)(v20 + 72);
      v26 = sub_340F900(v22, v25, (unsigned int)&v36, v12, v19, v20, v28, v29, v34);
      v15 = v36;
      v16 = v26;
      if ( v36 )
        goto LABEL_7;
      return v16;
    }
  }
  v31 = v20;
  v35 = v19;
  v27 = sub_33CF170(v8, v9);
  LODWORD(v11) = v35;
  v2 = v31;
  if ( !v27 )
    return 0;
  v13 = *(_QWORD *)(v31 + 80);
  v4 = *a1;
  v36 = v13;
  if ( v13 )
  {
LABEL_5:
    sub_B96E90((__int64)&v36, v13, 1);
    v2 = v31;
    LODWORD(v11) = v35;
  }
LABEL_6:
  v37 = *(_DWORD *)(v2 + 72);
  v14 = sub_3400BD0(v4, 0, (unsigned int)&v36, v12, v11, 0, 0);
  v15 = v36;
  v16 = v14;
  if ( v36 )
LABEL_7:
    sub_B91220((__int64)&v36, v15);
  return v16;
}
