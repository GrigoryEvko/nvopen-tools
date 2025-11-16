// Function: sub_2FEF750
// Address: 0x2fef750
//
__int64 __fastcall sub_2FEF750(__int64 a1)
{
  __int128 v2; // kr00_16
  unsigned __int64 v3; // rsi
  __int128 v4; // kr10_16
  unsigned __int64 v5; // rsi
  __int64 v6; // r15
  unsigned __int64 v7; // rsi
  unsigned __int64 v8; // rdx
  __int64 v9; // r14
  __int128 v10; // kr20_16
  __int64 v11; // rax
  __int64 v12; // rax
  __int128 v13; // rdi
  __int64 v14; // rax
  __int64 result; // rax
  __int64 v16; // rcx
  __int64 v17; // r8
  __int64 v18; // r9
  __int64 v19; // rdx
  const char *v20; // rax
  __int64 v21; // rcx
  __int64 v22; // r8
  __int64 v23; // r9
  __int64 v24; // rcx
  __int64 v25; // r8
  __int64 v26; // r9
  __m128i v27[2]; // [rsp+30h] [rbp-180h] BYREF
  char v28; // [rsp+50h] [rbp-160h]
  char v29; // [rsp+51h] [rbp-15Fh]
  __m128i v30[2]; // [rsp+60h] [rbp-150h] BYREF
  char v31; // [rsp+80h] [rbp-130h]
  char v32; // [rsp+81h] [rbp-12Fh]
  __m128i v33[3]; // [rsp+90h] [rbp-120h] BYREF
  __m128i v34[2]; // [rsp+C0h] [rbp-F0h] BYREF
  char v35; // [rsp+E0h] [rbp-D0h]
  char v36; // [rsp+E1h] [rbp-CFh]
  __m128i v37[3]; // [rsp+F0h] [rbp-C0h] BYREF
  __m128i v38[2]; // [rsp+120h] [rbp-90h] BYREF
  char v39; // [rsp+140h] [rbp-70h]
  char v40; // [rsp+141h] [rbp-6Fh]
  __int128 v41; // [rsp+150h] [rbp-60h] BYREF
  int v42; // [rsp+160h] [rbp-50h]

  sub_2FEE6E0((__int64)&v41, qword_5027888, qword_5027890);
  v2 = v41;
  v3 = qword_5027988;
  *(_DWORD *)(a1 + 216) = v42;
  sub_2FEE6E0((__int64)&v41, v3, qword_5027990);
  v4 = v41;
  v5 = qword_5027688;
  *(_DWORD *)(a1 + 224) = v42;
  sub_2FEE6E0((__int64)&v41, v5, qword_5027690);
  v6 = v41;
  v7 = qword_5027788;
  v8 = qword_5027790;
  *(_DWORD *)(a1 + 232) = v42;
  v9 = *((_QWORD *)&v41 + 1);
  sub_2FEE6E0((__int64)&v41, v7, v8);
  v10 = v41;
  *(_DWORD *)(a1 + 240) = v42;
  v11 = sub_2FEF600(v2);
  if ( v11 )
    v11 = *(_QWORD *)(v11 + 32);
  *(_QWORD *)(a1 + 184) = v11;
  v12 = sub_2FEF600(v4);
  if ( v12 )
    v12 = *(_QWORD *)(v12 + 32);
  *(_QWORD *)(a1 + 192) = v12;
  *(_QWORD *)&v13 = v6;
  *((_QWORD *)&v13 + 1) = v9;
  v14 = sub_2FEF600(v13);
  if ( v14 )
    v14 = *(_QWORD *)(v14 + 32);
  *(_QWORD *)(a1 + 200) = v14;
  result = sub_2FEF600(v10);
  if ( result )
  {
    result = *(_QWORD *)(result + 32);
    v19 = *(_QWORD *)(a1 + 184);
    *(_QWORD *)(a1 + 208) = result;
    if ( !v19 )
      goto LABEL_10;
  }
  else
  {
    v19 = *(_QWORD *)(a1 + 184);
    *(_QWORD *)(a1 + 208) = 0;
    if ( !v19 )
      goto LABEL_12;
  }
  if ( *(_QWORD *)(a1 + 192) )
  {
    v40 = 1;
    v38[0].m128i_i64[0] = (__int64)" specified!";
    v34[0].m128i_i64[0] = (__int64)"start-after";
    v30[0].m128i_i64[0] = (__int64)" and ";
    v20 = "start-before";
    v39 = 3;
    v36 = 1;
    v35 = 3;
    v32 = 1;
    v31 = 3;
    v29 = 1;
LABEL_16:
    v27[0].m128i_i64[0] = (__int64)v20;
    v28 = 3;
    sub_9C6370(v33, v27, v30, v16, v17, v18);
    sub_9C6370(v37, v33, v34, v21, v22, v23);
    sub_9C6370((__m128i *)&v41, v37, v38, v24, v25, v26);
    sub_C64D30((__int64)&v41, 1u);
  }
LABEL_10:
  if ( *(_QWORD *)(a1 + 200) && result )
  {
    v40 = 1;
    v38[0].m128i_i64[0] = (__int64)" specified!";
    v39 = 3;
    v36 = 1;
    v35 = 3;
    v32 = 1;
    v31 = 3;
    v29 = 1;
    v34[0].m128i_i64[0] = (__int64)"stop-after";
    v30[0].m128i_i64[0] = (__int64)" and ";
    v20 = "stop-before";
    goto LABEL_16;
  }
LABEL_12:
  *(_BYTE *)(a1 + 248) = (*(_QWORD *)(a1 + 192) | v19) == 0;
  return result;
}
