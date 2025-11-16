// Function: sub_346B280
// Address: 0x346b280
//
__int64 __fastcall sub_346B280(__int64 a1, __int64 a2, __int64 a3, __int64 *a4)
{
  __int16 *v8; // rax
  __int64 v9; // rsi
  unsigned __int16 v10; // cx
  __int64 v11; // r8
  unsigned int v12; // r10d
  _BOOL4 v13; // esi
  const __m128i *v14; // rax
  unsigned int v15; // esi
  _BOOL4 v16; // r9d
  __int64 v17; // r14
  __int64 v18; // r15
  __int64 v19; // r9
  unsigned int *v20; // rax
  __int64 v21; // rdx
  unsigned __int8 *v22; // rax
  __int64 result; // rax
  __int64 v24; // rdi
  __int128 v25; // rax
  __int64 v26; // r9
  __int128 v27; // rax
  __int64 v28; // r9
  int v29; // edx
  __int128 v30; // [rsp-10h] [rbp-80h]
  __int128 v31; // [rsp-10h] [rbp-80h]
  __int128 v32; // [rsp-10h] [rbp-80h]
  unsigned int v33; // [rsp+0h] [rbp-70h]
  unsigned int v34; // [rsp+0h] [rbp-70h]
  __int64 v35; // [rsp+8h] [rbp-68h]
  unsigned int v36; // [rsp+8h] [rbp-68h]
  __int64 v37; // [rsp+8h] [rbp-68h]
  unsigned __int16 v38; // [rsp+10h] [rbp-60h]
  __int128 v39; // [rsp+10h] [rbp-60h]
  unsigned __int8 v40; // [rsp+10h] [rbp-60h]
  __int64 v41; // [rsp+30h] [rbp-40h] BYREF
  int v42; // [rsp+38h] [rbp-38h]

  v8 = *(__int16 **)(a2 + 48);
  v9 = *(_QWORD *)(a2 + 80);
  v10 = *v8;
  v11 = *((_QWORD *)v8 + 1);
  v41 = v9;
  v12 = v10;
  if ( v9 )
  {
    v33 = v10;
    v35 = v11;
    v38 = v10;
    sub_B96E90((__int64)&v41, v9, 1);
    v12 = v33;
    v11 = v35;
    v10 = v38;
  }
  v13 = *(_DWORD *)(a2 + 24) != 61;
  v42 = *(_DWORD *)(a2 + 72);
  v14 = *(const __m128i **)(a2 + 40);
  v15 = v13 + 59;
  v16 = *(_DWORD *)(a2 + 24) != 61;
  v17 = v14[2].m128i_i64[1];
  v18 = v14[3].m128i_i64[0];
  v19 = (unsigned int)(v16 + 65);
  v39 = (__int128)_mm_loadu_si128(v14);
  if ( v10 == 1 )
  {
    if ( (*(_BYTE *)(a1 + (unsigned int)v19 + 6914) & 0xFB) == 0 )
    {
LABEL_5:
      v36 = v19;
      v20 = (unsigned int *)sub_33E5110(a4, v12, v11, v12, v11);
      *((_QWORD *)&v30 + 1) = v18;
      *(_QWORD *)&v30 = v17;
      v22 = sub_3411F20(a4, v36, (__int64)&v41, v20, v21, v36, v39, v30);
      *(_DWORD *)(a3 + 8) = 1;
      *(_QWORD *)a3 = v22;
      result = 1;
      goto LABEL_6;
    }
    v24 = 1;
  }
  else
  {
    result = 0;
    if ( !v10 )
      goto LABEL_6;
    v24 = v10;
    if ( !*(_QWORD *)(a1 + 8LL * v10 + 112) )
      goto LABEL_13;
    if ( (*(_BYTE *)((unsigned int)v19 + a1 + 500LL * v10 + 6414) & 0xFB) == 0 )
      goto LABEL_5;
    if ( !*(_QWORD *)(a1 + 8 * (v10 + 14LL)) )
      goto LABEL_13;
  }
  if ( (*(_BYTE *)(v15 + 500 * v24 + a1 + 6414) & 0xFB) != 0 )
  {
LABEL_13:
    result = 0;
    goto LABEL_6;
  }
  *((_QWORD *)&v31 + 1) = v18;
  *(_QWORD *)&v31 = v17;
  v34 = v12;
  v37 = v11;
  *(_QWORD *)&v25 = sub_3406EB0(a4, v15, (__int64)&v41, v12, v11, v19, v39, v31);
  *((_QWORD *)&v32 + 1) = v18;
  *(_QWORD *)&v32 = v17;
  *(_QWORD *)&v27 = sub_3406EB0(a4, 0x3Au, (__int64)&v41, v34, v37, v26, v25, v32);
  *(_QWORD *)a3 = sub_3406EB0(a4, 0x39u, (__int64)&v41, v34, v37, v28, v39, v27);
  *(_DWORD *)(a3 + 8) = v29;
  result = 1;
LABEL_6:
  if ( v41 )
  {
    v40 = result;
    sub_B91220((__int64)&v41, v41);
    return v40;
  }
  return result;
}
