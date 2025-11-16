// Function: sub_3401190
// Address: 0x3401190
//
unsigned __int8 *__fastcall sub_3401190(
        _QWORD *a1,
        __int64 a2,
        unsigned int a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __m128i a7)
{
  _QWORD *v7; // r12
  __int64 v10; // rcx
  __int64 v11; // r8
  unsigned __int8 *result; // rax
  unsigned __int16 *v13; // rbx
  __int64 v14; // r8
  unsigned int v15; // ecx
  unsigned __int16 *v16; // rbx
  const __m128i *v17; // rsi
  __int64 v18; // r8
  unsigned int v19; // r15d
  const __m128i **v20; // rsi
  char v21; // cl
  unsigned __int8 v22; // r8
  _QWORD *v23; // r9
  char v24; // r15
  unsigned __int16 *v25; // rax
  __int16 v26; // dx
  __int64 v27; // rax
  bool v28; // zf
  __int64 v30; // [rsp+8h] [rbp-78h]
  unsigned __int8 *v31; // [rsp+8h] [rbp-78h]
  __m128i v32; // [rsp+10h] [rbp-70h] BYREF
  __int64 (__fastcall *v33)(const __m128i **, const __m128i *, int); // [rsp+20h] [rbp-60h]
  bool (__fastcall *v34)(unsigned int *, __int64); // [rsp+28h] [rbp-58h]
  const __m128i *v35; // [rsp+30h] [rbp-50h] BYREF
  __int64 v36; // [rsp+38h] [rbp-48h]
  __int64 (__fastcall *v37)(const __m128i **, const __m128i *, int); // [rsp+40h] [rbp-40h]
  bool (__fastcall *v38)(unsigned int *, __int64); // [rsp+48h] [rbp-38h]

  v7 = (_QWORD *)a2;
  if ( *(_DWORD *)(a2 + 24) == 51 )
  {
    v16 = (unsigned __int16 *)(*(_QWORD *)(a2 + 48) + 16LL * a3);
    v17 = *(const __m128i **)(a2 + 80);
    v18 = *((_QWORD *)v16 + 1);
    v19 = *v16;
    v35 = v17;
    if ( v17 )
    {
      v30 = v18;
      sub_B96E90((__int64)&v35, (__int64)v17, 1);
      v18 = v30;
    }
    LODWORD(v36) = *((_DWORD *)v7 + 18);
    result = sub_3400BD0((__int64)a1, 0, (__int64)&v35, v19, v18, 0, a7, 0);
    if ( v35 )
    {
      v31 = result;
      sub_B91220((__int64)&v35, (__int64)v35);
      return v31;
    }
  }
  else
  {
    if ( *(_DWORD *)(a4 + 24) == 51 )
    {
      v13 = (unsigned __int16 *)(*(_QWORD *)(a2 + 48) + 16LL * a3);
      v14 = *((_QWORD *)v13 + 1);
      v15 = *v13;
      v35 = 0;
      LODWORD(v36) = 0;
      goto LABEL_7;
    }
    if ( (unsigned __int8)sub_33E0720(a2, a3, 0, a4, a5, a6) || (unsigned __int8)sub_33E0720(a4, a5, 0, v10, v11, a4) )
      return (unsigned __int8 *)v7;
    v32.m128i_i32[2] = a3;
    v34 = sub_33C9840;
    v32.m128i_i64[0] = a2;
    v33 = sub_33C7F80;
    v37 = 0;
    sub_33C7F80(&v35, &v32, 2);
    v20 = (const __m128i **)a5;
    v38 = v34;
    v37 = v33;
    v24 = sub_33CA8D0(v23, a5, (__int64)&v35, v21, v22);
    if ( v37 )
    {
      v20 = &v35;
      v37(&v35, (const __m128i *)&v35, 3);
    }
    if ( v33 )
    {
      v20 = (const __m128i **)&v32;
      v33((const __m128i **)&v32, &v32, 3);
    }
    v25 = (unsigned __int16 *)(v7[6] + 16LL * a3);
    if ( v24 )
    {
      v14 = *((_QWORD *)v25 + 1);
      v15 = *v25;
      LODWORD(v36) = 0;
      v35 = 0;
LABEL_7:
      v7 = sub_33F17F0(a1, 51, (__int64)&v35, v15, v14);
      if ( v35 )
        sub_B91220((__int64)&v35, (__int64)v35);
      return (unsigned __int8 *)v7;
    }
    v26 = *v25;
    v27 = *((_QWORD *)v25 + 1);
    LOWORD(v35) = v26;
    v36 = v27;
    v28 = sub_3281100((unsigned __int16 *)&v35, (__int64)v20) == 2;
    result = 0;
    if ( v28 )
      return (unsigned __int8 *)v7;
  }
  return result;
}
