// Function: sub_21CF3B0
// Address: 0x21cf3b0
//
__int64 *__fastcall sub_21CF3B0(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 *a4,
        int a5,
        int *a6,
        __m128i a7,
        double a8,
        __m128i a9,
        int a10,
        char a11)
{
  __int64 v11; // r15
  __int64 v13; // rcx
  __int64 *result; // rax
  bool v17; // al
  __int64 v18; // rsi
  unsigned __int8 *v19; // r15
  const void **v20; // rax
  unsigned int v21; // r15d
  char v22; // al
  bool v23; // cf
  __int64 v24; // rsi
  __int64 v25; // rax
  unsigned __int64 v26; // rdx
  __int64 v27; // rax
  unsigned __int64 v28; // rdx
  __int64 *v29; // rax
  __int64 v30; // rdx
  __int64 v31; // r13
  __int64 *v32; // r12
  __int64 v33; // rax
  unsigned __int64 v34; // rdx
  __int128 v35; // [rsp-20h] [rbp-80h]
  __int128 v36; // [rsp-20h] [rbp-80h]
  __int128 v37; // [rsp-20h] [rbp-80h]
  int *v38; // [rsp+8h] [rbp-58h]
  int *v39; // [rsp+8h] [rbp-58h]
  int *v40; // [rsp+8h] [rbp-58h]
  const void **v41; // [rsp+10h] [rbp-50h]
  __int64 v42; // [rsp+18h] [rbp-48h]
  char v43; // [rsp+18h] [rbp-48h]
  __int64 *v44; // [rsp+18h] [rbp-48h]
  __int64 v45; // [rsp+20h] [rbp-40h] BYREF
  int v46; // [rsp+28h] [rbp-38h]

  v11 = (unsigned int)a3;
  v13 = a2;
  if ( a5 != 1 )
  {
    if ( a5 != -1 )
      return 0;
    v38 = a6;
    v17 = sub_21CF340(a1);
    v13 = a2;
    a6 = v38;
    if ( v17 )
      return 0;
  }
  if ( *a6 == -1 )
    *a6 = 0;
  v18 = *(_QWORD *)(v13 + 72);
  v45 = v18;
  if ( v18 )
  {
    v39 = a6;
    v42 = v13;
    sub_1623A60((__int64)&v45, v18, 2);
    a6 = v39;
    v13 = v42;
  }
  v19 = (unsigned __int8 *)(*(_QWORD *)(v13 + 40) + 16 * v11);
  v40 = a6;
  v46 = *(_DWORD *)(v13 + 64);
  v20 = (const void **)*((_QWORD *)v19 + 1);
  v43 = *v19;
  v21 = *v19;
  v41 = v20;
  v22 = sub_21CF380();
  if ( a11 || *v40 > 0 )
  {
    if ( v43 == 9 )
    {
      v23 = v22 == 0;
      v24 = 4368;
LABEL_14:
      v25 = sub_1D38BB0((__int64)a4, v24 - (v23 - 1LL), (__int64)&v45, 5, 0, 0, a7, a8, a9, 0);
LABEL_15:
      *((_QWORD *)&v35 + 1) = a3;
      *(_QWORD *)&v35 = a2;
      result = sub_1D332F0(a4, 43, (__int64)&v45, v21, v41, 0, *(double *)a7.m128i_i64, a8, a9, v25, v26, v35);
      goto LABEL_16;
    }
    if ( v43 == 10 )
    {
      v25 = sub_1D38BB0((__int64)a4, 4367, (__int64)&v45, 5, 0, 0, a7, a8, a9, 0);
      goto LABEL_15;
    }
    result = 0;
  }
  else
  {
    if ( v43 == 9 )
    {
      v23 = v22 == 0;
      v24 = 4475;
      goto LABEL_14;
    }
    v27 = sub_1D38BB0((__int64)a4, 4367, (__int64)&v45, 5, 0, 0, a7, a8, a9, 0);
    *((_QWORD *)&v36 + 1) = a3;
    *(_QWORD *)&v36 = a2;
    v29 = sub_1D332F0(a4, 43, (__int64)&v45, v21, v41, 0, *(double *)a7.m128i_i64, a8, a9, v27, v28, v36);
    v31 = v30;
    v32 = v29;
    v33 = sub_1D38BB0((__int64)a4, 4255, (__int64)&v45, 5, 0, 0, a7, a8, a9, 0);
    *((_QWORD *)&v37 + 1) = v31;
    *(_QWORD *)&v37 = v32;
    result = sub_1D332F0(a4, 43, (__int64)&v45, v21, v41, 0, *(double *)a7.m128i_i64, a8, a9, v33, v34, v37);
  }
LABEL_16:
  if ( v45 )
  {
    v44 = result;
    sub_161E7C0((__int64)&v45, v45);
    return v44;
  }
  return result;
}
