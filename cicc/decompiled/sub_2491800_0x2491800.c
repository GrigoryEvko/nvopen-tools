// Function: sub_2491800
// Address: 0x2491800
//
_QWORD *__fastcall sub_2491800(
        __int64 a1,
        __int64 **a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        unsigned __int64 a6,
        __int64 a7)
{
  __int64 *v8; // rsi
  __int64 v9; // r12
  __int64 *v10; // r13
  __int64 v11; // rax
  unsigned __int64 v12; // r8
  unsigned __int64 v13; // rax
  __int64 v14; // rax
  __int64 *v15; // r9
  __int64 v16; // r8
  __int64 v17; // rdx
  __int64 v18; // r15
  __int64 v19; // rax
  __int64 *v20; // rax
  _QWORD *result; // rax
  __int64 v22; // r12
  unsigned __int64 v23; // r15
  __int64 i; // r13
  __int64 v25; // rdx
  __int64 v26; // r8
  __int64 v27; // rax
  __int64 v28; // r9
  unsigned __int64 v29; // rax
  __int64 v30; // rax
  __int64 *v31; // r9
  __int64 v32; // r8
  __int64 v33; // rdx
  __int64 v34; // r15
  __int64 v35; // rax
  __int64 *v36; // rax
  __int64 v37; // [rsp+8h] [rbp-A8h]
  const void *v38; // [rsp+18h] [rbp-98h]
  __int64 v41; // [rsp+28h] [rbp-88h]
  __int64 v42; // [rsp+28h] [rbp-88h]
  __int64 v43; // [rsp+28h] [rbp-88h]
  __int64 v44; // [rsp+28h] [rbp-88h]
  __int64 v46; // [rsp+30h] [rbp-80h]
  __int64 v47; // [rsp+38h] [rbp-78h]
  unsigned __int64 v48; // [rsp+48h] [rbp-68h] BYREF
  __int64 *v49; // [rsp+50h] [rbp-60h] BYREF
  __int64 v50; // [rsp+58h] [rbp-58h]
  __int64 v51; // [rsp+60h] [rbp-50h] BYREF
  __int64 v52; // [rsp+68h] [rbp-48h]
  __int64 v53; // [rsp+70h] [rbp-40h]

  *(_QWORD *)a1 = a1 + 16;
  v38 = (const void *)(a1 + 16);
  *(_QWORD *)(a1 + 8) = 0x300000000LL;
  v8 = *a2;
  v48 = 0;
  v48 = sub_A7A090((__int64 *)&v48, v8, -1, 41);
  v9 = sub_BCE3C0(v8, 0);
  v10 = (__int64 *)sub_BCB120(v8);
  v11 = sub_AE4420((__int64)(a2 + 39), (__int64)v8, 0);
  v12 = v48;
  *(_QWORD *)(a1 + 64) = 3;
  v37 = v12;
  v49 = &v51;
  v51 = v9;
  if ( a7 == 3 )
  {
    v53 = v11;
    v52 = v9;
    v50 = 0x300000003LL;
    v29 = sub_BCF480(v10, &v51, 3, 0);
    v30 = sub_BA8C10((__int64)a2, a5, a6, v29, v37);
    v31 = &v51;
    v32 = v30;
    v34 = v33;
    if ( v49 != &v51 )
    {
      v42 = v30;
      _libc_free((unsigned __int64)v49);
      v32 = v42;
    }
    v35 = *(unsigned int *)(a1 + 8);
    if ( v35 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 12) )
    {
      v44 = v32;
      sub_C8D5F0(a1, v38, v35 + 1, 0x10u, v32, (__int64)v31);
      v35 = *(unsigned int *)(a1 + 8);
      v32 = v44;
    }
    v36 = (__int64 *)(*(_QWORD *)a1 + 16 * v35);
    *v36 = v32;
    v36[1] = v34;
    ++*(_DWORD *)(a1 + 8);
    v49 = (__int64 *)v9;
    v50 = v9;
    result = (_QWORD *)sub_BCF480(v10, &v49, 2, 0);
    v22 = (__int64)result;
  }
  else
  {
    v52 = v11;
    v50 = 0x200000002LL;
    v13 = sub_BCF480(v10, &v51, 2, 0);
    v14 = sub_BA8C10((__int64)a2, a5, a6, v13, v37);
    v15 = &v51;
    v16 = v14;
    v18 = v17;
    if ( v49 != &v51 )
    {
      v41 = v14;
      _libc_free((unsigned __int64)v49);
      v16 = v41;
    }
    v19 = *(unsigned int *)(a1 + 8);
    if ( v19 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 12) )
    {
      v43 = v16;
      sub_C8D5F0(a1, v38, v19 + 1, 0x10u, v16, (__int64)v15);
      v19 = *(unsigned int *)(a1 + 8);
      v16 = v43;
    }
    v20 = (__int64 *)(*(_QWORD *)a1 + 16 * v19);
    *v20 = v16;
    v20[1] = v18;
    ++*(_DWORD *)(a1 + 8);
    v49 = (__int64 *)v9;
    result = (_QWORD *)sub_BCF480(v10, &v49, 1, 0);
    v22 = (__int64)result;
  }
  v23 = 0;
  for ( i = a3; *(_QWORD *)(a1 + 64) > v23; ++*(_DWORD *)(a1 + 8) )
  {
    v26 = sub_BA8C10((__int64)a2, *(_QWORD *)i, *(_QWORD *)(i + 8), v22, v48);
    v27 = *(unsigned int *)(a1 + 8);
    v28 = v25;
    if ( v27 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 12) )
    {
      v46 = v26;
      v47 = v25;
      sub_C8D5F0(a1, v38, v27 + 1, 0x10u, v26, v25);
      v27 = *(unsigned int *)(a1 + 8);
      v26 = v46;
      v28 = v47;
    }
    result = (_QWORD *)(*(_QWORD *)a1 + 16 * v27);
    ++v23;
    i += 16;
    *result = v26;
    result[1] = v28;
  }
  return result;
}
