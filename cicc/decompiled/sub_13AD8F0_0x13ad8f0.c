// Function: sub_13AD8F0
// Address: 0x13ad8f0
//
__int64 __fastcall sub_13AD8F0(__int64 a1, __int64 *a2, __int64 *a3, __int64 a4, _BYTE *a5)
{
  __int64 v6; // r14
  __int64 v7; // r13
  __int64 v8; // r15
  __int64 v9; // rbx
  unsigned __int8 v10; // r9
  __int64 v11; // rsi
  __int64 v12; // rsi
  __int64 v13; // rax
  __int64 v14; // r13
  __int64 v15; // r15
  __int64 v16; // rax
  __int64 v17; // rax
  __int64 v18; // rax
  char v19; // al
  unsigned int v20; // r9d
  __int64 v22; // rax
  __int64 v23; // rax
  __int64 v24; // rax
  __int64 v25; // rax
  __int64 v26; // rax
  __int64 v27; // rax
  __int64 v28; // rax
  __int64 v29; // rax
  __int64 v30; // rax
  __int64 v31; // r13
  __int64 v32; // rax
  __int64 v33; // rax
  __int64 v34; // rax
  __int64 v35; // rax
  __int64 v36; // rax
  __int64 v37; // [rsp+0h] [rbp-90h]
  __int64 v38; // [rsp+0h] [rbp-90h]
  unsigned __int8 v39; // [rsp+8h] [rbp-88h]
  __int64 v40; // [rsp+8h] [rbp-88h]
  unsigned __int8 v41; // [rsp+10h] [rbp-80h]
  unsigned __int8 v42; // [rsp+10h] [rbp-80h]
  __int64 v43; // [rsp+10h] [rbp-80h]
  unsigned __int8 v44; // [rsp+10h] [rbp-80h]
  unsigned __int8 v45; // [rsp+10h] [rbp-80h]
  __int64 v46; // [rsp+10h] [rbp-80h]
  unsigned __int8 v50; // [rsp+28h] [rbp-68h]
  unsigned __int8 v51; // [rsp+28h] [rbp-68h]
  __int64 v52; // [rsp+30h] [rbp-60h] BYREF
  unsigned int v53; // [rsp+38h] [rbp-58h]
  __int64 v54; // [rsp+40h] [rbp-50h] BYREF
  unsigned int v55; // [rsp+48h] [rbp-48h]
  __int64 v56; // [rsp+50h] [rbp-40h] BYREF
  unsigned int v57; // [rsp+58h] [rbp-38h]

  v6 = sub_13A62B0(a4);
  v7 = sub_13A6270(a4);
  v8 = sub_13A6280(a4);
  v9 = sub_13A6290(a4);
  v10 = sub_14560B0(v7);
  if ( v10 )
  {
    if ( !*(_WORD *)(v8 + 24) && !*(_WORD *)(v9 + 24) )
    {
      v11 = *(_QWORD *)(v8 + 32);
      v53 = *(_DWORD *)(v11 + 32);
      if ( v53 > 0x40 )
      {
        v45 = v10;
        sub_16A4FD0(&v52, v11 + 24);
        v10 = v45;
      }
      else
      {
        v52 = *(_QWORD *)(v11 + 24);
      }
      v12 = *(_QWORD *)(v9 + 32);
      v55 = *(_DWORD *)(v12 + 32);
      if ( v55 > 0x40 )
      {
        v44 = v10;
        sub_16A4FD0(&v54, v12 + 24);
        v10 = v44;
      }
      else
      {
        v54 = *(_QWORD *)(v12 + 24);
      }
      v41 = v10;
      sub_16A9F90(&v56, &v54, &v52);
      v13 = sub_13AD540(a1, *a3, v6);
      v14 = *(_QWORD *)(a1 + 8);
      v15 = v13;
      v16 = sub_145CF40(v14, &v56);
      v17 = sub_13A5B60(v14, v15, v16, 0, 0);
      *a2 = sub_14806B0(v14, *a2, v17, 0, 0);
      *a3 = sub_13AD5A0(a1, *a3, v6);
      v18 = sub_13AD540(a1, *a2, v6);
      v19 = sub_14560B0(v18);
      v20 = v41;
      if ( !v19 )
        *a5 = 0;
      if ( v57 > 0x40 && v56 )
      {
        j_j___libc_free_0_0(v56);
        v20 = v41;
      }
      if ( v55 > 0x40 && v54 )
      {
        v50 = v20;
        j_j___libc_free_0_0(v54);
        v20 = v50;
      }
      if ( v53 > 0x40 && v52 )
      {
        v51 = v20;
        j_j___libc_free_0_0(v52);
        return v51;
      }
      return v20;
    }
    return 0;
  }
  v39 = sub_14560B0(v8);
  if ( v39 )
  {
    v20 = 0;
    if ( !*(_WORD *)(v7 + 24) && !*(_WORD *)(v9 + 24) )
    {
      sub_13A38D0((__int64)&v52, *(_QWORD *)(v7 + 32) + 24LL);
      sub_13A38D0((__int64)&v54, *(_QWORD *)(v9 + 32) + 24LL);
      sub_16A9F90(&v56, &v54, &v52);
      v37 = sub_13AD540(a1, *a2, v6);
      v43 = *(_QWORD *)(a1 + 8);
      v22 = sub_145CF40(v43, &v56);
      v23 = sub_13A5B60(v43, v37, v22, 0, 0);
      v24 = sub_13A5B00(v43, *a2, v23, 0, 0);
      *a2 = v24;
      *a2 = sub_13AD5A0(a1, v24, v6);
      v25 = sub_13AD540(a1, *a3, v6);
      if ( !(unsigned __int8)sub_14560B0(v25) )
        *a5 = 0;
      sub_135E100(&v56);
      sub_135E100(&v54);
      sub_135E100(&v52);
      return v39;
    }
  }
  else
  {
    v42 = sub_13A7760(a1, 32, v7, v8);
    if ( v42 )
    {
      if ( *(_WORD *)(v7 + 24) || *(_WORD *)(v9 + 24) )
        return 0;
      sub_13A38D0((__int64)&v52, *(_QWORD *)(v7 + 32) + 24LL);
      sub_13A38D0((__int64)&v54, *(_QWORD *)(v9 + 32) + 24LL);
      sub_16A9F90(&v56, &v54, &v52);
      v38 = sub_13AD540(a1, *a2, v6);
      v40 = *(_QWORD *)(a1 + 8);
      v26 = sub_145CF40(v40, &v56);
      v27 = sub_13A5B60(v40, v38, v26, 0, 0);
      v28 = sub_13A5B00(v40, *a2, v27, 0, 0);
      *a2 = v28;
      *a2 = sub_13AD5A0(a1, v28, v6);
      v29 = sub_13AD660(a1, *a3, v6, v38);
      *a3 = v29;
      v30 = sub_13AD540(a1, v29, v6);
      if ( !(unsigned __int8)sub_14560B0(v30) )
        *a5 = 0;
      sub_135E100(&v56);
      sub_135E100(&v54);
      sub_135E100(&v52);
      return v42;
    }
    else
    {
      v46 = sub_13AD540(a1, *a2, v6);
      *a2 = sub_13A5B60(*(_QWORD *)(a1 + 8), *a2, v7, 0, 0);
      *a3 = sub_13A5B60(*(_QWORD *)(a1 + 8), *a3, v7, 0, 0);
      v31 = *(_QWORD *)(a1 + 8);
      v32 = sub_13A5B60(v31, v46, v9, 0, 0);
      v33 = sub_13A5B00(v31, *a2, v32, 0, 0);
      *a2 = v33;
      *a2 = sub_13AD5A0(a1, v33, v6);
      v34 = sub_13A5B60(*(_QWORD *)(a1 + 8), v46, v8, 0, 0);
      v35 = sub_13AD660(a1, *a3, v6, v34);
      *a3 = v35;
      v36 = sub_13AD540(a1, v35, v6);
      v20 = sub_14560B0(v36);
      if ( !(_BYTE)v20 )
      {
        v20 = 1;
        *a5 = 0;
      }
    }
  }
  return v20;
}
