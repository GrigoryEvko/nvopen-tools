// Function: sub_15E7430
// Address: 0x15e7430
//
_QWORD *__fastcall sub_15E7430(
        __int64 *a1,
        _QWORD *a2,
        unsigned int a3,
        _QWORD *a4,
        unsigned int a5,
        __int64 *a6,
        unsigned __int8 a7,
        __int64 a8,
        __int64 a9,
        __int64 a10,
        __int64 a11)
{
  __int64 *v15; // rbx
  __int64 *v16; // rax
  __int64 v17; // rdi
  __int64 *v18; // r13
  __int64 v19; // rax
  __int64 v20; // rax
  _QWORD *v21; // rax
  _QWORD *v22; // r14
  __int64 *v24; // rax
  __int64 *v25; // rax
  __int64 v26; // r12
  __int64 *v27; // rax
  __int64 *v28; // rax
  __int64 *v29; // rax
  __int64 v30; // rax
  __int64 v31; // rbx
  __int64 *v32; // rax
  int v34; // [rsp+1Ch] [rbp-94h] BYREF
  __int64 v35[4]; // [rsp+20h] [rbp-90h] BYREF
  __int64 v36[2]; // [rsp+40h] [rbp-70h] BYREF
  __int16 v37; // [rsp+50h] [rbp-60h]
  _QWORD v38[10]; // [rsp+60h] [rbp-50h] BYREF

  v15 = sub_15E7150(a1, a2);
  v16 = sub_15E7150(a1, a4);
  v17 = a1[3];
  v38[0] = v15;
  v18 = v16;
  v38[1] = v16;
  v38[2] = a6;
  v19 = sub_1643320(v17);
  v38[3] = sub_159C470(v19, a7, 0);
  v35[0] = *v15;
  v35[1] = *v18;
  v35[2] = *a6;
  v20 = sub_15E26F0(*(__int64 **)(*(_QWORD *)(a1[1] + 56) + 40LL), 133, v35, 3);
  v37 = 257;
  v21 = sub_15E6DE0(v20, (int)v38, 4, a1, (int)v36, 0, 0, 0);
  v22 = v21;
  if ( a3 )
  {
    v36[0] = v21[7];
    v24 = (__int64 *)sub_16498A0(v21);
    v22[7] = sub_1563C10(v36, v24, 1, 1);
    v25 = (__int64 *)sub_16498A0(v22);
    v34 = 0;
    v26 = sub_155D330(v25, a3);
    v36[0] = v22[7];
    v27 = (__int64 *)sub_16498A0(v22);
    v36[0] = sub_1563E10(v36, v27, &v34, 1, v26);
    v22[7] = v36[0];
    if ( !a5 )
      goto LABEL_3;
  }
  else if ( !a5 )
  {
    goto LABEL_3;
  }
  v36[0] = v22[7];
  v28 = (__int64 *)sub_16498A0(v22);
  v22[7] = sub_1563C10(v36, v28, 2, 1);
  v29 = (__int64 *)sub_16498A0(v22);
  v30 = sub_155D330(v29, a5);
  v34 = 1;
  v31 = v30;
  v36[0] = v22[7];
  v32 = (__int64 *)sub_16498A0(v22);
  v36[0] = sub_1563E10(v36, v32, &v34, 1, v31);
  v22[7] = v36[0];
LABEL_3:
  if ( a8 )
    sub_1625C10(v22, 1, a8);
  if ( a9 )
    sub_1625C10(v22, 5, a9);
  if ( a10 )
    sub_1625C10(v22, 7, a10);
  if ( a11 )
    sub_1625C10(v22, 8, a11);
  return v22;
}
