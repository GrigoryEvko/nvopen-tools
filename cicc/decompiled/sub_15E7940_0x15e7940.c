// Function: sub_15E7940
// Address: 0x15e7940
//
_QWORD *__fastcall sub_15E7940(
        __int64 *a1,
        _QWORD *a2,
        unsigned int a3,
        _QWORD *a4,
        unsigned int a5,
        __int64 *a6,
        unsigned __int8 a7,
        __int64 a8,
        __int64 a9,
        __int64 a10)
{
  __int64 *v14; // rbx
  __int64 *v15; // rax
  __int64 v16; // rdi
  __int64 *v17; // r13
  __int64 v18; // rax
  __int64 v19; // rax
  _QWORD *v20; // rax
  _QWORD *v21; // r14
  __int64 *v23; // rax
  __int64 *v24; // rax
  __int64 v25; // r12
  __int64 *v26; // rax
  __int64 *v27; // rax
  __int64 *v28; // rax
  __int64 v29; // rax
  __int64 v30; // rbx
  __int64 *v31; // rax
  int v33; // [rsp+1Ch] [rbp-94h] BYREF
  __int64 v34[4]; // [rsp+20h] [rbp-90h] BYREF
  __int64 v35[2]; // [rsp+40h] [rbp-70h] BYREF
  __int16 v36; // [rsp+50h] [rbp-60h]
  _QWORD v37[10]; // [rsp+60h] [rbp-50h] BYREF

  v14 = sub_15E7150(a1, a2);
  v15 = sub_15E7150(a1, a4);
  v16 = a1[3];
  v37[0] = v14;
  v17 = v15;
  v37[1] = v15;
  v37[2] = a6;
  v18 = sub_1643320(v16);
  v37[3] = sub_159C470(v18, a7, 0);
  v34[0] = *v14;
  v34[1] = *v17;
  v34[2] = *a6;
  v19 = sub_15E26F0(*(__int64 **)(*(_QWORD *)(a1[1] + 56) + 40LL), 135, v34, 3);
  v36 = 257;
  v20 = sub_15E6DE0(v19, (int)v37, 4, a1, (int)v35, 0, 0, 0);
  v21 = v20;
  if ( a3 )
  {
    v35[0] = v20[7];
    v23 = (__int64 *)sub_16498A0(v20);
    v21[7] = sub_1563C10(v35, v23, 1, 1);
    v24 = (__int64 *)sub_16498A0(v21);
    v33 = 0;
    v25 = sub_155D330(v24, a3);
    v35[0] = v21[7];
    v26 = (__int64 *)sub_16498A0(v21);
    v35[0] = sub_1563E10(v35, v26, &v33, 1, v25);
    v21[7] = v35[0];
    if ( !a5 )
      goto LABEL_3;
  }
  else if ( !a5 )
  {
    goto LABEL_3;
  }
  v35[0] = v21[7];
  v27 = (__int64 *)sub_16498A0(v21);
  v21[7] = sub_1563C10(v35, v27, 2, 1);
  v28 = (__int64 *)sub_16498A0(v21);
  v29 = sub_155D330(v28, a5);
  v33 = 1;
  v30 = v29;
  v35[0] = v21[7];
  v31 = (__int64 *)sub_16498A0(v21);
  v35[0] = sub_1563E10(v35, v31, &v33, 1, v30);
  v21[7] = v35[0];
LABEL_3:
  if ( a8 )
    sub_1625C10(v21, 1, a8);
  if ( a9 )
    sub_1625C10(v21, 7, a9);
  if ( a10 )
    sub_1625C10(v21, 8, a10);
  return v21;
}
