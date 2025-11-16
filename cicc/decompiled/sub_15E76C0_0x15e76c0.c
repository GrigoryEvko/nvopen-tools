// Function: sub_15E76C0
// Address: 0x15e76c0
//
_QWORD *__fastcall sub_15E76C0(
        __int64 *a1,
        _QWORD *a2,
        unsigned int a3,
        _QWORD *a4,
        unsigned int a5,
        __int64 *a6,
        unsigned int a7,
        __int64 a8,
        __int64 a9,
        __int64 a10,
        __int64 a11)
{
  __int64 *v15; // rbx
  __int64 *v16; // rax
  __int64 v17; // rdi
  __int64 *v18; // r15
  __int64 v19; // rax
  __int64 v20; // rax
  _QWORD *v21; // r14
  __int64 *v22; // rax
  __int64 v23; // rax
  __int64 *v24; // rax
  __int64 *v26; // rax
  __int64 v27; // r12
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
  v38[2] = a6;
  v18 = v16;
  v38[0] = v15;
  v38[1] = v16;
  v19 = sub_1643350(v17);
  v38[3] = sub_159C470(v19, a7, 0);
  v35[0] = *v15;
  v35[1] = *v18;
  v35[2] = *a6;
  v20 = sub_15E26F0(*(__int64 **)(*(_QWORD *)(a1[1] + 56) + 40LL), 134, v35, 3);
  v37 = 257;
  v21 = sub_15E6DE0(v20, (int)v38, 4, a1, (int)v36, 0, 0, 0);
  v36[0] = v21[7];
  v22 = (__int64 *)sub_16498A0(v21);
  v23 = sub_1563C10(v36, v22, 1, 1);
  v21[7] = v23;
  if ( a3 )
  {
    v26 = (__int64 *)sub_16498A0(v21);
    v34 = 0;
    v27 = sub_155D330(v26, a3);
    v36[0] = v21[7];
    v28 = (__int64 *)sub_16498A0(v21);
    v23 = sub_1563E10(v36, v28, &v34, 1, v27);
    v21[7] = v23;
  }
  v36[0] = v23;
  v24 = (__int64 *)sub_16498A0(v21);
  v21[7] = sub_1563C10(v36, v24, 2, 1);
  if ( a5 )
  {
    v29 = (__int64 *)sub_16498A0(v21);
    v30 = sub_155D330(v29, a5);
    v34 = 1;
    v31 = v30;
    v36[0] = v21[7];
    v32 = (__int64 *)sub_16498A0(v21);
    v36[0] = sub_1563E10(v36, v32, &v34, 1, v31);
    v21[7] = v36[0];
  }
  if ( a8 )
    sub_1625C10(v21, 1, a8);
  if ( a9 )
    sub_1625C10(v21, 5, a9);
  if ( a10 )
    sub_1625C10(v21, 7, a10);
  if ( a11 )
    sub_1625C10(v21, 8, a11);
  return v21;
}
