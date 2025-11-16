// Function: sub_15E7280
// Address: 0x15e7280
//
_QWORD *__fastcall sub_15E7280(
        __int64 *a1,
        _QWORD *a2,
        __int64 a3,
        __int64 *a4,
        unsigned int a5,
        unsigned __int8 a6,
        __int64 a7,
        __int64 a8,
        __int64 a9)
{
  __int64 *v13; // rax
  __int64 v14; // rdi
  __int64 *v15; // r15
  __int64 v16; // rax
  __int64 v17; // rax
  _QWORD *v18; // rax
  _QWORD *v19; // r12
  __int64 *v21; // rax
  __int64 *v22; // rax
  __int64 v23; // rbx
  __int64 *v24; // rax
  int v26; // [rsp+1Ch] [rbp-84h] BYREF
  __int64 v27[2]; // [rsp+20h] [rbp-80h] BYREF
  __int64 v28[2]; // [rsp+30h] [rbp-70h] BYREF
  __int16 v29; // [rsp+40h] [rbp-60h]
  _QWORD v30[10]; // [rsp+50h] [rbp-50h] BYREF

  v13 = sub_15E7150(a1, a2);
  v14 = a1[3];
  v30[2] = a4;
  v15 = v13;
  v30[0] = v13;
  v30[1] = a3;
  v16 = sub_1643320(v14);
  v30[3] = sub_159C470(v16, a6, 0);
  v27[0] = *v15;
  v27[1] = *a4;
  v17 = sub_15E26F0(*(__int64 **)(*(_QWORD *)(a1[1] + 56) + 40LL), 137, v27, 2);
  v29 = 257;
  v18 = sub_15E6DE0(v17, (int)v30, 4, a1, (int)v28, 0, 0, 0);
  v19 = v18;
  if ( a5 )
  {
    v28[0] = v18[7];
    v21 = (__int64 *)sub_16498A0(v18);
    v19[7] = sub_1563C10(v28, v21, 1, 1);
    v22 = (__int64 *)sub_16498A0(v19);
    v26 = 0;
    v23 = sub_155D330(v22, a5);
    v28[0] = v19[7];
    v24 = (__int64 *)sub_16498A0(v19);
    v28[0] = sub_1563E10(v28, v24, &v26, 1, v23);
    v19[7] = v28[0];
  }
  if ( a7 )
    sub_1625C10(v19, 1, a7);
  if ( a8 )
    sub_1625C10(v19, 7, a8);
  if ( a9 )
    sub_1625C10(v19, 8, a9);
  return v19;
}
