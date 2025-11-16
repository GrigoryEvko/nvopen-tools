// Function: sub_289B1E0
// Address: 0x289b1e0
//
__int64 __fastcall sub_289B1E0(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int16 a4,
        char a5,
        unsigned int a6,
        unsigned int a7,
        char a8,
        _BYTE *a9,
        _BYTE *a10,
        int a11,
        int a12,
        char a13,
        __int64 *a14,
        __int64 a15)
{
  _QWORD *v16; // rdi
  __int64 v17; // r12
  __int64 v18; // rax
  _BYTE *v19; // rax
  _BYTE *v20; // rax
  _BYTE *v21; // rax
  __int64 v22; // r15
  __int64 v23; // rax
  __int64 v24; // rax
  __int64 v27; // [rsp+20h] [rbp-B0h]
  _BYTE *v30; // [rsp+38h] [rbp-98h] BYREF
  _BYTE v31[32]; // [rsp+40h] [rbp-90h] BYREF
  __int16 v32; // [rsp+60h] [rbp-70h]
  _BYTE v33[32]; // [rsp+70h] [rbp-60h] BYREF
  __int16 v34; // [rsp+90h] [rbp-40h]

  if ( !a8 )
    a6 = a7;
  v16 = *(_QWORD **)(a15 + 72);
  v17 = a6;
  v32 = 257;
  v34 = 257;
  v18 = sub_BCB2E0(v16);
  v19 = (_BYTE *)sub_ACD640(v18, v17, 0);
  v20 = (_BYTE *)sub_A81850((unsigned int **)a15, a10, v19, (__int64)v31, 0, 0);
  v21 = (_BYTE *)sub_929C50((unsigned int **)a15, v20, a9, (__int64)v33, 0, 0);
  v34 = 257;
  v30 = v21;
  v27 = sub_921130((unsigned int **)a15, (__int64)a14, a3, &v30, 1, (__int64)v33, 0);
  v22 = sub_BCDA70(a14, a12 * a11);
  v23 = sub_BCB2E0(*(_QWORD **)(a15 + 72));
  v24 = sub_ACD640(v23, v17, 0);
  sub_289A9E0(a1, a2, v22, v27, a4, v24, a5, a11, a12, a13, a15);
  return a1;
}
