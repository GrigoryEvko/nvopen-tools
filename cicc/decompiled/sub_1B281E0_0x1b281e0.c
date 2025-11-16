// Function: sub_1B281E0
// Address: 0x1b281e0
//
__int64 __fastcall sub_1B281E0(
        _QWORD **a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        _QWORD *a7,
        __int64 a8,
        __int64 *a9,
        __int64 a10,
        __int64 a11,
        __int64 a12)
{
  __int64 v13; // rax
  _QWORD *v14; // rdi
  __int64 v15; // r14
  __int64 *v16; // rax
  __int64 v17; // rbx
  __int64 v18; // rax
  __int64 v19; // r13
  _QWORD *v20; // rbx
  _QWORD *v21; // rax
  _QWORD *v22; // r9
  _QWORD *v23; // rax
  __int64 v24; // rbx
  _QWORD *v25; // rax
  unsigned __int8 *v26; // rsi
  __int64 *v28; // rax
  __int64 v29; // rax
  __int64 v30; // rax
  __int64 v31; // rax
  __int64 v32; // [rsp+8h] [rbp-D8h]
  _QWORD *v33; // [rsp+10h] [rbp-D0h]
  _QWORD *v34; // [rsp+10h] [rbp-D0h]
  _QWORD v35[2]; // [rsp+30h] [rbp-B0h] BYREF
  unsigned __int8 *v36[2]; // [rsp+40h] [rbp-A0h] BYREF
  __int16 v37; // [rsp+50h] [rbp-90h]
  __int64 v38[2]; // [rsp+60h] [rbp-80h] BYREF
  __int64 v39; // [rsp+70h] [rbp-70h]
  _QWORD *v40; // [rsp+78h] [rbp-68h]
  __int64 v41; // [rsp+80h] [rbp-60h]
  int v42; // [rsp+88h] [rbp-58h]
  __int64 v43; // [rsp+90h] [rbp-50h]
  __int64 v44; // [rsp+98h] [rbp-48h]

  v35[0] = a2;
  v35[1] = a3;
  v13 = sub_1B28160(a1, a4, a5, a7, a8);
  v14 = *a1;
  v15 = v13;
  LOWORD(v39) = 261;
  v38[0] = (__int64)v35;
  v16 = (__int64 *)sub_1643270(v14);
  v17 = sub_16453E0(v16, 0);
  v18 = sub_1648B60(120);
  v19 = v18;
  if ( v18 )
    sub_15E2490(v18, v17, 7, (__int64)v38, (__int64)a1);
  v20 = *a1;
  LOWORD(v39) = 257;
  v21 = (_QWORD *)sub_22077B0(64);
  v22 = v21;
  if ( v21 )
  {
    v33 = v21;
    sub_157FB60(v21, (__int64)v20, (__int64)v38, v19, 0);
    v22 = v33;
  }
  v32 = (__int64)v22;
  v34 = *a1;
  v23 = sub_1648A60(56, 0);
  v24 = (__int64)v23;
  if ( v23 )
    sub_15F7190((__int64)v23, (__int64)v34, v32);
  v25 = (_QWORD *)sub_16498A0(v24);
  v38[0] = 0;
  v40 = v25;
  v41 = 0;
  v42 = 0;
  v43 = 0;
  v44 = 0;
  v38[1] = *(_QWORD *)(v24 + 40);
  v39 = v24 + 24;
  v26 = *(unsigned __int8 **)(v24 + 48);
  v36[0] = v26;
  if ( v26 )
  {
    sub_1623A60((__int64)v36, (__int64)v26, 2);
    if ( v38[0] )
      sub_161E7C0((__int64)v38, v38[0]);
    v38[0] = (__int64)v36[0];
    if ( v36[0] )
      sub_1623210((__int64)v36, v36[0], (__int64)v38);
  }
  v37 = 257;
  sub_1B27230((__int64)v38, *(_QWORD *)(v15 + 24), v15, a9, a10, (__int64 *)v36, 0);
  if ( a12 )
  {
    v28 = (__int64 *)sub_1643270(v40);
    v29 = sub_1644EA0(v28, 0, 0, 0);
    v30 = sub_1632080((__int64)a1, a11, a12, v29, 0);
    v31 = sub_1B28080(v30);
    v37 = 257;
    sub_1B27230((__int64)v38, *(_QWORD *)(v31 + 24), v31, 0, 0, (__int64 *)v36, 0);
  }
  if ( v38[0] )
    sub_161E7C0((__int64)v38, v38[0]);
  return v19;
}
