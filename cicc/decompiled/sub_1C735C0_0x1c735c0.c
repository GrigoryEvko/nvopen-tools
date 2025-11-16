// Function: sub_1C735C0
// Address: 0x1c735c0
//
__int64 __fastcall sub_1C735C0(
        __m128 a1,
        double a2,
        double a3,
        double a4,
        double a5,
        double a6,
        double a7,
        __m128 a8,
        __int64 a9,
        __int64 a10)
{
  unsigned int v10; // r12d
  __int64 v11; // rbx
  __int64 v12; // rdi
  __int64 v13; // rbx
  __int64 v14; // rdi
  __int64 v15; // rbx
  __int64 v16; // rdi
  __int64 v18; // [rsp+0h] [rbp-E0h] BYREF
  int v19; // [rsp+10h] [rbp-D0h] BYREF
  __int64 v20; // [rsp+18h] [rbp-C8h]
  int *v21; // [rsp+20h] [rbp-C0h]
  int *v22; // [rsp+28h] [rbp-B8h]
  __int64 v23; // [rsp+30h] [rbp-B0h]
  int v24; // [rsp+40h] [rbp-A0h] BYREF
  __int64 v25; // [rsp+48h] [rbp-98h]
  int *v26; // [rsp+50h] [rbp-90h]
  int *v27; // [rsp+58h] [rbp-88h]
  __int64 v28; // [rsp+60h] [rbp-80h]
  int v29; // [rsp+70h] [rbp-70h] BYREF
  __int64 v30; // [rsp+78h] [rbp-68h]
  int *v31; // [rsp+80h] [rbp-60h]
  int *v32; // [rsp+88h] [rbp-58h]
  __int64 v33; // [rsp+90h] [rbp-50h]
  int v34; // [rsp+A0h] [rbp-40h] BYREF
  _QWORD *v35; // [rsp+A8h] [rbp-38h]
  int *v36; // [rsp+B0h] [rbp-30h]
  int *v37; // [rsp+B8h] [rbp-28h]
  __int64 v38; // [rsp+C0h] [rbp-20h]
  int v39; // [rsp+C8h] [rbp-18h]

  v21 = &v19;
  v22 = &v19;
  v26 = &v24;
  v27 = &v24;
  v31 = &v29;
  v32 = &v29;
  v19 = 0;
  v20 = 0;
  v23 = 0;
  v24 = 0;
  v25 = 0;
  v28 = 0;
  v29 = 0;
  v30 = 0;
  v33 = 0;
  v34 = 0;
  v35 = 0;
  v36 = &v34;
  v37 = &v34;
  v38 = 0;
  v39 = 0;
  v10 = sub_1C70910(&v18, a10, a1, a2, a3, a4, a5, a6, a7, a8);
  sub_1C6F9B0(v35);
  v11 = v30;
  while ( v11 )
  {
    sub_1C6ECF0(*(_QWORD *)(v11 + 24));
    v12 = v11;
    v11 = *(_QWORD *)(v11 + 16);
    j_j___libc_free_0(v12, 48);
  }
  v13 = v25;
  while ( v13 )
  {
    sub_1C6F090(*(_QWORD *)(v13 + 24));
    v14 = v13;
    v13 = *(_QWORD *)(v13 + 16);
    j_j___libc_free_0(v14, 48);
  }
  v15 = v20;
  while ( v15 )
  {
    sub_1C6F260(*(_QWORD *)(v15 + 24));
    v16 = v15;
    v15 = *(_QWORD *)(v15 + 16);
    j_j___libc_free_0(v16, 48);
  }
  return v10;
}
