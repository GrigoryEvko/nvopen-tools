// Function: sub_1CB3850
// Address: 0x1cb3850
//
__int64 __fastcall sub_1CB3850(
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
  _QWORD *v10; // rax
  double v11; // xmm4_8
  double v12; // xmm5_8
  _QWORD *v13; // rdx
  char v14; // cl
  unsigned int v15; // r12d
  __int64 v16; // rax
  _QWORD *v18; // rbx
  _QWORD *v19; // r14
  __int64 v20; // rsi
  _QWORD *v21; // rbx
  _QWORD *v22; // r14
  __int64 v23; // rax
  _QWORD v24[2]; // [rsp+8h] [rbp-D8h] BYREF
  __int64 v25; // [rsp+18h] [rbp-C8h]
  __int64 v26; // [rsp+20h] [rbp-C0h]
  void *v27; // [rsp+30h] [rbp-B0h]
  __int64 v28; // [rsp+38h] [rbp-A8h] BYREF
  __int64 v29; // [rsp+40h] [rbp-A0h]
  __int64 v30; // [rsp+48h] [rbp-98h]
  __int64 i; // [rsp+50h] [rbp-90h]
  __int64 v32; // [rsp+60h] [rbp-80h] BYREF
  _QWORD *v33; // [rsp+68h] [rbp-78h]
  __int64 v34; // [rsp+70h] [rbp-70h]
  unsigned int v35; // [rsp+78h] [rbp-68h]
  _QWORD *v36; // [rsp+88h] [rbp-58h]
  unsigned int v37; // [rsp+98h] [rbp-48h]
  char v38; // [rsp+A0h] [rbp-40h]
  char v39; // [rsp+A9h] [rbp-37h]
  int v40; // [rsp+AAh] [rbp-36h]
  __int16 v41; // [rsp+AEh] [rbp-32h]

  v40 = 0;
  v41 = 0;
  v32 = 0;
  v35 = 128;
  v10 = (_QWORD *)sub_22077B0(6144);
  v34 = 0;
  v33 = v10;
  v28 = 2;
  v27 = &unk_49F8530;
  v29 = 0;
  v13 = v10 + 768;
  v30 = -8;
  for ( i = 0; v13 != v10; v10 += 6 )
  {
    if ( v10 )
    {
      v14 = v28;
      v10[2] = 0;
      v10[3] = -8;
      *v10 = &unk_49F8530;
      v10[1] = v14 & 6;
      v10[4] = i;
    }
  }
  v15 = (unsigned __int8)byte_4FBE7E0;
  v38 = 0;
  v39 = 1;
  if ( !byte_4FBE7E0 || (v15 = sub_1CB1E60((__int64)&v32, a10, a1, a2, a3, a4, v11, v12, a7, a8), !v38) )
  {
    v16 = v35;
    if ( !v35 )
      goto LABEL_7;
    goto LABEL_17;
  }
  if ( v37 )
  {
    v18 = v36;
    v19 = &v36[2 * v37];
    do
    {
      if ( *v18 != -8 && *v18 != -4 )
      {
        v20 = v18[1];
        if ( v20 )
          sub_161E7C0((__int64)(v18 + 1), v20);
      }
      v18 += 2;
    }
    while ( v19 != v18 );
  }
  j___libc_free_0(v36);
  v16 = v35;
  if ( v35 )
  {
LABEL_17:
    v21 = v33;
    v24[0] = 2;
    v22 = &v33[6 * v16];
    v24[1] = 0;
    v25 = -8;
    v26 = 0;
    v28 = 2;
    v29 = 0;
    v30 = -16;
    v27 = &unk_49F8530;
    i = 0;
    do
    {
      v23 = v21[3];
      *v21 = &unk_49EE2B0;
      if ( v23 != 0 && v23 != -8 && v23 != -16 )
        sub_1649B30(v21 + 1);
      v21 += 6;
    }
    while ( v22 != v21 );
    v27 = &unk_49EE2B0;
    if ( v30 != 0 && v30 != -8 && v30 != -16 )
      sub_1649B30(&v28);
    if ( v25 != 0 && v25 != -8 && v25 != -16 )
      sub_1649B30(v24);
  }
LABEL_7:
  j___libc_free_0(v33);
  return v15;
}
