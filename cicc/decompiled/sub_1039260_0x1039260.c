// Function: sub_1039260
// Address: 0x1039260
//
__int64 __fastcall sub_1039260(__int64 *a1, __int64 *a2, unsigned __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 *v7; // rbx
  __int64 v8; // rax
  __int64 v9; // r9
  __int64 v10; // rdx
  unsigned __int64 v11; // r8
  __int64 v12; // r13
  __int64 v13; // r13
  __int64 v14; // rax
  __int64 v15; // rax
  __int64 *v16; // rax
  __int64 v17; // r13
  __int64 *v18; // r15
  __int64 v19; // rax
  __int64 v20; // rax
  _QWORD *v21; // rax
  __int64 v22; // rax
  __int64 v23; // r9
  __int64 v24; // rdx
  unsigned __int64 v25; // r8
  unsigned __int64 v26; // rdx
  __int64 *v27; // rsi
  __int64 v28; // r12
  __int64 v30; // [rsp+0h] [rbp-B0h]
  char v31; // [rsp+18h] [rbp-98h]
  __int64 *v32; // [rsp+18h] [rbp-98h]
  __int64 v33; // [rsp+18h] [rbp-98h]
  __int64 *v34; // [rsp+20h] [rbp-90h] BYREF
  _QWORD *v35; // [rsp+28h] [rbp-88h]
  __int64 v36; // [rsp+30h] [rbp-80h] BYREF
  __int64 *v37; // [rsp+40h] [rbp-70h] BYREF
  __int64 v38; // [rsp+48h] [rbp-68h]
  _QWORD v39[12]; // [rsp+50h] [rbp-60h] BYREF

  v7 = (__int64 *)a5;
  v31 = a4;
  v37 = v39;
  v39[0] = sub_1038FB0(a2, a3, a1, a4, a5, a6);
  v38 = 0x600000001LL;
  sub_10391D0((__int64)&v34, v31);
  v8 = sub_B9B140(a1, v34, (size_t)v35);
  v10 = (unsigned int)v38;
  v11 = (unsigned int)v38 + 1LL;
  if ( v11 > HIDWORD(v38) )
  {
    v33 = v8;
    sub_C8D5F0((__int64)&v37, v39, (unsigned int)v38 + 1LL, 8u, v11, v9);
    v10 = (unsigned int)v38;
    v8 = v33;
  }
  v37[v10] = v8;
  LODWORD(v38) = v38 + 1;
  if ( v34 != &v36 )
    j_j___libc_free_0(v34, v36 + 1);
  if ( !a6 || (v12 = 2 * a6, v32 = &v7[v12], v7 == &v7[v12]) )
  {
    v26 = (unsigned int)v38;
  }
  else
  {
    do
    {
      v13 = *v7;
      v14 = sub_BCB2E0(a1);
      v15 = sub_ACD640(v14, v13, 0);
      v16 = sub_B98A20(v15, v13);
      v17 = v7[1];
      v18 = v16;
      v19 = sub_BCB2E0(a1);
      v20 = sub_ACD640(v19, v17, 0);
      v21 = sub_B98A20(v20, v17);
      v34 = v18;
      v35 = v21;
      v22 = sub_B9C770(a1, (__int64 *)&v34, (__int64 *)2, 0, 1);
      v24 = (unsigned int)v38;
      v25 = (unsigned int)v38 + 1LL;
      if ( v25 > HIDWORD(v38) )
      {
        v30 = v22;
        sub_C8D5F0((__int64)&v37, v39, (unsigned int)v38 + 1LL, 8u, v25, v23);
        v24 = (unsigned int)v38;
        v22 = v30;
      }
      v7 += 2;
      v37[v24] = v22;
      v26 = (unsigned int)(v38 + 1);
      LODWORD(v38) = v38 + 1;
    }
    while ( v32 != v7 );
  }
  v27 = v37;
  v28 = sub_B9C770(a1, v37, (__int64 *)v26, 0, 1);
  if ( v37 != v39 )
    _libc_free(v37, v27);
  return v28;
}
