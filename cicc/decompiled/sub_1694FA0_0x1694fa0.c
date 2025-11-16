// Function: sub_1694FA0
// Address: 0x1694fa0
//
void __fastcall sub_1694FA0(__int64 **a1, __int64 a2, __int64 *a3, __int64 a4, __int64 a5, unsigned int a6, int a7)
{
  __int64 *v10; // r12
  __int64 v11; // r8
  __int64 v12; // rax
  __int64 v13; // rax
  __int64 v14; // rax
  __int64 v15; // rdx
  __int64 v16; // rcx
  __int64 v17; // r8
  __int64 v18; // rax
  __int64 v19; // rax
  __int64 v20; // rax
  __int64 v21; // rdx
  __int64 v22; // rcx
  __int64 v23; // r13
  __int64 v24; // rax
  __int64 v25; // r14
  __int64 *v26; // r13
  __int64 *v27; // rdx
  __int64 v28; // r14
  __int64 v29; // rax
  __int64 v30; // rax
  __int64 v31; // rdx
  __int64 v32; // rcx
  __int64 v33; // r9
  __int64 v34; // rax
  __int64 v35; // r14
  __int64 v36; // rax
  __int64 v37; // rax
  __int64 v38; // rdx
  __int64 v39; // rcx
  __int64 v40; // r14
  __int64 v41; // rax
  __int64 v42; // rax
  __int64 v43; // [rsp+8h] [rbp-98h]
  __int64 v44; // [rsp+8h] [rbp-98h]
  __int64 *v46; // [rsp+28h] [rbp-78h]
  __int64 v47; // [rsp+28h] [rbp-78h]
  __int64 *v48; // [rsp+38h] [rbp-68h] BYREF
  __int64 *v49; // [rsp+40h] [rbp-60h] BYREF
  __int64 v50; // [rsp+48h] [rbp-58h]
  _BYTE v51[80]; // [rsp+50h] [rbp-50h] BYREF

  v10 = *a1;
  v49 = (__int64 *)v51;
  v48 = v10;
  v50 = 0x300000000LL;
  v11 = sub_161BD10(&v48, (__int64)"VP", 2);
  v12 = (unsigned int)v50;
  if ( (unsigned int)v50 >= HIDWORD(v50) )
  {
    v44 = v11;
    sub_16CD150(&v49, v51, 0, 8);
    v12 = (unsigned int)v50;
    v11 = v44;
  }
  v49[v12] = v11;
  LODWORD(v50) = v50 + 1;
  v13 = sub_1643350(v10);
  v14 = sub_159C470(v13, a6, 0);
  v17 = sub_161BD20((__int64)&v48, v14, v15, v16);
  v18 = (unsigned int)v50;
  if ( (unsigned int)v50 >= HIDWORD(v50) )
  {
    v47 = v17;
    sub_16CD150(&v49, v51, 0, 8);
    v18 = (unsigned int)v50;
    v17 = v47;
  }
  v49[v18] = v17;
  LODWORD(v50) = v50 + 1;
  v19 = sub_1643360(v10);
  v20 = sub_159C470(v19, a5, 0);
  v23 = sub_161BD20((__int64)&v48, v20, v21, v22);
  v24 = (unsigned int)v50;
  if ( (unsigned int)v50 >= HIDWORD(v50) )
  {
    sub_16CD150(&v49, v51, 0, 8);
    v24 = (unsigned int)v50;
  }
  v25 = 2 * a4;
  v49[v24] = v23;
  v26 = &a3[v25];
  v27 = (__int64 *)(unsigned int)(v50 + 1);
  LODWORD(v50) = v50 + 1;
  if ( a3 != &a3[v25] )
  {
    v46 = &a3[2 * (unsigned int)(a7 - 1)];
    do
    {
      v35 = *a3;
      v36 = sub_1643360(v10);
      v37 = sub_159C470(v36, v35, 0);
      v40 = sub_161BD20((__int64)&v48, v37, v38, v39);
      v41 = (unsigned int)v50;
      if ( (unsigned int)v50 >= HIDWORD(v50) )
      {
        sub_16CD150(&v49, v51, 0, 8);
        v41 = (unsigned int)v50;
      }
      v49[v41] = v40;
      v28 = a3[1];
      LODWORD(v50) = v50 + 1;
      v29 = sub_1643360(v10);
      v30 = sub_159C470(v29, v28, 0);
      v33 = sub_161BD20((__int64)&v48, v30, v31, v32);
      v34 = (unsigned int)v50;
      if ( (unsigned int)v50 >= HIDWORD(v50) )
      {
        v43 = v33;
        sub_16CD150(&v49, v51, 0, 8);
        v34 = (unsigned int)v50;
        v33 = v43;
      }
      v49[v34] = v33;
      v27 = (__int64 *)(unsigned int)(v50 + 1);
      LODWORD(v50) = v50 + 1;
      if ( v46 == a3 )
        break;
      a3 += 2;
    }
    while ( v26 != a3 );
  }
  v42 = sub_1627350(v10, v49, v27, 0, 1);
  sub_1625C10(a2, 2, v42);
  if ( v49 != (__int64 *)v51 )
    _libc_free((unsigned __int64)v49);
}
