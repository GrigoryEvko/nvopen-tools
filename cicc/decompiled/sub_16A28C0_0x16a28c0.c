// Function: sub_16A28C0
// Address: 0x16a28c0
//
_QWORD *__fastcall sub_16A28C0(_QWORD *a1, __int64 a2, unsigned int a3, unsigned int a4)
{
  __int64 *v5; // rsi
  void *v6; // rbx
  __int64 *v7; // rsi
  __int64 v9; // rax
  __int64 v10; // rsi
  __int64 v11; // r12
  __int64 v12; // rax
  __int64 v13; // rsi
  __int64 v14; // r12
  __int64 v15; // rax
  __int64 v16; // rsi
  __int64 v17; // r12
  __int64 v18; // rdx
  __int64 v19; // rsi
  __int64 v20; // rax
  __int64 v21; // rax
  __int64 v22; // rdx
  __int64 v23; // rsi
  __int64 v24; // rax
  __int64 v25; // rax
  __int64 v26; // rdx
  __int64 v27; // rsi
  __int64 v28; // rax
  __int64 v29; // rax
  __int64 v30; // [rsp+0h] [rbp-140h]
  __int64 v31; // [rsp+0h] [rbp-140h]
  __int64 v32; // [rsp+0h] [rbp-140h]
  __int16 *v33; // [rsp+8h] [rbp-138h]
  __int16 *v34; // [rsp+8h] [rbp-138h]
  __int64 v35; // [rsp+8h] [rbp-138h]
  __int64 v36; // [rsp+8h] [rbp-138h]
  __int64 v37; // [rsp+8h] [rbp-138h]
  __int64 v39; // [rsp+20h] [rbp-120h]
  __int64 v40; // [rsp+20h] [rbp-120h]
  __int64 v41; // [rsp+20h] [rbp-120h]
  __int16 *v43; // [rsp+30h] [rbp-110h] BYREF
  __int64 v44; // [rsp+38h] [rbp-108h]
  __int64 v45; // [rsp+50h] [rbp-F0h] BYREF
  __int64 v46; // [rsp+58h] [rbp-E8h]
  __int16 *v47; // [rsp+70h] [rbp-D0h] BYREF
  __int64 v48; // [rsp+78h] [rbp-C8h]
  __int64 v49; // [rsp+90h] [rbp-B0h] BYREF
  __int64 v50[3]; // [rsp+98h] [rbp-A8h] BYREF
  __int64 v51; // [rsp+B0h] [rbp-90h] BYREF
  _QWORD v52[4]; // [rsp+B8h] [rbp-88h] BYREF
  __int64 v53[3]; // [rsp+D8h] [rbp-68h] BYREF
  char v54[8]; // [rsp+F0h] [rbp-50h] BYREF
  _QWORD v55[9]; // [rsp+F8h] [rbp-48h] BYREF

  v5 = (__int64 *)(*(_QWORD *)(a2 + 8) + 40LL);
  v6 = sub_16982C0();
  if ( (void *)*v5 == v6 )
    sub_169C6E0(v53, (__int64)v5);
  else
    sub_16986C0(v53, v5);
  v33 = (__int16 *)v53[0];
  if ( (void *)v53[0] == v6 )
  {
    sub_169C6E0(&v47, (__int64)v53);
    sub_16A28C0(&v49, &v47, a3, a4);
    sub_169C7E0(&v51, &v49);
    sub_169C7E0(v55, &v51);
    v18 = v52[0];
    if ( v52[0] )
    {
      v19 = 32LL * *(_QWORD *)(v52[0] - 8LL);
      v20 = v52[0] + v19;
      if ( v52[0] != v52[0] + v19 )
      {
        do
        {
          v21 = v20 - 32;
          v30 = v18;
          v35 = v21;
          if ( v6 == *(void **)(v21 + 8) )
          {
            sub_169DEB0((__int64 *)(v21 + 16));
            v18 = v30;
            v20 = v35;
          }
          else
          {
            sub_1698460(v21 + 8);
            v20 = v35;
            v18 = v30;
          }
        }
        while ( v18 != v20 );
      }
      j_j_j___libc_free_0_0(v18 - 8);
    }
    v22 = v50[0];
    if ( v50[0] )
    {
      v23 = 32LL * *(_QWORD *)(v50[0] - 8);
      v24 = v50[0] + v23;
      if ( v50[0] != v50[0] + v23 )
      {
        do
        {
          v25 = v24 - 32;
          v31 = v22;
          v36 = v25;
          if ( v6 == *(void **)(v25 + 8) )
          {
            sub_169DEB0((__int64 *)(v25 + 16));
            v22 = v31;
            v24 = v36;
          }
          else
          {
            sub_1698460(v25 + 8);
            v24 = v36;
            v22 = v31;
          }
        }
        while ( v22 != v24 );
      }
      j_j_j___libc_free_0_0(v22 - 8);
    }
    v26 = v48;
    if ( v48 )
    {
      v27 = 32LL * *(_QWORD *)(v48 - 8);
      v28 = v48 + v27;
      if ( v48 != v48 + v27 )
      {
        do
        {
          v29 = v28 - 32;
          v32 = v26;
          v37 = v29;
          if ( v6 == *(void **)(v29 + 8) )
          {
            sub_169DEB0((__int64 *)(v29 + 16));
            v26 = v32;
            v28 = v37;
          }
          else
          {
            sub_1698460(v29 + 8);
            v28 = v37;
            v26 = v32;
          }
        }
        while ( v26 != v28 );
      }
      j_j_j___libc_free_0_0(v26 - 8);
    }
  }
  else
  {
    sub_16986C0(&v47, v53);
    sub_169C390((__int64)&v49, &v47, a3, a4);
    sub_1698450((__int64)&v51, (__int64)&v49);
    sub_169E320(v55, &v51, v33);
    sub_1698460((__int64)&v51);
    sub_1698460((__int64)&v49);
    sub_1698460((__int64)&v47);
  }
  v7 = (__int64 *)(*(_QWORD *)(a2 + 8) + 8LL);
  if ( (void *)*v7 == v6 )
    sub_169C6E0(v50, (__int64)v7);
  else
    sub_16986C0(v50, v7);
  v34 = (__int16 *)v50[0];
  if ( (void *)v50[0] == v6 )
  {
    sub_169C6E0(&v43, (__int64)v50);
    sub_16A28C0(&v45, &v43, a3, a4);
    sub_169C7E0(&v47, &v45);
    sub_169C7E0(v52, &v47);
    v9 = v48;
    if ( v48 )
    {
      v10 = 32LL * *(_QWORD *)(v48 - 8);
      v11 = v48 + v10;
      if ( v48 != v48 + v10 )
      {
        do
        {
          v11 -= 32;
          v39 = v9;
          if ( v6 == *(void **)(v11 + 8) )
            sub_169DEB0((__int64 *)(v11 + 16));
          else
            sub_1698460(v11 + 8);
          v9 = v39;
        }
        while ( v39 != v11 );
      }
      j_j_j___libc_free_0_0(v9 - 8);
    }
    v12 = v46;
    if ( v46 )
    {
      v13 = 32LL * *(_QWORD *)(v46 - 8);
      v14 = v46 + v13;
      if ( v46 != v46 + v13 )
      {
        do
        {
          v14 -= 32;
          v40 = v12;
          if ( v6 == *(void **)(v14 + 8) )
            sub_169DEB0((__int64 *)(v14 + 16));
          else
            sub_1698460(v14 + 8);
          v12 = v40;
        }
        while ( v40 != v14 );
      }
      j_j_j___libc_free_0_0(v12 - 8);
    }
    v15 = v44;
    if ( v44 )
    {
      v16 = 32LL * *(_QWORD *)(v44 - 8);
      v17 = v44 + v16;
      if ( v44 != v44 + v16 )
      {
        do
        {
          v17 -= 32;
          v41 = v15;
          if ( v6 == *(void **)(v17 + 8) )
            sub_169DEB0((__int64 *)(v17 + 16));
          else
            sub_1698460(v17 + 8);
          v15 = v41;
        }
        while ( v41 != v17 );
      }
      j_j_j___libc_free_0_0(v15 - 8);
    }
  }
  else
  {
    sub_16986C0(&v43, v50);
    sub_169C390((__int64)&v45, &v43, a3, a4);
    sub_1698450((__int64)&v47, (__int64)&v45);
    sub_169E320(v52, (__int64 *)&v47, v34);
    sub_1698460((__int64)&v47);
    sub_1698460((__int64)&v45);
    sub_1698460((__int64)&v43);
  }
  sub_169C810(a1, (__int64)&unk_42AE990, (__int64)&v51, (__int64)v54);
  sub_127D120(v52);
  sub_127D120(v50);
  sub_127D120(v55);
  sub_127D120(v53);
  return a1;
}
