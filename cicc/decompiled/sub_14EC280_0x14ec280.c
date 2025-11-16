// Function: sub_14EC280
// Address: 0x14ec280
//
__int64 *__fastcall sub_14EC280(
        __int64 *a1,
        int a2,
        int a3,
        int a4,
        __int64 *a5,
        __int64 *a6,
        __int64 *a7,
        __int64 *a8,
        __int64 *a9,
        __int64 *a10,
        __int64 *a11)
{
  __int64 v15; // r10
  __int64 v16; // r10
  __int64 v17; // r10
  __int64 v18; // r8
  __int64 v19; // r8
  __int64 v20; // r8
  __int64 v21; // r8
  __int64 v22; // r8
  __int64 v23; // r8
  __int64 v24; // rdi
  __int64 v25; // rdi
  __int64 v26; // rdi
  __int64 v27; // rsi
  __int64 v28; // rsi
  __int64 v29; // rsi
  __int64 v30; // rcx
  __int64 v31; // rcx
  __int64 v32; // rcx
  __int64 v33; // rdx
  __int64 v34; // rdx
  __int64 v35; // rdx
  __int64 v36; // rax
  __int64 v37; // rbx
  __int64 v38; // r12
  __int64 v39; // rbx
  __int64 v40; // rdi
  __int64 v41; // rbx
  __int64 v42; // r12
  __int64 v43; // rdi
  __int64 v45; // [rsp+0h] [rbp-110h] BYREF
  __int64 v46; // [rsp+8h] [rbp-108h]
  __int64 v47; // [rsp+10h] [rbp-100h]
  __int64 v48; // [rsp+20h] [rbp-F0h] BYREF
  __int64 v49; // [rsp+28h] [rbp-E8h]
  __int64 v50; // [rsp+30h] [rbp-E0h]
  __int64 v51[2]; // [rsp+40h] [rbp-D0h] BYREF
  __int64 v52; // [rsp+50h] [rbp-C0h]
  __int64 v53[2]; // [rsp+60h] [rbp-B0h] BYREF
  __int64 v54; // [rsp+70h] [rbp-A0h]
  _QWORD v55[2]; // [rsp+80h] [rbp-90h] BYREF
  __int64 v56; // [rsp+90h] [rbp-80h]
  _QWORD v57[2]; // [rsp+A0h] [rbp-70h] BYREF
  __int64 v58; // [rsp+B0h] [rbp-60h]
  __int64 v59[2]; // [rsp+C0h] [rbp-50h] BYREF
  __int64 v60; // [rsp+D0h] [rbp-40h]

  v15 = *a5;
  *a5 = 0;
  v59[0] = v15;
  v16 = a5[1];
  a5[1] = 0;
  v59[1] = v16;
  v17 = a5[2];
  a5[2] = 0;
  v18 = *a6;
  *a6 = 0;
  v57[0] = v18;
  v19 = a6[1];
  a6[1] = 0;
  v57[1] = v19;
  v20 = a6[2];
  a6[2] = 0;
  v58 = v20;
  v21 = *a7;
  v60 = v17;
  v55[0] = v21;
  v22 = a7[1];
  a7[1] = 0;
  *a7 = 0;
  v55[1] = v22;
  v23 = a7[2];
  a7[2] = 0;
  v24 = *a8;
  *a8 = 0;
  v53[0] = v24;
  v25 = a8[1];
  a8[1] = 0;
  v53[1] = v25;
  v26 = a8[2];
  a8[2] = 0;
  v27 = *a9;
  *a9 = 0;
  v51[0] = v27;
  v28 = a9[1];
  a9[1] = 0;
  v51[1] = v28;
  v29 = a9[2];
  a9[2] = 0;
  v30 = *a10;
  v56 = v23;
  v48 = v30;
  v31 = a10[1];
  v54 = v26;
  v49 = v31;
  v32 = a10[2];
  v52 = v29;
  v50 = v32;
  a10[2] = 0;
  a10[1] = 0;
  *a10 = 0;
  v33 = *a11;
  *a11 = 0;
  v45 = v33;
  v34 = a11[1];
  a11[1] = 0;
  v46 = v34;
  v35 = a11[2];
  a11[2] = 0;
  v47 = v35;
  v36 = sub_22077B0(104);
  v37 = v36;
  if ( v36 )
    sub_142CF20(v36, a2, a3, a4, v59, v57, v55, v53, v51, &v48, &v45);
  *a1 = v37;
  v38 = v45;
  v39 = v46;
  if ( v46 != v45 )
  {
    do
    {
      v40 = *(_QWORD *)(v38 + 16);
      if ( v40 )
        j_j___libc_free_0(v40, *(_QWORD *)(v38 + 32) - v40);
      v38 += 40;
    }
    while ( v39 != v38 );
    v38 = v45;
  }
  if ( v38 )
    j_j___libc_free_0(v38, v47 - v38);
  v41 = v49;
  v42 = v48;
  if ( v49 != v48 )
  {
    do
    {
      v43 = *(_QWORD *)(v42 + 16);
      if ( v43 )
        j_j___libc_free_0(v43, *(_QWORD *)(v42 + 32) - v43);
      v42 += 40;
    }
    while ( v41 != v42 );
    v42 = v48;
  }
  if ( v42 )
    j_j___libc_free_0(v42, v50 - v42);
  if ( v51[0] )
    j_j___libc_free_0(v51[0], v52 - v51[0]);
  if ( v53[0] )
    j_j___libc_free_0(v53[0], v54 - v53[0]);
  if ( v55[0] )
    j_j___libc_free_0(v55[0], v56 - v55[0]);
  if ( v57[0] )
    j_j___libc_free_0(v57[0], v58 - v57[0]);
  if ( v59[0] )
    j_j___libc_free_0(v59[0], v60 - v59[0]);
  return a1;
}
