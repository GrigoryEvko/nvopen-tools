// Function: sub_2428400
// Address: 0x2428400
//
__int64 __fastcall sub_2428400(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        unsigned __int64 *a6,
        __int64 a7)
{
  __int64 v7; // r10
  unsigned __int64 *v8; // r15
  unsigned __int64 *v9; // r14
  unsigned __int64 *v12; // rbx
  __int64 v13; // r11
  __int64 v15; // rcx
  unsigned __int64 *v16; // rdx
  unsigned __int64 *v17; // r12
  unsigned __int64 v18; // rsi
  unsigned __int64 v19; // rdi
  __int64 v20; // rdx
  __int64 v21; // r12
  unsigned __int64 v22; // rdx
  unsigned __int64 v23; // rdi
  __int64 v24; // rbx
  __int64 v25; // r15
  unsigned __int64 *v26; // r12
  __int64 v27; // r14
  unsigned __int64 v28; // rax
  unsigned __int64 v29; // rdi
  __int64 v30; // rax
  __int64 v31; // rcx
  unsigned __int64 *v32; // rdx
  unsigned __int64 *v33; // r15
  unsigned __int64 v34; // rsi
  unsigned __int64 v35; // rdi
  __int64 v36; // rdx
  __int64 v37; // r15
  __int64 v38; // r13
  __int64 v39; // rdx
  unsigned __int64 v40; // rdi
  __int64 v41; // r13
  unsigned __int64 *v42; // rbx
  unsigned __int64 v43; // rax
  unsigned __int64 v44; // rdi
  __int64 v45; // rbx
  __int64 v46; // [rsp+8h] [rbp-68h]
  __int64 v47; // [rsp+10h] [rbp-60h]
  __int64 v48; // [rsp+10h] [rbp-60h]
  __int64 v49; // [rsp+18h] [rbp-58h]
  __int64 v50; // [rsp+18h] [rbp-58h]
  __int64 v51; // [rsp+20h] [rbp-50h]
  __int64 v52; // [rsp+20h] [rbp-50h]
  unsigned __int64 *v53; // [rsp+28h] [rbp-48h]
  __int64 v54; // [rsp+28h] [rbp-48h]
  unsigned __int64 *v55; // [rsp+28h] [rbp-48h]
  __int64 v56; // [rsp+28h] [rbp-48h]
  __int64 v57; // [rsp+30h] [rbp-40h]
  __int64 v58; // [rsp+30h] [rbp-40h]
  __int64 v59; // [rsp+30h] [rbp-40h]
  __int64 v60; // [rsp+38h] [rbp-38h]
  __int64 v61; // [rsp+38h] [rbp-38h]
  __int64 v62; // [rsp+38h] [rbp-38h]

  v7 = a2;
  v8 = (unsigned __int64 *)a2;
  v9 = a6;
  v12 = (unsigned __int64 *)a1;
  v60 = a3;
  if ( a4 > a5 && a5 <= a7 )
  {
    v13 = a1;
    if ( !a5 )
      return v13;
    v50 = a2 - a1;
    v30 = (a2 - a1) >> 3;
    v48 = a3 - a2;
    v31 = (a3 - a2) >> 3;
    v46 = v30;
    if ( a3 - a2 <= 0 )
    {
      if ( v50 <= 0 )
        return (__int64)v12;
      v59 = 0;
      v56 = 0;
    }
    else
    {
      v32 = (unsigned __int64 *)a2;
      v33 = a6;
      do
      {
        v34 = *v32;
        *v32 = 0;
        v35 = *v33;
        *v33 = v34;
        if ( v35 )
        {
          v52 = v7;
          v55 = v32;
          v58 = v30;
          v61 = v31;
          j_j___libc_free_0(v35);
          v7 = v52;
          v32 = v55;
          v30 = v58;
          v31 = v61;
        }
        ++v32;
        ++v33;
        --v31;
      }
      while ( v31 );
      v36 = 8;
      if ( v48 > 0 )
        v36 = v48;
      v56 = v36;
      v59 = v36 >> 3;
      if ( v50 <= 0 )
        goto LABEL_39;
    }
    v37 = v7 - 8 * v46;
    v38 = -8 * v46 + a3;
    do
    {
      v39 = *(_QWORD *)(v37 + 8 * v30 - 8);
      *(_QWORD *)(v37 + 8 * v30 - 8) = 0;
      v40 = *(_QWORD *)(v38 + 8 * v30 - 8);
      *(_QWORD *)(v38 + 8 * v30 - 8) = v39;
      if ( v40 )
      {
        v62 = v30;
        j_j___libc_free_0(v40);
        v30 = v62;
      }
      --v30;
    }
    while ( v30 );
LABEL_39:
    if ( v56 > 0 )
    {
      v41 = v59;
      v42 = (unsigned __int64 *)a1;
      do
      {
        v43 = *v9;
        *v9 = 0;
        v44 = *v42;
        *v42 = v43;
        if ( v44 )
          j_j___libc_free_0(v44);
        ++v9;
        ++v42;
        --v41;
      }
      while ( v41 );
      v45 = 8 * v59;
      if ( v59 <= 0 )
        v45 = 8;
      return a1 + v45;
    }
    return (__int64)v12;
  }
  if ( a4 <= a7 )
  {
    v13 = a3;
    if ( !a4 )
      return v13;
    v49 = a3 - a2;
    v47 = a2 - a1;
    v15 = (a2 - a1) >> 3;
    v57 = (a3 - a2) >> 3;
    if ( a2 - a1 <= 0 )
    {
      if ( v49 <= 0 )
        return v60;
      v54 = 0;
      v21 = 0;
    }
    else
    {
      v16 = (unsigned __int64 *)a1;
      v17 = a6;
      do
      {
        v18 = *v16;
        *v16 = 0;
        v19 = *v17;
        *v17 = v18;
        if ( v19 )
        {
          v51 = v15;
          v53 = v16;
          j_j___libc_free_0(v19);
          v15 = v51;
          v16 = v53;
        }
        ++v16;
        ++v17;
        --v15;
      }
      while ( v15 );
      v20 = 8;
      if ( v47 > 0 )
        v20 = v47;
      v9 = (unsigned __int64 *)((char *)v9 + v20);
      v21 = v20;
      v54 = v20 >> 3;
      if ( v49 <= 0 )
        goto LABEL_18;
    }
    do
    {
      v22 = *v8;
      *v8 = 0;
      v23 = *v12;
      *v12 = v22;
      if ( v23 )
        j_j___libc_free_0(v23);
      ++v8;
      ++v12;
      --v57;
    }
    while ( v57 );
LABEL_18:
    if ( v21 > 0 )
    {
      v24 = v54;
      v25 = -8 * v54;
      v26 = &v9[-v54];
      v27 = a3 - 8 * v54;
      do
      {
        v28 = v26[v24 - 1];
        v26[v24 - 1] = 0;
        v29 = *(_QWORD *)(v27 + 8 * v24 - 8);
        *(_QWORD *)(v27 + 8 * v24 - 8) = v28;
        if ( v29 )
          j_j___libc_free_0(v29);
        --v24;
      }
      while ( v24 );
      if ( v54 <= 0 )
        v25 = -8;
      return a3 + v25;
    }
    return v60;
  }
  return sub_24252C0(a1, a2, a3);
}
