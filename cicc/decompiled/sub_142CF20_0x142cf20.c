// Function: sub_142CF20
// Address: 0x142cf20
//
_QWORD *__fastcall sub_142CF20(
        __int64 a1,
        int a2,
        int a3,
        int a4,
        __int64 *a5,
        _QWORD *a6,
        _QWORD *a7,
        __int64 *a8,
        __int64 *a9,
        __int64 *a10,
        _QWORD *a11)
{
  __int64 v12; // r14
  __int64 v13; // r15
  __int64 v14; // r13
  _QWORD *result; // rax
  __int64 v16; // r8
  __int64 v17; // r8
  __int64 v18; // r14
  __int64 v19; // r9
  __int64 v20; // r11
  __int64 v21; // r8
  __int64 v22; // rdi
  __int64 v23; // rsi
  __int64 v24; // r10
  __int64 v25; // rsi
  __int64 v26; // r12
  __int64 v27; // r15
  __int64 v28; // r13
  __int64 v29; // rdx
  _QWORD *v30; // rax
  _QWORD *v31; // r14
  __int64 v32; // rbx
  __int64 v33; // r8
  __int64 v34; // r12
  __int64 v35; // rdi
  __int64 v36; // rbx
  __int64 v37; // r8
  __int64 v38; // r12
  __int64 v39; // rdi
  __int64 v40; // rdi
  __int64 v41; // rdi
  __int64 i; // rbx
  __int64 v43; // rdi
  __int64 j; // rbx
  __int64 v45; // rdi
  __int64 v46; // rdx
  __int64 v47; // rcx
  __int64 v48; // [rsp+8h] [rbp-88h]
  __int64 v49; // [rsp+10h] [rbp-80h]
  __int64 v50; // [rsp+10h] [rbp-80h]
  __int64 v51; // [rsp+10h] [rbp-80h]
  __int64 v52; // [rsp+18h] [rbp-78h]
  __int64 v53; // [rsp+18h] [rbp-78h]
  __int64 v54; // [rsp+20h] [rbp-70h]
  __int64 v55; // [rsp+20h] [rbp-70h]
  __int64 v56; // [rsp+28h] [rbp-68h]
  __int64 v57; // [rsp+30h] [rbp-60h]
  __int64 v58; // [rsp+30h] [rbp-60h]
  __int64 v59; // [rsp+38h] [rbp-58h]
  __int64 v60; // [rsp+38h] [rbp-58h]
  __int64 v61; // [rsp+40h] [rbp-50h]
  __int64 v62; // [rsp+48h] [rbp-48h]
  _QWORD *v63; // [rsp+50h] [rbp-40h]
  __int64 v64; // [rsp+58h] [rbp-38h]

  v12 = a5[1];
  v13 = *a5;
  a5[1] = 0;
  v14 = a5[2];
  a5[2] = 0;
  *a5 = 0;
  *(_QWORD *)(a1 + 48) = v12;
  *(_DWORD *)(a1 + 68) = a4;
  *(_DWORD *)(a1 + 12) = a2;
  result = a11;
  *(_DWORD *)(a1 + 8) = 1;
  *(_QWORD *)(a1 + 16) = 0;
  *(_QWORD *)(a1 + 24) = 0;
  *(_QWORD *)(a1 + 32) = 0;
  *(_QWORD *)(a1 + 40) = v13;
  *(_QWORD *)(a1 + 56) = v14;
  *(_QWORD *)a1 = &unk_49EB4B8;
  *(_DWORD *)(a1 + 64) = a3;
  *(_QWORD *)(a1 + 72) = *a6;
  v16 = a6[1];
  a6[1] = 0;
  *(_QWORD *)(a1 + 80) = v16;
  v17 = a6[2];
  *a6 = 0;
  a6[2] = 0;
  *(_QWORD *)(a1 + 88) = v17;
  *(_QWORD *)(a1 + 96) = 0;
  v18 = a7[1];
  v63 = (_QWORD *)*a7;
  if ( *a7 != v18 || a8[1] != *a8 || a9[1] != *a9 || a10[1] != *a10 || a11[1] != *a11 )
  {
    a7[1] = 0;
    v19 = a7[2];
    *a7 = 0;
    a7[2] = 0;
    v20 = a8[1];
    v21 = a8[2];
    a8[1] = 0;
    a8[2] = 0;
    v22 = *a8;
    *a8 = 0;
    v23 = *a9;
    v24 = a9[1];
    *a9 = 0;
    a9[1] = 0;
    v62 = v23;
    v25 = a9[2];
    a9[2] = 0;
    v48 = v19;
    v26 = a10[1];
    v64 = *a10;
    v49 = v20;
    v52 = v21;
    v54 = v24;
    v57 = a10[2];
    a10[2] = 0;
    a10[1] = 0;
    *a10 = 0;
    v27 = *a11;
    v61 = v22;
    v28 = a11[1];
    v29 = a11[2];
    a11[1] = 0;
    a11[2] = 0;
    *a11 = 0;
    v59 = v29;
    v30 = (_QWORD *)sub_22077B0(120);
    if ( v30 )
    {
      v30[8] = v25;
      v30[1] = v18;
      *v30 = v63;
      v30[2] = v48;
      v30[3] = v22;
      v30[4] = v49;
      v30[5] = v52;
      v30[6] = v62;
      v30[7] = v54;
      v30[9] = v64;
      v30[11] = v57;
      v30[14] = v59;
      v53 = 0;
      v55 = 0;
      v60 = 0;
      v56 = 0;
      v58 = 0;
      v64 = 0;
      v62 = 0;
      v61 = 0;
      v63 = 0;
      v30[10] = v26;
      v26 = 0;
      v30[12] = v27;
      v27 = 0;
      v30[13] = v28;
      v28 = 0;
    }
    else
    {
      v46 = v59 - v27;
      v47 = v57 - v64;
      v58 = v48 - (_QWORD)v63;
      v56 = v52 - v22;
      v60 = v25 - v62;
      v55 = v46;
      v53 = v47;
    }
    v31 = *(_QWORD **)(a1 + 96);
    *(_QWORD *)(a1 + 96) = v30;
    if ( v31 )
    {
      v32 = v31[13];
      v33 = v31[12];
      if ( v32 != v33 )
      {
        v50 = v26;
        v34 = v31[12];
        do
        {
          v35 = *(_QWORD *)(v34 + 16);
          if ( v35 )
            j_j___libc_free_0(v35, *(_QWORD *)(v34 + 32) - v35);
          v34 += 40;
        }
        while ( v32 != v34 );
        v26 = v50;
        v33 = v31[12];
      }
      if ( v33 )
        j_j___libc_free_0(v33, v31[14] - v33);
      v36 = v31[10];
      v37 = v31[9];
      if ( v36 != v37 )
      {
        v51 = v26;
        v38 = v31[9];
        do
        {
          v39 = *(_QWORD *)(v38 + 16);
          if ( v39 )
            j_j___libc_free_0(v39, *(_QWORD *)(v38 + 32) - v39);
          v38 += 40;
        }
        while ( v36 != v38 );
        v26 = v51;
        v37 = v31[9];
      }
      if ( v37 )
        j_j___libc_free_0(v37, v31[11] - v37);
      v40 = v31[6];
      if ( v40 )
        j_j___libc_free_0(v40, v31[8] - v40);
      v41 = v31[3];
      if ( v41 )
        j_j___libc_free_0(v41, v31[5] - v41);
      if ( *v31 )
        j_j___libc_free_0(*v31, v31[2] - *v31);
      j_j___libc_free_0(v31, 120);
    }
    for ( i = v27; v28 != i; i += 40 )
    {
      v43 = *(_QWORD *)(i + 16);
      if ( v43 )
        j_j___libc_free_0(v43, *(_QWORD *)(i + 32) - v43);
    }
    if ( v27 )
      j_j___libc_free_0(v27, v55);
    for ( j = v64; v26 != j; j += 40 )
    {
      v45 = *(_QWORD *)(j + 16);
      if ( v45 )
        j_j___libc_free_0(v45, *(_QWORD *)(j + 32) - v45);
    }
    if ( v64 )
      j_j___libc_free_0(v64, v53);
    if ( v62 )
      j_j___libc_free_0(v62, v60);
    if ( v61 )
      j_j___libc_free_0(v61, v56);
    result = v63;
    if ( v63 )
      return (_QWORD *)j_j___libc_free_0(v63, v58);
  }
  return result;
}
