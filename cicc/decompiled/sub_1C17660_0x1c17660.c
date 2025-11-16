// Function: sub_1C17660
// Address: 0x1c17660
//
__int64 *__fastcall sub_1C17660(__int64 *a1, _QWORD **a2, __int64 a3, __int64 a4, int a5, __m128i a6)
{
  __int64 *v9; // rax
  __int64 *v10; // r15
  __int64 v11; // rsi
  __int64 v12; // rax
  __int64 v13; // rbx
  _QWORD *v14; // rdx
  __int64 v15; // r14
  unsigned __int64 *v16; // r15
  unsigned __int64 *v17; // r13
  unsigned __int64 v18; // rdi
  unsigned __int64 *v19; // r15
  unsigned __int64 v20; // r13
  unsigned __int64 v21; // rdi
  unsigned __int64 v22; // rdi
  _QWORD **v23; // r15
  _BYTE *v24; // rbx
  unsigned __int64 v25; // r13
  __int64 v26; // rdi
  _QWORD **v29; // [rsp+38h] [rbp-1C8h] BYREF
  __m128i *v30; // [rsp+40h] [rbp-1C0h] BYREF
  unsigned __int64 v31; // [rsp+48h] [rbp-1B8h]
  __int64 v32; // [rsp+50h] [rbp-1B0h]
  __int64 v33; // [rsp+58h] [rbp-1A8h]
  _QWORD v34[2]; // [rsp+60h] [rbp-1A0h] BYREF
  _QWORD *v35; // [rsp+70h] [rbp-190h]
  __int64 v36; // [rsp+78h] [rbp-188h]
  _QWORD v37[3]; // [rsp+80h] [rbp-180h] BYREF
  int v38; // [rsp+98h] [rbp-168h]
  _QWORD *v39; // [rsp+A0h] [rbp-160h]
  __int64 v40; // [rsp+A8h] [rbp-158h]
  _QWORD v41[2]; // [rsp+B0h] [rbp-150h] BYREF
  _QWORD *v42; // [rsp+C0h] [rbp-140h]
  __int64 v43; // [rsp+C8h] [rbp-138h]
  _QWORD v44[2]; // [rsp+D0h] [rbp-130h] BYREF
  __int64 v45; // [rsp+E0h] [rbp-120h]
  __int64 v46; // [rsp+E8h] [rbp-118h]
  __int64 v47; // [rsp+F0h] [rbp-110h]
  _BYTE *v48; // [rsp+F8h] [rbp-108h]
  __int64 v49; // [rsp+100h] [rbp-100h]
  _BYTE v50[248]; // [rsp+108h] [rbp-F8h] BYREF

  if ( !a4 )
  {
    *a1 = 0;
    return a1;
  }
  v35 = v37;
  v39 = v41;
  v42 = v44;
  v34[0] = 0;
  v34[1] = 0;
  v36 = 0;
  LOBYTE(v37[0]) = 0;
  v37[2] = 0;
  v38 = 0;
  v40 = 0;
  LOBYTE(v41[0]) = 0;
  v43 = 0;
  LOBYTE(v44[0]) = 0;
  v45 = 0;
  v46 = 0;
  v47 = 0;
  v48 = v50;
  v49 = 0x400000000LL;
  v9 = (__int64 *)sub_22077B0(8);
  v10 = v9;
  if ( v9 )
  {
    sub_1602D10(v9);
    sub_16C2FC0(&v30, *a2);
    sub_166F050(&v29, (__int64)v34, (__int64)v10, 1u, (__int8 *)byte_3F871B3, 0, a6, v30, v31, v32, v33);
    v11 = (__int64)v29;
    v29 = 0;
    if ( v11 )
    {
LABEL_4:
      v12 = sub_22077B0(152);
      v13 = v12;
      if ( v12 )
        sub_1C17480(v12, v11, a3, a4, a5, 1, 1);
      v14 = (_QWORD *)sub_22077B0(104);
      if ( v14 )
      {
        memset(v14, 0, 0x68u);
        v14[11] = 1;
        v14[2] = v14 + 4;
        v14[3] = 0x400000000LL;
        v14[8] = v14 + 10;
      }
      v15 = *(_QWORD *)(v13 + 144);
      *(_QWORD *)(v13 + 144) = v14;
      if ( v15 )
      {
        v16 = *(unsigned __int64 **)(v15 + 16);
        v17 = &v16[*(unsigned int *)(v15 + 24)];
        while ( v17 != v16 )
        {
          v18 = *v16++;
          _libc_free(v18);
        }
        v19 = *(unsigned __int64 **)(v15 + 64);
        v20 = (unsigned __int64)&v19[2 * *(unsigned int *)(v15 + 72)];
        if ( v19 != (unsigned __int64 *)v20 )
        {
          do
          {
            v21 = *v19;
            v19 += 2;
            _libc_free(v21);
          }
          while ( (unsigned __int64 *)v20 != v19 );
          v20 = *(_QWORD *)(v15 + 64);
        }
        if ( v20 != v15 + 80 )
          _libc_free(v20);
        v22 = *(_QWORD *)(v15 + 16);
        if ( v22 != v15 + 32 )
          _libc_free(v22);
        j_j___libc_free_0(v15, 104);
      }
      *a1 = v13;
      v23 = v29;
      goto LABEL_22;
    }
    sub_16025D0(v10);
    j_j___libc_free_0(v10, 8);
    v23 = v29;
  }
  else
  {
    v23 = 0;
    sub_16C2FC0(&v30, *a2);
    sub_166F050(&v29, (__int64)v34, 0, 1u, (__int8 *)byte_3F871B3, 0, a6, v30, v31, v32, v33);
    v11 = (__int64)v29;
    v29 = 0;
    if ( v11 )
      goto LABEL_4;
  }
  *a1 = 0;
LABEL_22:
  if ( v23 )
  {
    sub_1633490(v23);
    j_j___libc_free_0(v23, 736);
  }
  v24 = v48;
  v25 = (unsigned __int64)&v48[48 * (unsigned int)v49];
  if ( v48 != (_BYTE *)v25 )
  {
    do
    {
      v25 -= 48LL;
      v26 = *(_QWORD *)(v25 + 16);
      if ( v26 != v25 + 32 )
        j_j___libc_free_0(v26, *(_QWORD *)(v25 + 32) + 1LL);
    }
    while ( v24 != (_BYTE *)v25 );
    v25 = (unsigned __int64)v48;
  }
  if ( (_BYTE *)v25 != v50 )
    _libc_free(v25);
  if ( v45 )
    j_j___libc_free_0(v45, v47 - v45);
  if ( v42 != v44 )
    j_j___libc_free_0(v42, v44[0] + 1LL);
  if ( v39 != v41 )
    j_j___libc_free_0(v39, v41[0] + 1LL);
  if ( v35 != v37 )
    j_j___libc_free_0(v35, v37[0] + 1LL);
  return a1;
}
