// Function: sub_386C330
// Address: 0x386c330
//
__int64 __fastcall sub_386C330(
        __int64 a1,
        __int64 a2,
        __m128 a3,
        double a4,
        double a5,
        double a6,
        double a7,
        double a8,
        double a9,
        __m128 a10)
{
  __int64 v11; // r14
  double v12; // xmm4_8
  double v13; // xmm5_8
  int v14; // eax
  unsigned __int64 *v15; // rdi
  _QWORD *v16; // rax
  bool v17; // zf
  _QWORD *v18; // rax
  _QWORD *v19; // r14
  _BYTE *v20; // r15
  _QWORD *v21; // rcx
  unsigned __int64 v22; // r13
  __int64 v23; // rsi
  __int64 *v24; // rax
  __int64 *v25; // rsi
  _QWORD *v26; // rdx
  _QWORD *v27; // rdi
  __int64 v28; // r8
  _QWORD *v29; // rbx
  __int64 v30; // r13
  __int64 v31; // rax
  _QWORD *v33; // rdx
  _QWORD *v34; // r9
  __int64 v35; // r8
  __int64 v36; // rdi
  _QWORD *v37; // [rsp+0h] [rbp-150h]
  __int64 v38; // [rsp+8h] [rbp-148h]
  _QWORD v39[2]; // [rsp+10h] [rbp-140h] BYREF
  __int64 v40; // [rsp+20h] [rbp-130h]
  _QWORD v41[2]; // [rsp+30h] [rbp-120h] BYREF
  _QWORD *v42; // [rsp+40h] [rbp-110h]
  _BYTE *v43; // [rsp+50h] [rbp-100h] BYREF
  __int64 v44; // [rsp+58h] [rbp-F8h]
  _BYTE v45[240]; // [rsp+60h] [rbp-F0h] BYREF

  if ( !a2 )
    return 0;
  v40 = a2;
  v39[0] = 6;
  v39[1] = 0;
  if ( a2 != -16 && a2 != -8 )
    sub_164C220((__int64)v39);
  v11 = *(_QWORD *)(a2 + 8);
  v43 = v45;
  v44 = 0x800000000LL;
  if ( !v11 )
  {
    v19 = v45;
    goto LABEL_67;
  }
  do
  {
    v18 = sub_1648700(v11);
    v41[0] = 6;
    v41[1] = 0;
    if ( !v18 )
    {
      v42 = 0;
      v14 = v44;
      if ( (unsigned int)v44 < HIDWORD(v44) )
        goto LABEL_11;
LABEL_22:
      sub_386C170((__int64)&v43, 0);
      v14 = v44;
      goto LABEL_11;
    }
    v42 = v18;
    if ( v18 != (_QWORD *)-16LL && v18 != (_QWORD *)-8LL )
      sub_164C220((__int64)v41);
    v14 = v44;
    if ( (unsigned int)v44 >= HIDWORD(v44) )
      goto LABEL_22;
LABEL_11:
    v15 = (unsigned __int64 *)&v43[24 * v14];
    if ( v15 )
    {
      *v15 = 6;
      v15[1] = 0;
      v16 = v42;
      v17 = v42 + 1 == 0;
      v15[2] = (unsigned __int64)v42;
      if ( v16 != 0 && !v17 && v16 != (_QWORD *)-16LL )
        sub_1649AC0(v15, v41[0] & 0xFFFFFFFFFFFFFFF8LL);
      v14 = v44;
    }
    LODWORD(v44) = v14 + 1;
    if ( v42 != 0 && v42 + 1 != 0 && v42 != (_QWORD *)-16LL )
      sub_1649B30(v41);
    v11 = *(_QWORD *)(v11 + 8);
  }
  while ( v11 );
  v19 = v43;
  v20 = &v43[24 * (unsigned int)v44];
  if ( v43 == v20 )
  {
LABEL_67:
    v30 = v40;
    goto LABEL_50;
  }
  v21 = (_QWORD *)(a1 + 600);
  do
  {
    v22 = v19[2];
    if ( *(_BYTE *)(v22 + 16) != 23 )
      goto LABEL_43;
    v23 = 3LL * (*(_DWORD *)(v22 + 20) & 0xFFFFFFF);
    if ( (*(_BYTE *)(v22 + 23) & 0x40) != 0 )
    {
      v24 = *(__int64 **)(v22 - 8);
      v25 = &v24[v23];
    }
    else
    {
      v24 = (__int64 *)(v22 - v23 * 8);
      v25 = (__int64 *)v19[2];
    }
    if ( *(_QWORD *)(a1 + 632) )
    {
      v33 = *(_QWORD **)(a1 + 608);
      if ( !v33 )
        goto LABEL_34;
      v34 = v21;
      do
      {
        while ( 1 )
        {
          v35 = v33[2];
          v36 = v33[3];
          if ( v22 <= v33[4] )
            break;
          v33 = (_QWORD *)v33[3];
          if ( !v36 )
            goto LABEL_61;
        }
        v34 = v33;
        v33 = (_QWORD *)v33[2];
      }
      while ( v35 );
LABEL_61:
      if ( v21 == v34 || v22 < v34[4] )
      {
LABEL_34:
        if ( v24 != v25 )
        {
          v28 = 0;
          do
          {
            if ( *v24 != v28 && v22 != *v24 )
            {
              if ( v28 )
                goto LABEL_43;
              v28 = *v24;
            }
            v24 += 3;
          }
          while ( v25 != v24 );
          if ( v28 )
          {
            v37 = v21;
            v38 = v28;
            sub_164D160(v19[2], v28, a3, a4, a5, a6, v12, v13, a9, a10);
            sub_386B550((__int64 *)a1, v22);
            sub_386C330(a1, v38);
            v21 = v37;
          }
        }
      }
    }
    else
    {
      v26 = *(_QWORD **)(a1 + 512);
      v27 = &v26[*(unsigned int *)(a1 + 520)];
      if ( v26 == v27 )
        goto LABEL_34;
      while ( v22 != *v26 )
      {
        if ( v27 == ++v26 )
          goto LABEL_34;
      }
      if ( v27 == v26 )
        goto LABEL_34;
    }
LABEL_43:
    v19 += 3;
  }
  while ( v19 != (_QWORD *)v20 );
  v29 = v43;
  v30 = v40;
  v19 = &v43[24 * (unsigned int)v44];
  if ( v43 != (_BYTE *)v19 )
  {
    do
    {
      v31 = *(v19 - 1);
      v19 -= 3;
      if ( v31 != 0 && v31 != -8 && v31 != -16 )
        sub_1649B30(v19);
    }
    while ( v29 != v19 );
    v19 = v43;
  }
LABEL_50:
  if ( v19 != (_QWORD *)v45 )
    _libc_free((unsigned __int64)v19);
  if ( v40 != -8 && v40 != 0 && v40 != -16 )
    sub_1649B30(v39);
  return v30;
}
