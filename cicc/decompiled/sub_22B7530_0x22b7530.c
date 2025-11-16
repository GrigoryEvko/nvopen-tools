// Function: sub_22B7530
// Address: 0x22b7530
//
__int64 __fastcall sub_22B7530(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int64 a7,
        __int64 a8,
        __int64 a9,
        __int64 a10,
        __int64 **a11,
        __int64 a12)
{
  __int64 *v13; // r12
  __int64 v14; // rax
  __int64 *v15; // r13
  unsigned int v16; // esi
  __int64 v17; // rcx
  int v18; // eax
  unsigned int v19; // edx
  _DWORD *v20; // r8
  int v21; // edi
  unsigned int v22; // edx
  __int64 v23; // rsi
  __int64 v24; // rcx
  unsigned int v25; // eax
  __int64 *v26; // rbx
  __int64 v27; // r8
  unsigned int v28; // esi
  int *v29; // r14
  int v30; // eax
  unsigned int v31; // edx
  _DWORD *v32; // rdi
  int v33; // ecx
  __int64 v34; // rdx
  __int64 v35; // rdi
  __int64 v36; // r8
  unsigned int v37; // eax
  __int64 *v38; // rbx
  __int64 v39; // r9
  _DWORD *v40; // r10
  int v41; // eax
  _DWORD *v42; // r10
  int v43; // edx
  int v44; // ebx
  unsigned int v45; // r12d
  int v47; // r9d
  int v48; // r14d
  __int64 v49; // [rsp+0h] [rbp-B0h]
  __int64 v50; // [rsp+8h] [rbp-A8h]
  int v51; // [rsp+8h] [rbp-A8h]
  int v52; // [rsp+8h] [rbp-A8h]
  __int64 *v53; // [rsp+28h] [rbp-88h]
  _DWORD *v54; // [rsp+38h] [rbp-78h] BYREF
  __int64 v55; // [rsp+40h] [rbp-70h] BYREF
  __int64 v56; // [rsp+48h] [rbp-68h]
  __int64 v57; // [rsp+50h] [rbp-60h]
  __int64 v58; // [rsp+58h] [rbp-58h]
  __int64 v59; // [rsp+60h] [rbp-50h] BYREF
  __int64 v60; // [rsp+68h] [rbp-48h]
  __int64 v61; // [rsp+70h] [rbp-40h]
  __int64 v62; // [rsp+78h] [rbp-38h]

  v55 = 0;
  v56 = 0;
  v13 = *(__int64 **)a8;
  v14 = *(_QWORD *)(a8 + 8);
  v57 = 0;
  v58 = 0;
  v15 = *a11;
  v59 = 0;
  v60 = 0;
  v61 = 0;
  v62 = 0;
  if ( (_DWORD)v14 )
  {
    v16 = 0;
    v17 = 0;
    v53 = &v13[(unsigned int)v14];
    while ( 1 )
    {
      v34 = *(unsigned int *)(a7 + 48);
      v35 = *v13;
      v36 = *(_QWORD *)(a7 + 32);
      if ( (_DWORD)v34 )
      {
        v37 = (v34 - 1) & (((unsigned int)v35 >> 9) ^ ((unsigned int)v35 >> 4));
        v38 = (__int64 *)(v36 + 16LL * v37);
        v39 = *v38;
        if ( v35 == *v38 )
          goto LABEL_12;
        v44 = 1;
        while ( v39 != -4096 )
        {
          v48 = v44 + 1;
          v37 = (v34 - 1) & (v44 + v37);
          v38 = (__int64 *)(v36 + 16LL * v37);
          v39 = *v38;
          if ( v35 == *v38 )
            goto LABEL_12;
          v44 = v48;
        }
      }
      v38 = (__int64 *)(v36 + 16 * v34);
LABEL_12:
      if ( !v16 )
      {
        ++v55;
        v54 = 0;
        goto LABEL_14;
      }
      v18 = *((_DWORD *)v38 + 2);
      v19 = (v16 - 1) & (37 * v18);
      v20 = (_DWORD *)(v17 + 4LL * v19);
      v21 = *v20;
      if ( v18 != *v20 )
      {
        v52 = 1;
        v40 = 0;
        while ( v21 != -1 )
        {
          if ( v21 != -2 || v40 )
            v20 = v40;
          v19 = (v16 - 1) & (v52 + v19);
          v21 = *(_DWORD *)(v17 + 4LL * v19);
          if ( v18 == v21 )
            goto LABEL_4;
          ++v52;
          v40 = v20;
          v20 = (_DWORD *)(v17 + 4LL * v19);
        }
        if ( !v40 )
          v40 = v20;
        ++v55;
        v41 = v57 + 1;
        v54 = v40;
        if ( 4 * ((int)v57 + 1) < 3 * v16 )
        {
          if ( v16 - (v41 + HIDWORD(v57)) <= v16 >> 3 )
          {
            v49 = a7;
LABEL_15:
            sub_A08C50((__int64)&v55, v16);
            sub_22B31A0((__int64)&v55, (int *)v38 + 2, &v54);
            v40 = v54;
            a7 = v49;
            v41 = v57 + 1;
          }
          LODWORD(v57) = v41;
          if ( *v40 != -1 )
            --HIDWORD(v57);
          *v40 = *((_DWORD *)v38 + 2);
          v22 = *(_DWORD *)(a10 + 48);
          v23 = *v15;
          v24 = *(_QWORD *)(a10 + 32);
          if ( !v22 )
            goto LABEL_19;
          goto LABEL_5;
        }
LABEL_14:
        v49 = a7;
        v16 *= 2;
        goto LABEL_15;
      }
LABEL_4:
      v22 = *(_DWORD *)(a10 + 48);
      v23 = *v15;
      v24 = *(_QWORD *)(a10 + 32);
      if ( !v22 )
        goto LABEL_19;
LABEL_5:
      v25 = (v22 - 1) & (((unsigned int)v23 >> 9) ^ ((unsigned int)v23 >> 4));
      v26 = (__int64 *)(v24 + 16LL * v25);
      v27 = *v26;
      if ( v23 == *v26 )
      {
LABEL_6:
        v28 = v62;
        v29 = (int *)(v26 + 1);
        if ( !(_DWORD)v62 )
          goto LABEL_20;
        goto LABEL_7;
      }
      v47 = 1;
      while ( v27 != -4096 )
      {
        v25 = (v22 - 1) & (v47 + v25);
        v26 = (__int64 *)(v24 + 16LL * v25);
        v27 = *v26;
        if ( v23 == *v26 )
          goto LABEL_6;
        ++v47;
      }
LABEL_19:
      v28 = v62;
      v26 = (__int64 *)(v24 + 16LL * v22);
      v29 = (int *)(v26 + 1);
      if ( !(_DWORD)v62 )
      {
LABEL_20:
        ++v59;
        v54 = 0;
LABEL_21:
        v50 = a7;
        v28 *= 2;
        goto LABEL_22;
      }
LABEL_7:
      v30 = *((_DWORD *)v26 + 2);
      v31 = (v28 - 1) & (37 * v30);
      v32 = (_DWORD *)(v60 + 4LL * v31);
      v33 = *v32;
      if ( v30 == *v32 )
        goto LABEL_8;
      v51 = 1;
      v42 = 0;
      while ( v33 != -1 )
      {
        if ( v42 || v33 != -2 )
          v32 = v42;
        v31 = (v28 - 1) & (v51 + v31);
        v33 = *(_DWORD *)(v60 + 4LL * v31);
        if ( v30 == v33 )
          goto LABEL_8;
        ++v51;
        v42 = v32;
        v32 = (_DWORD *)(v60 + 4LL * v31);
      }
      if ( !v42 )
        v42 = v32;
      ++v59;
      v43 = v61 + 1;
      v54 = v42;
      if ( 4 * ((int)v61 + 1) >= 3 * v28 )
        goto LABEL_21;
      if ( v28 - HIDWORD(v61) - v43 > v28 >> 3 )
        goto LABEL_39;
      v50 = a7;
LABEL_22:
      sub_A08C50((__int64)&v59, v28);
      sub_22B31A0((__int64)&v59, v29, &v54);
      v42 = v54;
      a7 = v50;
      v43 = v61 + 1;
LABEL_39:
      LODWORD(v61) = v43;
      if ( *v42 != -1 )
        --HIDWORD(v61);
      *v42 = *((_DWORD *)v26 + 2);
LABEL_8:
      ++v13;
      ++v15;
      if ( v13 == v53 )
        break;
      v17 = v56;
      v16 = v58;
    }
  }
  v45 = sub_22B65C0(a7 + 24, a9, (__int64 **)a8, (__int64)&v59);
  if ( (_BYTE)v45 )
    v45 = sub_22B65C0(a10 + 24, a12, a11, (__int64)&v55);
  sub_C7D6A0(v60, 4LL * (unsigned int)v62, 4);
  sub_C7D6A0(v56, 4LL * (unsigned int)v58, 4);
  return v45;
}
