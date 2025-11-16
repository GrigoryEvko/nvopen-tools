// Function: sub_1C9BBA0
// Address: 0x1c9bba0
//
__int64 __fastcall sub_1C9BBA0(
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
  __int64 v10; // rbx
  __int64 v11; // r15
  __int64 i; // rbx
  _BYTE *v14; // rsi
  unsigned __int64 *v15; // rbx
  double v16; // xmm4_8
  double v17; // xmm5_8
  __int64 v18; // r14
  __int64 v19; // r12
  __int64 v20; // r14
  __int64 v21; // rax
  __int64 v22; // rax
  unsigned __int64 v23; // rcx
  unsigned __int64 *v24; // rax
  unsigned __int64 *v25; // r9
  unsigned __int64 *v26; // rdx
  unsigned __int64 v27; // rdi
  unsigned __int64 v28; // rsi
  unsigned __int64 *v29; // rsi
  unsigned __int64 v30; // rdi
  unsigned __int64 v31; // rdx
  unsigned __int64 v32; // rdi
  unsigned __int64 v33; // rax
  unsigned __int64 *v34; // rsi
  unsigned __int64 v35; // rcx
  unsigned __int64 v36; // rdx
  _QWORD *v37; // r9
  __int64 v38; // rax
  __int64 v39; // rdx
  __int64 v40; // r12
  __int64 v41; // rsi
  __int64 v42; // rdx
  __int64 v43; // rbx
  __int64 v44; // r12
  _QWORD *v46; // [rsp+18h] [rbp-C8h]
  __int64 v48; // [rsp+20h] [rbp-C0h]
  __int64 v49; // [rsp+28h] [rbp-B8h]
  __int64 v50; // [rsp+30h] [rbp-B0h] BYREF
  unsigned __int64 v51; // [rsp+38h] [rbp-A8h] BYREF
  __int64 v52; // [rsp+40h] [rbp-A0h] BYREF
  _BYTE *v53; // [rsp+48h] [rbp-98h]
  _BYTE *v54; // [rsp+50h] [rbp-90h]
  unsigned __int64 *v55[4]; // [rsp+60h] [rbp-80h] BYREF
  char v56[8]; // [rsp+80h] [rbp-60h] BYREF
  int v57; // [rsp+88h] [rbp-58h] BYREF
  __int64 v58; // [rsp+90h] [rbp-50h]
  int *v59; // [rsp+98h] [rbp-48h]
  int *v60; // [rsp+A0h] [rbp-40h]
  __int64 v61; // [rsp+A8h] [rbp-38h]

  v10 = *(_QWORD *)(a2 + 80);
  v57 = 0;
  v58 = 0;
  v59 = &v57;
  v60 = &v57;
  v61 = 0;
  v52 = 0;
  v53 = 0;
  v54 = 0;
  if ( !v10 )
    BUG();
  v11 = *(_QWORD *)(v10 + 24);
  for ( i = v10 + 16; i != v11; v11 = *(_QWORD *)(v11 + 8) )
  {
    if ( !v11 )
      BUG();
    if ( *(_BYTE *)(v11 - 8) == 53 && (unsigned __int8)sub_1C95B10(v11 - 24) )
    {
      v55[0] = (unsigned __int64 *)(v11 - 24);
      v55[1] = (unsigned __int64 *)(v11 - 24);
      v55[2] = 0;
      sub_1C99A10((__int64)v56, (__int64 *)v55);
      sub_1C997D0(a1, v11 - 24, v11 - 24, 0, (__int64)v56);
      v55[0] = (unsigned __int64 *)(v11 - 24);
      v14 = v53;
      if ( v53 == v54 )
      {
        sub_12879C0((__int64)&v52, v53, v55);
      }
      else
      {
        if ( v53 )
        {
          *(_QWORD *)v53 = v11 - 24;
          v14 = v53;
        }
        v53 = v14 + 8;
      }
    }
  }
  v15 = (unsigned __int64 *)(a1 + 384);
  v46 = (_QWORD *)(a1 + 376);
  sub_1C96F50(*(_QWORD **)(a1 + 392));
  *(_QWORD *)(a1 + 392) = 0;
  *(_QWORD *)(a1 + 400) = a1 + 384;
  *(_QWORD *)(a1 + 408) = a1 + 384;
  *(_QWORD *)(a1 + 416) = 0;
  v18 = *(_QWORD *)(a2 + 80);
  v48 = a2 + 72;
  if ( v48 != v18 )
  {
    while ( 1 )
    {
      if ( !v18 )
        BUG();
      v19 = v18 + 16;
      if ( v18 + 16 != *(_QWORD *)(v18 + 24) )
        break;
LABEL_52:
      v18 = *(_QWORD *)(v18 + 8);
      if ( v48 == v18 )
        goto LABEL_53;
    }
    v49 = v18;
    v20 = *(_QWORD *)(v18 + 24);
    while ( 1 )
    {
      if ( !v20 )
        BUG();
      if ( *(_BYTE *)(v20 - 8) != 78 )
        goto LABEL_15;
      v21 = *(_QWORD *)(v20 - 48);
      if ( *(_BYTE *)(v21 + 16) )
        goto LABEL_15;
      if ( (*(_BYTE *)(v21 + 33) & 0x20) == 0 )
        goto LABEL_15;
      if ( *(_DWORD *)(v21 + 36) != 38 )
        goto LABEL_15;
      v50 = v20 - 24;
      v22 = sub_1601A30(v20 - 24, 0);
      v23 = v22;
      if ( !v22 || *(_BYTE *)(v22 + 16) != 54 )
        goto LABEL_15;
      v51 = v22;
      v24 = *(unsigned __int64 **)(a1 + 392);
      if ( !v24 )
        break;
      v25 = (unsigned __int64 *)(a1 + 384);
      v26 = *(unsigned __int64 **)(a1 + 392);
      do
      {
        while ( 1 )
        {
          v27 = v26[2];
          v28 = v26[3];
          if ( v26[4] >= v23 )
            break;
          v26 = (unsigned __int64 *)v26[3];
          if ( !v28 )
            goto LABEL_28;
        }
        v25 = v26;
        v26 = (unsigned __int64 *)v26[2];
      }
      while ( v27 );
LABEL_28:
      if ( v15 != v25 && v25[4] <= v23 )
      {
LABEL_40:
        v34 = (unsigned __int64 *)(a1 + 384);
        do
        {
          while ( 1 )
          {
            v35 = v24[2];
            v36 = v24[3];
            if ( v24[4] >= v51 )
              break;
            v24 = (unsigned __int64 *)v24[3];
            if ( !v36 )
              goto LABEL_44;
          }
          v34 = v24;
          v24 = (unsigned __int64 *)v24[2];
        }
        while ( v35 );
LABEL_44:
        if ( v15 != v34 && v34[4] <= v51 )
          goto LABEL_47;
        goto LABEL_46;
      }
      v29 = (unsigned __int64 *)(a1 + 384);
      do
      {
        while ( 1 )
        {
          v30 = v24[2];
          v31 = v24[3];
          if ( v24[4] >= v23 )
            break;
          v24 = (unsigned __int64 *)v24[3];
          if ( !v31 )
            goto LABEL_34;
        }
        v29 = v24;
        v24 = (unsigned __int64 *)v24[2];
      }
      while ( v30 );
LABEL_34:
      if ( v15 == v29 || v29[4] > v23 )
        goto LABEL_36;
LABEL_37:
      v32 = v29[5];
      v33 = v29[7];
      v29[5] = 0;
      v29[6] = 0;
      v29[7] = 0;
      if ( v32 )
        j_j___libc_free_0(v32, v33 - v32);
      v24 = *(unsigned __int64 **)(a1 + 392);
      if ( v24 )
        goto LABEL_40;
      v34 = (unsigned __int64 *)(a1 + 384);
LABEL_46:
      v55[0] = &v51;
      v34 = sub_1C9B790(v46, v34, v55);
LABEL_47:
      v37 = (_QWORD *)v34[6];
      if ( v37 == (_QWORD *)v34[7] )
      {
        sub_1C98CC0((__int64)(v34 + 5), (_BYTE *)v34[6], &v50);
LABEL_15:
        v20 = *(_QWORD *)(v20 + 8);
        if ( v19 == v20 )
          goto LABEL_51;
      }
      else
      {
        if ( v37 )
        {
          *v37 = v50;
          v37 = (_QWORD *)v34[6];
        }
        v34[6] = (unsigned __int64)(v37 + 1);
        v20 = *(_QWORD *)(v20 + 8);
        if ( v19 == v20 )
        {
LABEL_51:
          v18 = v49;
          goto LABEL_52;
        }
      }
    }
    v29 = (unsigned __int64 *)(a1 + 384);
LABEL_36:
    v55[0] = &v51;
    v29 = sub_1C9B790(v46, v29, v55);
    goto LABEL_37;
  }
LABEL_53:
  v38 = *(_QWORD *)(a1 + 424);
  v39 = *(_QWORD *)(a1 + 432);
  v40 = (__int64)v59;
  if ( v38 != v39 )
  {
    *(_QWORD *)(a1 + 432) = v38;
    if ( (int *)v40 == &v57 )
      goto LABEL_64;
    goto LABEL_55;
  }
  if ( v59 != &v57 )
  {
    do
    {
LABEL_55:
      v41 = *(_QWORD *)(v40 + 32);
      if ( *(_BYTE *)(*(_QWORD *)v41 + 8LL) == 15 && *(_BYTE *)(*(_QWORD *)(*(_QWORD *)v41 + 24LL) + 8LL) == 13 )
        sub_1C9BA90((_QWORD *)a1, v41, a3, a4, a5, a6, v16, v17, a9, a10);
      v40 = sub_220EEE0(v40);
    }
    while ( (int *)v40 != &v57 );
    v39 = *(_QWORD *)(a1 + 432);
    v38 = *(_QWORD *)(a1 + 424);
  }
  v42 = (v39 - v38) >> 3;
  if ( (_DWORD)v42 )
  {
    v43 = 0;
    v44 = 8LL * (unsigned int)(v42 - 1);
    while ( 1 )
    {
      sub_15F20C0(*(_QWORD **)(v38 + v43));
      if ( v44 == v43 )
        break;
      v38 = *(_QWORD *)(a1 + 424);
      v43 += 8;
    }
  }
LABEL_64:
  if ( v52 )
    j_j___libc_free_0(v52, &v54[-v52]);
  return sub_1C96D30(v58);
}
