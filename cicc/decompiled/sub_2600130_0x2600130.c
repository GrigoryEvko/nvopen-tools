// Function: sub_2600130
// Address: 0x2600130
//
unsigned __int64 __fastcall sub_2600130(_QWORD **a1, __int64 a2, __int64 *a3)
{
  _QWORD *v4; // rax
  __int64 v5; // rbx
  __int64 v6; // rsi
  __int64 v7; // rdi
  __int64 v8; // rcx
  __int64 v9; // r8
  __int64 v10; // rax
  __int64 v11; // r12
  __int64 v12; // r11
  __int64 v13; // r14
  __int64 *v14; // r11
  unsigned int v15; // r15d
  __int64 *v16; // rbx
  __int64 *v17; // r13
  __int64 *v18; // rdi
  __int64 v19; // r9
  __int64 v20; // rax
  int v21; // esi
  unsigned int v22; // edx
  __int64 *v23; // rdi
  __int64 *v24; // r10
  int v25; // edi
  int v26; // r10d
  unsigned __int64 v27; // r12
  unsigned int v28; // r14d
  unsigned int **v30; // rax
  __int64 v31; // r13
  __int64 v32; // rdx
  unsigned int *v33; // rbx
  unsigned int v34; // eax
  __int64 v35; // rax
  __int64 v36; // rax
  int v37; // edx
  int v38; // ebx
  signed __int64 v39; // r13
  signed __int64 v40; // rax
  int v41; // edx
  unsigned __int64 v42; // rcx
  signed __int64 v43; // rdx
  signed __int64 v44; // rax
  __int64 v45; // rcx
  signed __int64 v46; // rax
  bool v47; // of
  int v48; // r13d
  unsigned int **v49; // r14
  int v50; // r14d
  unsigned int *v51; // r13
  signed __int64 v52; // rax
  signed __int64 v53; // rax
  unsigned __int64 v54; // rax
  unsigned __int64 v55; // rax
  unsigned __int64 v56; // rax
  unsigned int v58; // [rsp+Ch] [rbp-B4h]
  _QWORD *v60; // [rsp+18h] [rbp-A8h]
  unsigned int **v61; // [rsp+20h] [rbp-A0h]
  unsigned int **v62; // [rsp+28h] [rbp-98h]
  int v63; // [rsp+30h] [rbp-90h]
  unsigned int *v64; // [rsp+30h] [rbp-90h]
  __int64 *v65; // [rsp+30h] [rbp-90h]
  __int64 v66; // [rsp+40h] [rbp-80h] BYREF
  __int64 *v67; // [rsp+48h] [rbp-78h] BYREF
  __int64 v68; // [rsp+50h] [rbp-70h] BYREF
  __int64 v69; // [rsp+58h] [rbp-68h]
  __int64 v70; // [rsp+60h] [rbp-60h]
  __int64 v71; // [rsp+68h] [rbp-58h]
  __int64 v72; // [rsp+70h] [rbp-50h] BYREF
  __int64 v73; // [rsp+78h] [rbp-48h]
  __int64 v74; // [rsp+80h] [rbp-40h]
  __int64 v75; // [rsp+88h] [rbp-38h]

  v4 = **(_QWORD ***)a2;
  v5 = *v4;
  v70 = 0;
  v71 = 0;
  v69 = 0;
  v6 = *(_QWORD *)(v5 + 16);
  v7 = *(_QWORD *)(v5 + 8);
  v60 = v4;
  v68 = 0;
  sub_25FFEC0(v7, v6, (__int64)&v68);
  v74 = 0;
  v75 = 0;
  v10 = *(_QWORD *)(v5 + 16);
  v73 = 0;
  v11 = *(_QWORD *)(v5 + 8);
  v12 = *(_QWORD *)(v10 + 8);
  v72 = 0;
  if ( v11 != v12 )
  {
    v13 = v12;
    v14 = a3;
    v15 = 0;
    while ( 1 )
    {
      if ( **(_BYTE **)(v11 + 16) == 31 )
      {
        v16 = *(__int64 **)(v11 + 24);
        v17 = &v16[*(unsigned int *)(v11 + 32)];
        if ( v17 != v16 )
          break;
      }
LABEL_3:
      v11 = *(_QWORD *)(v11 + 8);
      if ( v11 == v13 )
      {
        v58 = v15;
        a3 = v14;
        goto LABEL_27;
      }
    }
    while ( 1 )
    {
      v20 = *v16;
      v66 = *v16;
      if ( (_DWORD)v71 )
      {
        v9 = (unsigned int)(v71 - 1);
        LODWORD(v8) = v9 & (((unsigned int)v20 >> 9) ^ ((unsigned int)v20 >> 4));
        v18 = (__int64 *)(v69 + 8LL * (unsigned int)v8);
        v19 = *v18;
        if ( v20 == *v18 )
        {
LABEL_8:
          if ( v18 != (__int64 *)(v69 + 8LL * (unsigned int)v71) )
            goto LABEL_9;
        }
        else
        {
          v25 = 1;
          while ( v19 != -4096 )
          {
            v26 = v25 + 1;
            LODWORD(v8) = v9 & (v25 + v8);
            v18 = (__int64 *)(v69 + 8LL * (unsigned int)v8);
            v19 = *v18;
            if ( v20 == *v18 )
              goto LABEL_8;
            v25 = v26;
          }
        }
      }
      v21 = v75;
      if ( !(_DWORD)v75 )
      {
        ++v72;
        v67 = 0;
        goto LABEL_67;
      }
      v9 = v73;
      v22 = (v75 - 1) & (((unsigned int)v20 >> 9) ^ ((unsigned int)v20 >> 4));
      v23 = (__int64 *)(v73 + 8LL * v22);
      v8 = *v23;
      if ( v20 != *v23 )
      {
        v63 = 1;
        v24 = 0;
        while ( v8 != -4096 )
        {
          if ( v24 || v8 != -8192 )
            v23 = v24;
          v22 = (v75 - 1) & (v63 + v22);
          v8 = *(_QWORD *)(v73 + 8LL * v22);
          if ( v20 == v8 )
            goto LABEL_9;
          ++v63;
          v24 = v23;
          v23 = (__int64 *)(v73 + 8LL * v22);
        }
        if ( !v24 )
          v24 = v23;
        ++v72;
        LODWORD(v8) = v74 + 1;
        v67 = v24;
        if ( 4 * ((int)v74 + 1) < (unsigned int)(3 * v75) )
        {
          if ( (int)v75 - HIDWORD(v74) - (int)v8 > (unsigned int)v75 >> 3 )
          {
LABEL_19:
            LODWORD(v74) = v8;
            if ( *v24 != -4096 )
              --HIDWORD(v74);
            *v24 = v20;
            ++v15;
            goto LABEL_9;
          }
          v65 = v14;
LABEL_68:
          sub_CF28B0((__int64)&v72, v21);
          sub_D6B660((__int64)&v72, &v66, &v67);
          v20 = v66;
          v24 = v67;
          v14 = v65;
          LODWORD(v8) = v74 + 1;
          goto LABEL_19;
        }
LABEL_67:
        v65 = v14;
        v21 = 2 * v75;
        goto LABEL_68;
      }
LABEL_9:
      if ( v17 == ++v16 )
        goto LABEL_3;
    }
  }
  v58 = 0;
LABEL_27:
  v27 = 0;
  v28 = *(_DWORD *)(a2 + 152);
  *(_DWORD *)(a2 + 208) = v58;
  if ( !v28 )
    goto LABEL_28;
  v30 = *(unsigned int ***)(a2 + 144);
  v31 = 2LL * *(unsigned int *)(a2 + 160);
  v32 = (__int64)&v30[v31];
  v62 = &v30[v31];
  if ( v30 == &v30[v31] )
    goto LABEL_33;
  while ( 1 )
  {
    v33 = *v30;
    if ( *v30 != (unsigned int *)-1LL && v33 != (unsigned int *)-2LL )
      break;
    v30 += 2;
    if ( (unsigned int **)v32 == v30 )
      goto LABEL_33;
  }
  if ( v30 == v62 )
  {
LABEL_33:
    v34 = v28;
    v27 = 0;
    goto LABEL_34;
  }
  v27 = 0;
  v48 = 0;
  v49 = v30;
  do
  {
    v64 = &v33[(_QWORD)v49[1]];
    if ( v64 == v33 )
      goto LABEL_55;
    v61 = v49;
    v50 = v48;
    v51 = v33;
    do
    {
      sub_25F6560(v60, *v51, v32, v8, v9);
      v52 = sub_DFD4A0(a3);
      v8 = v52 * v58;
      if ( is_mul_ok(v52, v58) )
      {
        if ( (_DWORD)v32 == 1 )
          v50 = 1;
        v47 = __OFADD__(v8, v27);
        v27 += v8;
        if ( v47 )
        {
          v27 = 0x8000000000000000LL;
          if ( v8 > 0 )
            v27 = 0x7FFFFFFFFFFFFFFFLL;
        }
        goto LABEL_53;
      }
      if ( v58 && v52 > 0 )
      {
        v54 = 0x7FFFFFFFFFFFFFFFLL;
        if ( (_DWORD)v32 != 1 )
          goto LABEL_78;
        v47 = __OFADD__(0x7FFFFFFFFFFFFFFFLL, v27);
        v27 += 0x7FFFFFFFFFFFFFFFLL;
        v50 = 1;
        if ( v47 )
          v27 = 0x7FFFFFFFFFFFFFFFLL;
      }
      else
      {
        v54 = 0x8000000000000000LL;
        if ( (_DWORD)v32 != 1 )
        {
LABEL_78:
          v47 = __OFADD__(v54, v27);
          v27 += v54;
          if ( v47 )
            v27 = v54;
          goto LABEL_53;
        }
        v47 = __OFADD__(0x8000000000000000LL, v27);
        v27 += 0x8000000000000000LL;
        v50 = 1;
        if ( v47 )
          v27 = 0x8000000000000000LL;
      }
LABEL_53:
      ++v51;
    }
    while ( v64 != v51 );
    v48 = v50;
    v49 = v61;
LABEL_55:
    v53 = sub_DFD270((__int64)a3, 2, 2);
    v8 = v53 * v58;
    if ( is_mul_ok(v53, v58) )
    {
      if ( (_DWORD)v32 == 1 )
        v48 = 1;
      v47 = __OFADD__(v8, v27);
      v27 += v8;
      if ( v47 )
      {
        v27 = 0x8000000000000000LL;
        if ( v8 > 0 )
          v27 = 0x7FFFFFFFFFFFFFFFLL;
      }
    }
    else
    {
      if ( v58 && v53 > 0 )
      {
        v56 = 0x7FFFFFFFFFFFFFFFLL;
        if ( (_DWORD)v32 == 1 )
        {
          v47 = __OFADD__(0x7FFFFFFFFFFFFFFFLL, v27);
          v27 += 0x7FFFFFFFFFFFFFFFLL;
          v48 = 1;
          if ( v47 )
            v27 = 0x7FFFFFFFFFFFFFFFLL;
          goto LABEL_59;
        }
      }
      else
      {
        v56 = 0x8000000000000000LL;
        if ( (_DWORD)v32 == 1 )
        {
          v47 = __OFADD__(0x8000000000000000LL, v27);
          v27 += 0x8000000000000000LL;
          v48 = 1;
          if ( v47 )
            v27 = 0x8000000000000000LL;
          goto LABEL_59;
        }
      }
      v47 = __OFADD__(v56, v27);
      v27 += v56;
      if ( v47 )
        v27 = v56;
    }
LABEL_59:
    v49 += 2;
    if ( v49 == v62 )
      break;
    while ( 1 )
    {
      v33 = *v49;
      if ( *v49 != (unsigned int *)-1LL && v33 != (unsigned int *)-2LL )
        break;
      v49 += 2;
      if ( v62 == v49 )
        goto LABEL_63;
    }
  }
  while ( v49 != v62 );
LABEL_63:
  v34 = *(_DWORD *)(a2 + 152);
LABEL_34:
  if ( v34 <= 1 )
    goto LABEL_28;
  sub_BCB2D0(*a1);
  v35 = sub_BCB2D0(*a1);
  v36 = sub_DFD2D0(a3, 53, v35);
  v38 = v37;
  v39 = v36;
  v40 = sub_DFD270((__int64)a3, 2, 2);
  if ( v41 == 1 )
    v38 = 1;
  v42 = *(unsigned int *)(a2 + 152);
  v43 = v40 * v39;
  if ( !is_mul_ok(v40, v39) )
  {
    if ( v39 <= 0 )
    {
      if ( v40 < 0 && v39 < 0 )
      {
LABEL_94:
        v44 = 0x7FFFFFFFFFFFFFFFLL * v42;
        if ( ((0x7FFFFFFFFFFFFFFFLL * v42) & 0x8000000000000000LL) != 0LL || !is_mul_ok(v42, 0x7FFFFFFFFFFFFFFFuLL) )
          goto LABEL_96;
LABEL_39:
        v45 = v58;
        if ( is_mul_ok(v58, v44) )
        {
          v46 = v58 * v44;
          goto LABEL_41;
        }
        if ( v44 <= 0 )
        {
LABEL_107:
          v55 = 0x8000000000000000LL;
          if ( v38 == 1 )
          {
            v55 = 0x8000000000000000LL;
            v47 = __OFADD__(0x8000000000000000LL, v27);
            v27 += 0x8000000000000000LL;
            if ( !v47 )
              goto LABEL_28;
            goto LABEL_102;
          }
          goto LABEL_101;
        }
LABEL_99:
        if ( !v45 )
          goto LABEL_107;
        v55 = 0x7FFFFFFFFFFFFFFFLL;
        if ( v38 == 1 )
        {
          v47 = __OFADD__(0x7FFFFFFFFFFFFFFFLL, v27);
          v27 += 0x7FFFFFFFFFFFFFFFLL;
          if ( !v47 )
            goto LABEL_28;
          goto LABEL_102;
        }
LABEL_101:
        v47 = __OFADD__(v55, v27);
        v27 += v55;
        if ( !v47 )
          goto LABEL_28;
LABEL_102:
        v27 = v55;
        goto LABEL_28;
      }
    }
    else if ( v40 > 0 )
    {
      goto LABEL_94;
    }
    v44 = v42 << 63;
    if ( !is_mul_ok(v42, 0x8000000000000000LL) )
      goto LABEL_106;
    goto LABEL_39;
  }
  v44 = v42 * v43;
  if ( is_mul_ok(v42, v43) )
    goto LABEL_39;
  if ( v43 > 0 )
  {
LABEL_96:
    if ( !*(_DWORD *)(a2 + 152) )
      goto LABEL_106;
    v45 = v58;
    v46 = 0x7FFFFFFFFFFFFFFFLL * v58;
    if ( v46 >= 0 && is_mul_ok(v58, 0x7FFFFFFFFFFFFFFFuLL) )
      goto LABEL_41;
    goto LABEL_99;
  }
LABEL_106:
  v46 = (unsigned __int64)v58 << 63;
  if ( !is_mul_ok(v58, 0x8000000000000000LL) )
    goto LABEL_107;
LABEL_41:
  v47 = __OFADD__(v46, v27);
  v27 += v46;
  if ( v47 )
  {
    v27 = 0x7FFFFFFFFFFFFFFFLL;
    if ( v46 <= 0 )
      v27 = 0x8000000000000000LL;
  }
LABEL_28:
  sub_C7D6A0(v73, 8LL * (unsigned int)v75, 8);
  sub_C7D6A0(v69, 8LL * (unsigned int)v71, 8);
  return v27;
}
