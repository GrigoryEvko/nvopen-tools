// Function: sub_2E87480
// Address: 0x2e87480
//
void __fastcall sub_2E87480(__int64 a1, __int64 a2, __int64 *a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v8; // rdx
  __int64 v9; // rax
  int *v10; // rcx
  unsigned __int64 v11; // r13
  __int64 v12; // r9
  unsigned int v13; // eax
  __int64 v14; // rax
  size_t v15; // r11
  __int64 *v16; // r15
  __int64 v17; // r8
  __int64 v18; // rdx
  __int64 v19; // rax
  int *v20; // r12
  int v21; // eax
  __int64 v22; // rbx
  __int64 **v23; // r12
  __int64 v24; // rax
  int *v25; // rsi
  __int64 v26; // rax
  unsigned __int64 v27; // rbx
  _QWORD *v28; // rdi
  __int64 v29; // rcx
  int v30; // eax
  __int64 v31; // rax
  __int64 **v32; // rbx
  __int64 **v33; // rcx
  __int64 *v34; // r14
  __int64 *v35; // r13
  __int64 v36; // rax
  __int64 v37; // rsi
  __int64 v38; // rdi
  unsigned __int64 v39; // rdi
  unsigned __int64 v40; // rax
  unsigned __int64 v41; // rdi
  int v42; // eax
  __int64 v43; // r13
  const void *v44; // r12
  signed __int64 v45; // r14
  __int64 v46; // rbx
  __int64 v47; // rax
  unsigned __int64 v48; // rdx
  int v49; // edx
  unsigned __int64 v50; // rax
  unsigned __int64 v51; // rax
  unsigned __int64 v52; // rsi
  unsigned __int64 v53; // rdi
  unsigned __int64 v54; // rdi
  unsigned __int64 v55; // rax
  unsigned __int64 v56; // rax
  char v57; // al
  int v58; // r14d
  int v59; // eax
  int v60; // eax
  __int64 v61; // r12
  const void *v62; // r14
  size_t v63; // r8
  _BYTE *v64; // rdi
  __int64 v65; // r14
  __int64 v66; // rax
  unsigned __int64 v67; // r9
  unsigned __int64 v68; // r10
  unsigned __int64 v69; // rax
  unsigned __int64 v70; // r9
  unsigned __int64 v71; // r10
  unsigned __int64 v72; // rdi
  char v73; // r10
  unsigned __int64 v74; // rax
  unsigned __int64 v75; // rsi
  unsigned __int64 v76; // rsi
  __int64 v77; // [rsp+0h] [rbp-90h]
  __int64 **v78; // [rsp+8h] [rbp-88h]
  __int64 v79; // [rsp+10h] [rbp-80h]
  __int64 v82; // [rsp+30h] [rbp-60h]
  __int64 v83; // [rsp+30h] [rbp-60h]
  size_t v84; // [rsp+30h] [rbp-60h]
  size_t na; // [rsp+38h] [rbp-58h]
  size_t nb; // [rsp+38h] [rbp-58h]
  char n; // [rsp+38h] [rbp-58h]
  char nc; // [rsp+38h] [rbp-58h]
  size_t nd; // [rsp+38h] [rbp-58h]
  _QWORD *v90; // [rsp+40h] [rbp-50h] BYREF
  __int64 v91; // [rsp+48h] [rbp-48h]
  _BYTE v92[64]; // [rsp+50h] [rbp-40h] BYREF

  if ( !a4 )
    goto LABEL_4;
  v8 = *a3;
  if ( a4 == 1 )
  {
    sub_2E86DF0(a1, a2, v8);
    return;
  }
  v9 = *(_QWORD *)(v8 + 48);
  v10 = (int *)(v9 & 0xFFFFFFFFFFFFFFF8LL);
  if ( (v9 & 0xFFFFFFFFFFFFFFF8LL) == 0 )
  {
LABEL_4:
    sub_2E868D0(a1, a2);
    return;
  }
  if ( (v9 & 7) != 0 )
  {
    if ( (v9 & 7) != 3 || !*v10 )
      goto LABEL_4;
    v90 = v92;
    v91 = 0x200000000LL;
LABEL_100:
    v60 = v9 & 7;
    if ( v60 )
    {
      if ( v60 == 3 )
      {
        v61 = (__int64)&v10[2 * *v10 + 4];
LABEL_103:
        v62 = v10 + 4;
LABEL_104:
        v63 = v61 - (_QWORD)v62;
        v11 = (v61 - (__int64)v62) >> 3;
        goto LABEL_105;
      }
      v61 = 0;
    }
    else
    {
      *(_QWORD *)(v8 + 48) = v10;
      v65 = *a3;
      v61 = v8 + 56;
      v66 = *(_QWORD *)(*a3 + 48);
      v10 = (int *)(v66 & 0xFFFFFFFFFFFFFFF8LL);
      if ( (v66 & 0xFFFFFFFFFFFFFFF8LL) == 0 )
        goto LABEL_130;
      v60 = v66 & 7;
      if ( !v60 )
      {
        *(_QWORD *)(v65 + 48) = v10;
        v62 = (const void *)(v65 + 48);
        goto LABEL_104;
      }
    }
    if ( v60 == 3 )
      goto LABEL_103;
LABEL_130:
    v63 = v61;
    v62 = 0;
    v11 = v61 >> 3;
LABEL_105:
    if ( v63 <= 0x10 )
    {
      v64 = v92;
      LODWORD(v10) = 0;
    }
    else
    {
      nd = v63;
      sub_C8D5F0((__int64)&v90, v92, v11, 8u, v63, a6);
      LODWORD(v10) = v91;
      v63 = nd;
      v64 = &v90[(unsigned int)v91];
    }
    if ( v62 != (const void *)v61 )
    {
      memcpy(v64, v62, v63);
      LODWORD(v10) = v91;
    }
    goto LABEL_8;
  }
  *(_QWORD *)(v8 + 48) = v10;
  v8 = *a3;
  v9 = *(_QWORD *)(*a3 + 48);
  v90 = v92;
  v91 = 0x200000000LL;
  v10 = (int *)(v9 & 0xFFFFFFFFFFFFFFF8LL);
  if ( (v9 & 0xFFFFFFFFFFFFFFF8LL) != 0 )
    goto LABEL_100;
  LODWORD(v11) = 0;
LABEL_8:
  v12 = (__int64)&a3[a4];
  LODWORD(v91) = v11 + (_DWORD)v10;
  v13 = v11 + (_DWORD)v10;
  if ( (__int64 *)v12 == a3 + 1 )
    goto LABEL_47;
  v14 = (__int64)a3;
  v15 = (size_t)&a3[a4];
  v16 = a3 + 1;
  v17 = v14;
  while ( 1 )
  {
LABEL_10:
    v18 = *v16;
    v19 = *(_QWORD *)(*v16 + 48);
    v20 = (int *)(v19 & 0xFFFFFFFFFFFFFFF8LL);
    if ( (v19 & 0xFFFFFFFFFFFFFFF8LL) != 0 )
    {
      v21 = v19 & 7;
      if ( !v21 )
      {
        *(_QWORD *)(v18 + 48) = v20;
        v22 = *(_QWORD *)v17;
        v23 = (__int64 **)(v18 + 48);
        v24 = *(_QWORD *)(*(_QWORD *)v17 + 48LL);
        v25 = (int *)(v24 & 0xFFFFFFFFFFFFFFF8LL);
        if ( (v24 & 0xFFFFFFFFFFFFFFF8LL) == 0 )
          goto LABEL_13;
        v29 = 1;
        goto LABEL_19;
      }
      if ( v21 == 3 )
      {
        v22 = *(_QWORD *)v17;
        v29 = *v20;
        v23 = (__int64 **)(v20 + 4);
        v24 = *(_QWORD *)(*(_QWORD *)v17 + 48LL);
        v25 = (int *)(v24 & 0xFFFFFFFFFFFFFFF8LL);
        if ( (v24 & 0xFFFFFFFFFFFFFFF8LL) != 0 )
          goto LABEL_19;
LABEL_85:
        if ( !v29 )
          goto LABEL_45;
LABEL_13:
        v26 = *(_QWORD *)(v18 + 48);
        v27 = v26 & 0xFFFFFFFFFFFFFFF8LL;
        if ( (v26 & 0xFFFFFFFFFFFFFFF8LL) == 0 )
          goto LABEL_14;
LABEL_34:
        if ( (v26 & 7) != 0 )
        {
          if ( (v26 & 7) != 3 || !*(_DWORD *)v27 )
          {
LABEL_14:
            sub_2E868D0(a1, a2);
            v28 = v90;
            if ( v90 == (_QWORD *)v92 )
              return;
            goto LABEL_15;
          }
          v42 = v26 & 7;
          if ( !v42 )
          {
LABEL_38:
            *(_QWORD *)(v18 + 48) = v27;
            v43 = v18 + 56;
            v42 = v27 & 7;
            if ( (v27 & 7) == 0 )
            {
              v44 = (const void *)(v18 + 48);
              v45 = 8;
              v46 = 1;
LABEL_40:
              v47 = (unsigned int)v91;
              v48 = (unsigned int)v91 + v46;
              if ( v48 > HIDWORD(v91) )
              {
                v82 = v17;
                na = v15;
                sub_C8D5F0((__int64)&v90, v92, v48, 8u, v17, v12);
                v47 = (unsigned int)v91;
                v17 = v82;
                v15 = na;
              }
              if ( (const void *)v43 == v44 )
              {
                v49 = v47;
              }
              else
              {
                v83 = v17;
                nb = v15;
                memcpy(&v90[v47], v44, v45);
                v49 = v91;
                v15 = nb;
                v17 = v83;
              }
              LODWORD(v91) = v46 + v49;
              goto LABEL_45;
            }
            goto LABEL_93;
          }
        }
        else
        {
          *(_QWORD *)(v18 + 48) = v27;
          v42 = v27 & 7;
          if ( (v27 & 7) == 0 )
            goto LABEL_38;
        }
        if ( v42 == 3 )
        {
          v43 = v27 + 8LL * *(int *)v27 + 16;
LABEL_94:
          v44 = (const void *)(v27 + 16);
          v45 = v43 - (v27 + 16);
          v46 = v45 >> 3;
          goto LABEL_40;
        }
        v43 = 0;
LABEL_93:
        if ( v42 != 3 )
        {
          v45 = v43;
          v44 = 0;
          v46 = v43 >> 3;
          goto LABEL_40;
        }
        goto LABEL_94;
      }
    }
    v22 = *(_QWORD *)v17;
    v24 = *(_QWORD *)(*(_QWORD *)v17 + 48LL);
    v25 = (int *)(v24 & 0xFFFFFFFFFFFFFFF8LL);
    if ( (v24 & 0xFFFFFFFFFFFFFFF8LL) == 0 )
      goto LABEL_45;
    v29 = 0;
    v23 = 0;
LABEL_19:
    v30 = v24 & 7;
    if ( v30 )
    {
      if ( v30 != 3 )
        goto LABEL_85;
      v31 = *v25;
      v32 = (__int64 **)(v25 + 4);
    }
    else
    {
      *(_QWORD *)(v22 + 48) = v25;
      v31 = 1;
      v32 = (__int64 **)(v22 + 48);
    }
    if ( v29 != v31 )
      goto LABEL_13;
    v33 = &v32[v29];
    if ( v33 != v32 )
      break;
LABEL_45:
    if ( (__int64 *)v15 == ++v16 )
      goto LABEL_46;
  }
  do
  {
    v34 = *v32;
    v35 = *v23;
    v36 = **v32;
    v37 = **v23;
    if ( !v36 )
    {
      if ( !v37 )
        goto LABEL_58;
      v12 = 0;
      if ( (v37 & 4) != 0 )
      {
LABEL_30:
        v40 = 0;
        if ( (v37 & 4) == 0 )
          goto LABEL_58;
LABEL_31:
        v41 = v37 & 0xFFFFFFFFFFFFFFF8LL;
        goto LABEL_32;
      }
      goto LABEL_27;
    }
    v38 = (v36 >> 2) & 1;
    if ( ((v36 >> 2) & 1) == 0 )
    {
      v39 = v36 & 0xFFFFFFFFFFFFFFF8LL;
      v12 = v36 & 0xFFFFFFFFFFFFFFF8LL;
      if ( !v37 )
      {
        if ( v39 )
          goto LABEL_13;
        goto LABEL_58;
      }
      if ( (v37 & 4) != 0 )
      {
        if ( v39 )
          goto LABEL_13;
LABEL_29:
        if ( (v36 & 4) == 0 )
          goto LABEL_30;
        goto LABEL_55;
      }
LABEL_27:
      if ( v12 != (v37 & 0xFFFFFFFFFFFFFFF8LL) )
        goto LABEL_13;
      if ( !v36 )
        goto LABEL_30;
      goto LABEL_29;
    }
    if ( !v37 )
    {
      if ( !(_BYTE)v38 )
        goto LABEL_58;
      v40 = v36 & 0xFFFFFFFFFFFFFFF8LL;
      v41 = 0;
      goto LABEL_32;
    }
    v12 = 0;
    if ( (v37 & 4) == 0 )
      goto LABEL_27;
    if ( !(_BYTE)v38 )
      goto LABEL_30;
LABEL_55:
    v40 = v36 & 0xFFFFFFFFFFFFFFF8LL;
    v41 = 0;
    if ( (v37 & 4) != 0 )
      goto LABEL_31;
LABEL_32:
    if ( v41 != v40 )
    {
      v26 = *(_QWORD *)(v18 + 48);
      v27 = v26 & 0xFFFFFFFFFFFFFFF8LL;
      if ( (v26 & 0xFFFFFFFFFFFFFFF8LL) == 0 )
        goto LABEL_14;
      goto LABEL_34;
    }
LABEL_58:
    v50 = v35[3];
    if ( (v50 & 0xFFFFFFFFFFFFFFF9LL) != 0 )
    {
      v72 = v50 >> 3;
      v12 = v35[3] & 6;
      v73 = v35[3] & 2;
      if ( (_BYTE)v12 == 2 || (v35[3] & 1) != 0 )
      {
        v76 = HIWORD(v50);
        if ( !v73 )
          v76 = HIDWORD(v50);
        v52 = (v76 + 7) >> 3;
      }
      else
      {
        v12 = (unsigned __int16)((unsigned int)v50 >> 8);
        v74 = HIDWORD(v50);
        v75 = HIWORD(v35[3]);
        if ( !v73 )
          LODWORD(v75) = v74;
        v52 = ((unsigned __int64)(unsigned int)(v12 * v75) + 7) >> 3;
        if ( (v72 & 1) != 0 )
          v52 |= 0x4000000000000000uLL;
      }
      v51 = v34[3];
      if ( (v51 & 0xFFFFFFFFFFFFFFF9LL) == 0 )
        goto LABEL_13;
    }
    else
    {
      v51 = v34[3];
      if ( (v51 & 0xFFFFFFFFFFFFFFF9LL) == 0 )
        goto LABEL_67;
      v52 = -1;
    }
    v12 = *((unsigned __int8 *)v34 + 24);
    v53 = v51 >> 3;
    n = v34[3] & 2;
    if ( (v12 & 6) == 2 || (v12 &= 1u, (_DWORD)v12) )
    {
      v54 = HIDWORD(v51);
      v55 = HIWORD(v51);
      if ( !n )
        v55 = v54;
      v56 = (v55 + 7) >> 3;
    }
    else
    {
      v67 = v51;
      v68 = v51;
      v69 = HIWORD(v51);
      v70 = v67 >> 8;
      v71 = HIDWORD(v68);
      if ( !n )
        LODWORD(v69) = v71;
      v12 = (unsigned int)v69 * (unsigned __int16)v70;
      v56 = (unsigned __int64)(v12 + 7) >> 3;
      if ( (v53 & 1) != 0 )
        v56 |= 0x4000000000000000uLL;
    }
    if ( v56 != v52 )
      goto LABEL_13;
LABEL_67:
    if ( v34[1] != v35[1] )
      goto LABEL_13;
    if ( *((_WORD *)v34 + 16) != *((_WORD *)v35 + 16) )
      goto LABEL_13;
    if ( v35[5] != v34[5] )
      goto LABEL_13;
    if ( v35[6] != v34[6] )
      goto LABEL_13;
    if ( v34[7] != v35[7] )
      goto LABEL_13;
    if ( v35[8] != v34[8] )
      goto LABEL_13;
    if ( v34[9] != v35[9] )
      goto LABEL_13;
    v77 = v17;
    v78 = v33;
    v79 = v18;
    v84 = v15;
    nc = sub_2EAC4F0(v35);
    v57 = sub_2EAC4F0(v34);
    v15 = v84;
    v18 = v79;
    v17 = v77;
    if ( v57 != nc )
      goto LABEL_13;
    v58 = sub_2EAC1E0(v34);
    v59 = sub_2EAC1E0(v35);
    v15 = v84;
    v18 = v79;
    v33 = v78;
    v17 = v77;
    if ( v58 != v59 )
      goto LABEL_13;
    ++v32;
    ++v23;
  }
  while ( v78 != v32 );
  if ( (__int64 *)v84 != ++v16 )
    goto LABEL_10;
LABEL_46:
  v13 = v91;
LABEL_47:
  sub_2E86A90(a1, a2, v90, v13);
  v28 = v90;
  if ( v90 != (_QWORD *)v92 )
LABEL_15:
    _libc_free((unsigned __int64)v28);
}
