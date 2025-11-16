// Function: sub_2F3A550
// Address: 0x2f3a550
//
void __fastcall sub_2F3A550(__int64 a1)
{
  __int64 v2; // r13
  __int64 v3; // rdi
  __int64 *v4; // r14
  __int64 *i; // r12
  __int64 v6; // rdi
  char *v7; // rsi
  __int64 v8; // rdx
  __int64 v9; // rax
  unsigned __int64 v10; // rax
  unsigned __int64 v11; // rcx
  void (*v12)(void); // rax
  __int64 v13; // r12
  __int64 v14; // r13
  __int64 v15; // rax
  unsigned __int64 v16; // rdx
  const void *v17; // r15
  __int64 v18; // r12
  __int64 v19; // rax
  char *v20; // r13
  signed __int64 v21; // rdx
  __int64 v22; // rdx
  __int64 v23; // rax
  int v24; // r13d
  unsigned int v25; // r12d
  unsigned int v26; // eax
  __int64 v27; // r15
  __int64 v28; // rax
  __int64 v29; // rsi
  __int64 v30; // rax
  __int64 v31; // r12
  __int64 (*v32)(); // rax
  _BYTE *v33; // rsi
  __int64 v34; // rax
  __int64 v35; // rdi
  __int64 v36; // r15
  __int64 (*v37)(); // rcx
  int v38; // eax
  __int64 *v39; // r13
  unsigned __int64 v40; // rax
  __int64 v41; // rdi
  __int64 (*v42)(); // rax
  _BYTE *v43; // rsi
  __int64 v44; // rdi
  __int64 *v45; // rdi
  __int64 v46; // rax
  void (*v47)(); // rdx
  __int64 (*v48)(); // rax
  __int64 v49; // r15
  __int64 v50; // rsi
  __int64 *v51; // r15
  __int64 v52; // rsi
  void (*v53)(void); // rax
  int v54; // r13d
  int j; // r15d
  char *v56; // rax
  __int64 v57; // [rsp+0h] [rbp-80h]
  char v58; // [rsp+Bh] [rbp-75h]
  unsigned int v59; // [rsp+Ch] [rbp-74h]
  __int64 v60; // [rsp+10h] [rbp-70h]
  __int64 v61; // [rsp+18h] [rbp-68h]
  unsigned int v62; // [rsp+18h] [rbp-68h]
  char v63; // [rsp+18h] [rbp-68h]
  __int64 v64; // [rsp+28h] [rbp-58h] BYREF
  unsigned __int64 v65; // [rsp+30h] [rbp-50h] BYREF
  unsigned __int64 v66; // [rsp+38h] [rbp-48h]
  __int64 v67; // [rsp+40h] [rbp-40h]

  v2 = a1 + 48;
  sub_2F97F60(a1, *(_QWORD *)(a1 + 3576), 0, 0, 0, 0);
  v3 = *(_QWORD *)(a1 + 3568);
  if ( v3
    && (*(unsigned int (__fastcall **)(__int64, __int64, _QWORD, _QWORD, _QWORD, __int64))(*(_QWORD *)v3 + 24LL))(
         v3,
         v2,
         *(_QWORD *)(a1 + 912),
         *(_QWORD *)(a1 + 920),
         *(unsigned int *)(a1 + 3632),
         a1 + 3344) )
  {
    sub_2F8EBD0(a1);
    sub_2F97F60(a1, *(_QWORD *)(a1 + 3576), 0, 0, 0, 0);
  }
  v4 = *(__int64 **)(a1 + 3616);
  for ( i = *(__int64 **)(a1 + 3608); v4 != i; ++i )
  {
    v6 = *i;
    (*(void (__fastcall **)(__int64, __int64))(*(_QWORD *)v6 + 24LL))(v6, a1);
  }
  v7 = *(char **)(a1 + 3488);
  v8 = *(_QWORD *)(a1 + 3480);
  *(_QWORD *)(a1 + 3472) = v2;
  v9 = *(_QWORD *)(a1 + 56) - *(_QWORD *)(a1 + 48);
  LODWORD(v65) = 0;
  v10 = v9 >> 8;
  v11 = (__int64)&v7[-v8] >> 2;
  if ( v10 > v11 )
  {
    sub_1CFD340(a1 + 3480, v7, v10 - v11, &v65);
  }
  else if ( v10 < v11 )
  {
    v56 = (char *)(v8 + 4 * v10);
    if ( v7 != v56 )
      *(_QWORD *)(a1 + 3488) = v56;
  }
  v12 = *(void (**)(void))(**(_QWORD **)(a1 + 3560) + 32LL);
  if ( v12 != nullsub_1618 )
    v12();
  sub_2F3A250(a1, a1 + 72);
  v13 = *(_QWORD *)(a1 + 48);
  v14 = *(_QWORD *)(a1 + 56);
  if ( v13 != v14 )
  {
    do
    {
      while ( *(_DWORD *)(v13 + 216) || (*(_BYTE *)(v13 + 249) & 2) != 0 )
      {
        v13 += 256;
        if ( v14 == v13 )
          goto LABEL_16;
      }
      sub_35033F0(a1 + 3456, v13);
      *(_BYTE *)(v13 + 249) |= 2u;
      v13 += 256;
    }
    while ( v14 != v13 );
LABEL_16:
    v15 = *(_QWORD *)(a1 + 56) - *(_QWORD *)(a1 + 48);
    v65 = 0;
    v66 = 0;
    v67 = 0;
    v16 = v15 >> 8;
    v57 = a1 + 3584;
    if ( v15 < 0 )
      sub_4262D8((__int64)"vector::reserve");
    v17 = *(const void **)(a1 + 3584);
    if ( v16 <= (__int64)(*(_QWORD *)(a1 + 3600) - (_QWORD)v17) >> 3 )
      goto LABEL_22;
    v61 = 8 * v16;
    v18 = *(_QWORD *)(a1 + 3592) - (_QWORD)v17;
    if ( v16 )
    {
      v19 = sub_22077B0(8 * v16);
      v17 = *(const void **)(a1 + 3584);
      v20 = (char *)v19;
      v21 = *(_QWORD *)(a1 + 3592) - (_QWORD)v17;
      if ( v21 <= 0 )
        goto LABEL_20;
    }
    else
    {
      v21 = *(_QWORD *)(a1 + 3592) - (_QWORD)v17;
      v20 = 0;
      if ( v18 <= 0 )
      {
LABEL_20:
        if ( !v17 )
        {
LABEL_21:
          *(_QWORD *)(a1 + 3584) = v20;
          *(_QWORD *)(a1 + 3592) = &v20[v18];
          *(_QWORD *)(a1 + 3600) = &v20[v61];
          goto LABEL_22;
        }
LABEL_103:
        j_j___libc_free_0((unsigned __int64)v17);
        goto LABEL_21;
      }
    }
    memmove(v20, v17, v21);
    goto LABEL_103;
  }
  v65 = 0;
  v66 = 0;
  v67 = 0;
  v57 = a1 + 3584;
LABEL_22:
  v58 = 0;
  v59 = 0;
  while ( 1 )
  {
    v22 = *(_QWORD *)(a1 + 3544);
    v23 = *(_QWORD *)(a1 + 3536);
    if ( *(_QWORD *)(a1 + 3512) == *(_QWORD *)(a1 + 3504) )
      break;
    v24 = (v22 - v23) >> 3;
    if ( !v24 )
      goto LABEL_36;
LABEL_25:
    v62 = -1;
    v25 = 0;
    while ( 1 )
    {
      v27 = v25;
      v28 = *(_QWORD *)(v23 + 8LL * v25);
      v29 = v28;
      if ( (*(_BYTE *)(v28 + 254) & 1) == 0 )
        break;
      if ( *(_DWORD *)(v28 + 240) <= v59 )
        goto LABEL_34;
LABEL_27:
      ++v25;
      if ( (*(_BYTE *)(v29 + 254) & 1) != 0 )
      {
        v26 = *(_DWORD *)(v29 + 240);
        if ( v26 >= v62 )
          goto LABEL_30;
        goto LABEL_29;
      }
      sub_2F8F5D0(v29);
      if ( *(_DWORD *)(v29 + 240) < v62 )
      {
        v49 = *(_QWORD *)(*(_QWORD *)(a1 + 3536) + 8 * v27);
        if ( (*(_BYTE *)(v49 + 254) & 1) == 0 )
          sub_2F8F5D0(v49);
        v26 = *(_DWORD *)(v49 + 240);
LABEL_29:
        v62 = v26;
      }
LABEL_30:
      if ( v24 == v25 )
        goto LABEL_35;
LABEL_31:
      v23 = *(_QWORD *)(a1 + 3536);
    }
    v60 = v28;
    sub_2F8F5D0(v28);
    v29 = *(_QWORD *)(*(_QWORD *)(a1 + 3536) + 8LL * v25);
    if ( *(_DWORD *)(v60 + 240) > v59 )
      goto LABEL_27;
LABEL_34:
    --v24;
    sub_35033F0(a1 + 3456, v29);
    v30 = *(_QWORD *)(*(_QWORD *)(a1 + 3536) + 8LL * v25);
    *(_BYTE *)(v30 + 249) |= 2u;
    *(_QWORD *)(*(_QWORD *)(a1 + 3536) + 8LL * v25) = *(_QWORD *)(*(_QWORD *)(a1 + 3544) - 8LL);
    *(_QWORD *)(a1 + 3544) -= 8LL;
    if ( v24 != v25 )
      goto LABEL_31;
LABEL_35:
    if ( *(_QWORD *)(a1 + 3512) != *(_QWORD *)(a1 + 3504) )
    {
LABEL_36:
      v63 = 0;
      v31 = 0;
      while ( 1 )
      {
        v34 = sub_3503120(a1 + 3456);
        v35 = *(_QWORD *)(a1 + 3560);
        v64 = v34;
        v36 = v34;
        v37 = *(__int64 (**)())(*(_QWORD *)v35 + 24LL);
        if ( v37 == sub_2EC0B50 )
        {
          v32 = *(__int64 (**)())(*(_QWORD *)v35 + 72LL);
          if ( v32 == sub_2F39240 )
            goto LABEL_66;
        }
        else
        {
          v38 = ((__int64 (__fastcall *)(__int64, __int64, _QWORD))v37)(v35, v34, 0);
          if ( v38 )
          {
            v33 = (_BYTE *)v66;
            v63 |= v38 == 2;
            if ( v66 == v67 )
              goto LABEL_48;
LABEL_41:
            if ( v33 )
            {
              *(_QWORD *)v33 = v64;
              v33 = (_BYTE *)v66;
            }
            v66 = (unsigned __int64)(v33 + 8);
            goto LABEL_44;
          }
          v35 = *(_QWORD *)(a1 + 3560);
          v36 = v64;
          v32 = *(__int64 (**)())(*(_QWORD *)v35 + 72LL);
          if ( v32 == sub_2F39240 )
          {
LABEL_66:
            if ( v31 )
            {
LABEL_67:
              if ( v36 )
              {
                v50 = v31;
                v31 = v36;
                sub_35033F0(a1 + 3456, v50);
              }
LABEL_50:
              v39 = (__int64 *)v66;
              v40 = v65;
              if ( v66 != v65 )
                goto LABEL_74;
              goto LABEL_51;
            }
            goto LABEL_93;
          }
        }
        if ( !((unsigned __int8 (__fastcall *)(__int64, __int64))v32)(v35, v36) )
        {
          v36 = v64;
          if ( v31 )
            goto LABEL_67;
LABEL_93:
          v39 = (__int64 *)v66;
          v40 = v65;
          v31 = v36;
          if ( v66 != v65 )
            goto LABEL_74;
          goto LABEL_78;
        }
        if ( v31 )
        {
          v33 = (_BYTE *)v66;
          if ( v66 != v67 )
            goto LABEL_41;
LABEL_48:
          sub_2ECAD30((__int64)&v65, v33, &v64);
          if ( *(_QWORD *)(a1 + 3512) == *(_QWORD *)(a1 + 3504) )
          {
LABEL_49:
            if ( v31 )
              goto LABEL_50;
            v39 = (__int64 *)v66;
            v40 = v65;
            if ( v65 != v66 )
              goto LABEL_74;
            goto LABEL_79;
          }
        }
        else
        {
          v31 = v64;
LABEL_44:
          if ( *(_QWORD *)(a1 + 3512) == *(_QWORD *)(a1 + 3504) )
            goto LABEL_49;
        }
      }
    }
LABEL_72:
    v39 = (__int64 *)v66;
    v40 = v65;
    if ( v65 == v66 )
    {
LABEL_81:
      v53 = *(void (**)(void))(**(_QWORD **)(a1 + 3560) + 80LL);
      if ( v53 != nullsub_1620 )
        v53();
      goto LABEL_83;
    }
    v63 = 0;
    v31 = 0;
LABEL_74:
    v51 = (__int64 *)v40;
    do
    {
      v52 = *v51++;
      sub_35033F0(a1 + 3456, v52);
    }
    while ( v39 != v51 );
    if ( v65 != v66 )
      v66 = v65;
LABEL_78:
    if ( v31 )
    {
LABEL_51:
      v41 = *(_QWORD *)(a1 + 3560);
      v42 = *(__int64 (**)())(*(_QWORD *)v41 + 56LL);
      if ( v42 != sub_2F39230 )
      {
        v54 = ((__int64 (__fastcall *)(__int64, __int64))v42)(v41, v31);
        if ( v54 )
        {
          for ( j = 0; j != v54; ++j )
            sub_2F3A4B0((_QWORD *)a1);
        }
      }
      v64 = v31;
      v43 = *(_BYTE **)(a1 + 3592);
      if ( v43 == *(_BYTE **)(a1 + 3600) )
      {
        sub_2ECAD30(v57, v43, &v64);
        v44 = v64;
      }
      else
      {
        if ( v43 )
        {
          *(_QWORD *)v43 = v31;
          v43 = *(_BYTE **)(a1 + 3592);
        }
        v44 = v31;
        *(_QWORD *)(a1 + 3592) = v43 + 8;
      }
      sub_2F8F720(v44, v59);
      sub_2F3A250(a1, v64);
      *(_BYTE *)(v64 + 249) |= 4u;
      sub_35032B0(a1 + 3456);
      v45 = *(__int64 **)(a1 + 3560);
      v46 = *v45;
      v47 = *(void (**)())(*v45 + 40);
      if ( v47 != nullsub_1619 )
      {
        ((void (__fastcall *)(__int64 *, __int64))v47)(v45, v31);
        v46 = **(_QWORD **)(a1 + 3560);
      }
      v48 = *(__int64 (**)())(v46 + 16);
      v58 = 1;
      if ( v48 != sub_2F39220 && (unsigned __int8)v48() )
        goto LABEL_81;
    }
    else
    {
LABEL_79:
      if ( v58 || !v63 )
        goto LABEL_81;
      sub_2F3A4B0((_QWORD *)a1);
LABEL_83:
      ++v59;
      v58 = 0;
    }
  }
  if ( v22 != v23 )
  {
    v24 = (v22 - v23) >> 3;
    if ( !v24 )
      goto LABEL_72;
    goto LABEL_25;
  }
  if ( v65 )
    j_j___libc_free_0(v65);
  *(_QWORD *)(a1 + 3472) = 0;
}
