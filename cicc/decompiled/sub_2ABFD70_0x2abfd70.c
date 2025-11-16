// Function: sub_2ABFD70
// Address: 0x2abfd70
//
__int64 __fastcall sub_2ABFD70(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 *v6; // r14
  int v7; // r13d
  __int64 *v8; // r12
  char v9; // bl
  __int64 v10; // r14
  __int64 v11; // r8
  __int64 j; // r9
  __int64 v13; // rdx
  __int64 *v14; // rbx
  _BYTE *v15; // r13
  __int64 v16; // rdi
  __int64 v17; // rsi
  _QWORD *v18; // rax
  __int64 v19; // rcx
  __int64 v20; // rcx
  __int64 v21; // rsi
  unsigned int i; // eax
  __int64 v23; // rdi
  int v24; // eax
  __int64 v25; // rax
  __int64 v26; // rax
  unsigned __int64 v27; // rdx
  __int64 v28; // r13
  __int64 *v29; // r13
  __int64 v30; // rdx
  __int64 *v31; // rbx
  _BYTE *v32; // r12
  __int64 v33; // rdi
  __int64 v34; // rsi
  _QWORD *v35; // rax
  __int64 v36; // rcx
  __int64 v37; // rcx
  __int64 v38; // rsi
  int v39; // r9d
  unsigned int k; // eax
  __int64 v41; // rdi
  unsigned int v42; // eax
  __int64 v44; // rax
  __int64 v45; // rsi
  int v46; // r10d
  int v47; // r11d
  unsigned int v48; // edi
  unsigned int m; // r9d
  unsigned int *v50; // rcx
  unsigned int v51; // r9d
  __int64 v52; // rax
  __int64 v53; // rsi
  int v54; // r10d
  int v55; // r11d
  unsigned int v56; // edi
  unsigned int *v57; // rcx
  unsigned int v58; // r9d
  int v59; // r8d
  unsigned int *v60; // rcx
  __int64 v61; // rdx
  unsigned int v62; // edi
  unsigned int *v63; // rcx
  __int64 v64; // rdx
  int v65; // edi
  const void *v66; // [rsp+8h] [rbp-88h]
  __int64 *v67; // [rsp+10h] [rbp-80h]
  int v68; // [rsp+18h] [rbp-78h]
  __int64 v69; // [rsp+28h] [rbp-68h]
  __int64 v70; // [rsp+28h] [rbp-68h]
  __int64 v73; // [rsp+38h] [rbp-58h]
  _QWORD *v74; // [rsp+38h] [rbp-58h]
  unsigned int v75; // [rsp+40h] [rbp-50h]
  unsigned __int8 v76; // [rsp+47h] [rbp-49h]
  unsigned __int64 v77; // [rsp+58h] [rbp-38h] BYREF

  v6 = (__int64 *)a4;
  v75 = a5;
  v76 = BYTE4(a5);
  v67 = (__int64 *)a3;
  if ( a3 == a4 )
  {
LABEL_30:
    *(_QWORD *)a1 = a1 + 16;
    *(_QWORD *)(a1 + 8) = 0x400000000LL;
    goto LABEL_31;
  }
  v7 = a5;
  v8 = (__int64 *)a3;
  v9 = BYTE4(a5);
  while ( 1 )
  {
    v10 = *v8;
    LODWORD(v77) = v7;
    BYTE4(v77) = v9;
    if ( *(_BYTE *)v10 > 0x1Cu
      && (BYTE4(v77) || v7 != 1)
      && (unsigned __int8)sub_B19060(*(_QWORD *)(a2 + 416) + 56LL, *(_QWORD *)(v10 + 40), a3, a4)
      && !(unsigned __int8)sub_D48480(*(_QWORD *)(a2 + 416), v10, a3, a4)
      && (unsigned int)sub_2AAA2B0(a2, v10, v7, v9) != 5
      && (!sub_2ABFD00(a2 + 192, (__int64)&v77) || !(unsigned __int8)sub_2AB2DA0(a2, v10, v77)) )
    {
      break;
    }
    v8 += 4;
    if ( (__int64 *)a4 == v8 )
    {
      v67 = v8;
      v6 = (__int64 *)a4;
      goto LABEL_30;
    }
  }
  v6 = (__int64 *)a4;
  v67 = v8;
  v66 = (const void *)(a1 + 16);
  *(_QWORD *)a1 = a1 + 16;
  *(_QWORD *)(a1 + 8) = 0x400000000LL;
  if ( (__int64 *)a4 == v8 )
  {
LABEL_31:
    v68 = 0;
    v25 = 0;
    goto LABEL_91;
  }
  v13 = v75;
  v73 = 0;
  do
  {
    v14 = v8 + 4;
    if ( v6 == v8 + 4 )
    {
LABEL_37:
      v26 = *(unsigned int *)(a1 + 8);
      v68 = v73 + 1;
      v27 = v26 + v73 + 1;
      if ( v27 > *(unsigned int *)(a1 + 12) )
        goto LABEL_90;
      v28 = *(_QWORD *)a1 + 8 * v26;
      goto LABEL_39;
    }
    while ( 1 )
    {
      v15 = (_BYTE *)*v14;
      v8 = v14;
      if ( *(_BYTE *)*v14 <= 0x1Cu || !v76 && v75 == 1 )
        goto LABEL_36;
      v16 = *(_QWORD *)(a2 + 416);
      v17 = *((_QWORD *)v15 + 5);
      if ( *(_BYTE *)(v16 + 84) )
        break;
      if ( sub_C8CA60(v16 + 56, v17) )
      {
        v16 = *(_QWORD *)(a2 + 416);
        goto LABEL_22;
      }
LABEL_36:
      v14 += 4;
      if ( v6 == v14 )
        goto LABEL_37;
    }
    v18 = *(_QWORD **)(v16 + 64);
    v19 = (__int64)&v18[*(unsigned int *)(v16 + 76)];
    if ( v18 == (_QWORD *)v19 )
      goto LABEL_36;
    while ( v17 != *v18 )
    {
      if ( (_QWORD *)v19 == ++v18 )
        goto LABEL_36;
    }
LABEL_22:
    if ( (unsigned __int8)sub_D48480(v16, (__int64)v15, v13, v19) )
      goto LABEL_36;
    v20 = *(unsigned int *)(a2 + 376);
    v21 = *(_QWORD *)(a2 + 360);
    if ( (_DWORD)v20 )
    {
      v11 = (unsigned int)(v20 - 1);
      j = 1;
      v69 = (v76 == 0) + 37 * v75 - 1;
      for ( i = v11
              & (((0xBF58476D1CE4E5B9LL
                 * (v69 | ((unsigned __int64)(((unsigned int)v15 >> 9) ^ ((unsigned int)v15 >> 4)) << 32))) >> 31)
               ^ (484763065 * v69)); ; i = v11 & v24 )
      {
        v23 = v21 + 40LL * i;
        v13 = *(_QWORD *)v23;
        if ( v15 == *(_BYTE **)v23 && *(_DWORD *)(v23 + 8) == v75 && *(_BYTE *)(v23 + 12) == v76 )
          break;
        if ( v13 == -4096 && *(_DWORD *)(v23 + 8) == -1 && *(_BYTE *)(v23 + 12) )
          goto LABEL_77;
        v24 = j + i;
        j = (unsigned int)(j + 1);
      }
      if ( v23 != v21 + 40 * v20 && *(_DWORD *)(v23 + 16) == 5 )
        goto LABEL_36;
    }
LABEL_77:
    v52 = *(unsigned int *)(a2 + 216);
    v53 = *(_QWORD *)(a2 + 200);
    if ( (_DWORD)v52 )
    {
      v54 = v52 - 1;
      v55 = 1;
      v56 = (v52 - 1) & ((v76 == 0) + 37 * v75 - 1);
      for ( j = v56; ; j = v54 & v58 )
      {
        v57 = (unsigned int *)(v53 + 72LL * (unsigned int)j);
        v13 = *v57;
        if ( (_DWORD)v13 == v75 )
        {
          v11 = v76;
          if ( *((_BYTE *)v57 + 4) == v76 )
            break;
        }
        if ( (_DWORD)v13 == -1 && *((_BYTE *)v57 + 4) )
          goto LABEL_88;
        v58 = v55 + j;
        ++v55;
      }
      if ( v76 )
      {
        LODWORD(v11) = 1;
        goto LABEL_102;
      }
      v11 = 1;
      if ( v75 != 1 )
      {
LABEL_102:
        while ( 1 )
        {
          v63 = (unsigned int *)(v53 + 72LL * v56);
          v64 = *v63;
          if ( (_DWORD)v64 == v75 && *((_BYTE *)v63 + 4) == v76 )
            break;
          if ( (_DWORD)v64 == -1 && *((_BYTE *)v63 + 4) )
          {
            if ( (unsigned __int8)sub_B19060(v53 + 72 * v52 + 8, (__int64)v15, v64, v53 + 72 * v52) )
              goto LABEL_36;
            goto LABEL_88;
          }
          v65 = v11 + v56;
          LODWORD(v11) = v11 + 1;
          v56 = v54 & v65;
        }
        if ( !(unsigned __int8)sub_B19060((__int64)(v63 + 2), (__int64)v15, v64, (__int64)v63) )
          goto LABEL_88;
      }
      goto LABEL_36;
    }
LABEL_88:
    ++v73;
  }
  while ( v6 != v14 );
  v25 = *(unsigned int *)(a1 + 8);
  v68 = v73;
  v27 = v25 + v73;
  if ( v25 + v73 > (unsigned __int64)*(unsigned int *)(a1 + 12) )
  {
LABEL_90:
    sub_C8D5F0(a1, v66, v27, 8u, v11, j);
    v25 = *(unsigned int *)(a1 + 8);
  }
LABEL_91:
  v28 = *(_QWORD *)a1 + 8 * v25;
  if ( v6 == v67 )
    goto LABEL_63;
LABEL_39:
  v74 = (_QWORD *)v28;
  v29 = v67;
  while ( 2 )
  {
    v30 = (__int64)v74;
    if ( v74 )
      *v74 = *v29;
    v31 = v29 + 4;
    if ( v6 != v29 + 4 )
    {
LABEL_43:
      v32 = (_BYTE *)*v31;
      v29 = v31;
      if ( *(_BYTE *)*v31 > 0x1Cu && (v76 || v75 != 1) )
      {
        v33 = *(_QWORD *)(a2 + 416);
        v34 = *((_QWORD *)v32 + 5);
        if ( *(_BYTE *)(v33 + 84) )
        {
          v35 = *(_QWORD **)(v33 + 64);
          v36 = (__int64)&v35[*(unsigned int *)(v33 + 76)];
          if ( v35 == (_QWORD *)v36 )
            goto LABEL_61;
          while ( v34 != *v35 )
          {
            if ( (_QWORD *)v36 == ++v35 )
              goto LABEL_61;
          }
        }
        else
        {
          if ( !sub_C8CA60(v33 + 56, v34) )
            goto LABEL_61;
          v33 = *(_QWORD *)(a2 + 416);
        }
        if ( !(unsigned __int8)sub_D48480(v33, (__int64)v32, v30, v36) )
        {
          v37 = *(unsigned int *)(a2 + 376);
          v38 = *(_QWORD *)(a2 + 360);
          if ( !(_DWORD)v37 )
            goto LABEL_70;
          v39 = 1;
          v70 = (v76 == 0) + 37 * v75 - 1;
          for ( k = (v37 - 1)
                  & (((0xBF58476D1CE4E5B9LL
                     * (v70 | ((unsigned __int64)(((unsigned int)v32 >> 9) ^ ((unsigned int)v32 >> 4)) << 32))) >> 31)
                   ^ (484763065 * v70)); ; k = (v37 - 1) & v42 )
          {
            v41 = v38 + 40LL * k;
            v30 = *(_QWORD *)v41;
            if ( v32 == *(_BYTE **)v41 && *(_DWORD *)(v41 + 8) == v75 && *(_BYTE *)(v41 + 12) == v76 )
              break;
            if ( v30 == -4096 && *(_DWORD *)(v41 + 8) == -1 && *(_BYTE *)(v41 + 12) )
              goto LABEL_70;
            v42 = v39 + k;
            ++v39;
          }
          if ( v41 == v38 + 40 * v37 || *(_DWORD *)(v41 + 16) != 5 )
          {
LABEL_70:
            v44 = *(unsigned int *)(a2 + 216);
            v45 = *(_QWORD *)(a2 + 200);
            if ( !(_DWORD)v44 )
            {
LABEL_84:
              ++v74;
              if ( v6 == v31 )
                break;
              continue;
            }
            v46 = v44 - 1;
            v47 = 1;
            v48 = (v44 - 1) & ((v76 == 0) + 37 * v75 - 1);
            for ( m = v48; ; m = v46 & v51 )
            {
              v50 = (unsigned int *)(v45 + 72LL * m);
              v30 = *v50;
              if ( (_DWORD)v30 == v75 && *((_BYTE *)v50 + 4) == v76 )
                break;
              if ( (_DWORD)v30 == -1 && *((_BYTE *)v50 + 4) )
                goto LABEL_84;
              v51 = v47 + m;
              ++v47;
            }
            if ( v76 )
            {
              v59 = 1;
              goto LABEL_96;
            }
            v59 = 1;
            if ( v75 != 1 )
            {
LABEL_96:
              while ( 1 )
              {
                v60 = (unsigned int *)(v45 + 72LL * v48);
                v61 = *v60;
                if ( (_DWORD)v61 == v75 && *((_BYTE *)v60 + 4) == v76 )
                  break;
                if ( (_DWORD)v61 == -1 && *((_BYTE *)v60 + 4) )
                {
                  if ( (unsigned __int8)sub_B19060(v45 + 72 * v44 + 8, (__int64)v32, v61, v45 + 72 * v44) )
                    goto LABEL_61;
                  goto LABEL_84;
                }
                v62 = v59 + v48;
                ++v59;
                v48 = v46 & v62;
              }
              if ( !(unsigned __int8)sub_B19060((__int64)(v60 + 2), (__int64)v32, v61, (__int64)v60) )
                goto LABEL_84;
            }
          }
        }
      }
LABEL_61:
      v31 += 4;
      if ( v6 == v31 )
        break;
      goto LABEL_43;
    }
    break;
  }
  LODWORD(v25) = *(_DWORD *)(a1 + 8);
LABEL_63:
  *(_DWORD *)(a1 + 8) = v68 + v25;
  return a1;
}
