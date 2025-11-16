// Function: sub_29197E0
// Address: 0x29197e0
//
__int64 __fastcall sub_29197E0(
        __int64 a1,
        __int64 *a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int64 a7,
        __int64 a8,
        _BYTE *a9,
        char *a10,
        _BYTE *a11,
        __int64 *a12)
{
  __int64 *v12; // r14
  __int64 v13; // rdx
  unsigned int v14; // ebx
  __int64 *v15; // r14
  __int64 v16; // r12
  __int64 v17; // rax
  __int64 v18; // r12
  __int64 v19; // rax
  __int64 v20; // r12
  _QWORD *v21; // rcx
  __int64 v22; // rax
  __int64 *v23; // rdx
  char v24; // dl
  __int64 v25; // rax
  __int64 v26; // rbx
  __int64 v27; // rax
  char *v28; // r14
  __int64 v29; // rax
  char *v30; // r12
  __int64 v31; // rax
  char *v32; // rsi
  __int64 v33; // rax
  char *v34; // rax
  char *v35; // rax
  __int64 *v36; // rbx
  char *v37; // rax
  size_t v38; // r14
  __int64 v39; // r12
  __int64 *v40; // r12
  __int64 v41; // r14
  char *v43; // r12
  __int64 v44; // rbx
  unsigned int v45; // eax
  __int64 *v46; // rdi
  int v47; // edx
  __int64 v48; // rbx
  unsigned __int64 v49; // rax
  char *v50; // rbx
  __int64 v51; // rcx
  __int64 v52; // rdx
  char *v53; // rax
  char *v54; // rsi
  __int64 v55; // rcx
  char *v56; // rbx
  int v57; // edx
  char *v58; // rax
  __int64 v59; // rbx
  char *v60; // rsi
  size_t v61; // rdx
  __int64 v62; // rax
  __int64 v63; // [rsp+0h] [rbp-A0h]
  __int64 *v65; // [rsp+18h] [rbp-88h]
  __int64 *v66; // [rsp+20h] [rbp-80h]
  __int64 *v67; // [rsp+28h] [rbp-78h]
  unsigned int v68; // [rsp+30h] [rbp-70h]
  __int64 v69; // [rsp+30h] [rbp-70h]
  __int64 v70; // [rsp+30h] [rbp-70h]
  __int64 *v71; // [rsp+38h] [rbp-68h]
  __int64 v72; // [rsp+48h] [rbp-58h]
  __int64 v73; // [rsp+60h] [rbp-40h]

  v12 = *(__int64 **)(a1 + 32);
  v67 = &a2[a3];
  v63 = a6;
  v65 = &v12[*(unsigned int *)(a1 + 40)];
  v71 = v12;
  if ( v12 != v65 )
  {
    while ( 1 )
    {
      v66 = (__int64 *)*v71;
      if ( (unsigned __int8)sub_BCBCB0(*v71) )
      {
        v13 = sub_9208B0(a7, (__int64)v66);
        HIDWORD(v73) = HIDWORD(v13);
        v14 = v13;
        if ( a2 != v67 )
          break;
      }
LABEL_3:
      if ( v65 == ++v71 )
        goto LABEL_25;
    }
    v15 = a2;
    while ( 1 )
    {
      v16 = *v15;
      v68 = sub_9208B0(a7, *v15);
      v17 = sub_9208B0(a7, *(_QWORD *)(v16 + 24));
      HIDWORD(v73) = HIDWORD(v17);
      if ( v14 != v68 && v14 != (_DWORD)v17 && !(v68 % v14) )
      {
        LODWORD(v72) = v68 / v14;
        BYTE4(v72) = 0;
        a6 = sub_BCE1B0(v66, v72);
        if ( (unsigned int)*(unsigned __int8 *)(a6 + 8) - 17 <= 1 )
        {
          v18 = *(_QWORD *)a5;
          v19 = *(unsigned int *)(*(_QWORD *)a5 + 8LL);
          if ( !(_DWORD)v19 )
            goto LABEL_16;
          v69 = a6;
          v20 = **(_QWORD **)v18;
          v73 = sub_9208B0(*(_QWORD *)(a5 + 8), a6);
          if ( v73 == sub_9208B0(*(_QWORD *)(a5 + 8), v20) )
          {
            a6 = v69;
            v18 = *(_QWORD *)a5;
            v19 = *(unsigned int *)(*(_QWORD *)a5 + 8LL);
LABEL_16:
            if ( v19 + 1 > (unsigned __int64)*(unsigned int *)(v18 + 12) )
            {
              v70 = a6;
              sub_C8D5F0(v18, (const void *)(v18 + 16), v19 + 1, 8u, a5, a6);
              v19 = *(unsigned int *)(v18 + 8);
              a6 = v70;
            }
            *(_QWORD *)(*(_QWORD *)v18 + 8 * v19) = a6;
            ++*(_DWORD *)(v18 + 8);
            v21 = *(_QWORD **)(a5 + 16);
            v22 = *(_QWORD *)(a6 + 24);
            if ( *v21 )
            {
              if ( v22 != *v21 )
                **(_BYTE **)(a5 + 24) = 0;
            }
            else
            {
              *v21 = v22;
            }
            if ( *(_BYTE *)(v22 + 8) == 14 )
            {
              **(_BYTE **)(a5 + 32) = 1;
              v23 = *(__int64 **)(a5 + 40);
              if ( *v23 )
              {
                if ( a6 != *v23 )
                  **(_BYTE **)(a5 + 48) = 0;
              }
              else
              {
                *v23 = a6;
              }
            }
            goto LABEL_7;
          }
          *(_DWORD *)(*(_QWORD *)a5 + 8LL) = 0;
        }
      }
LABEL_7:
      if ( v67 == ++v15 )
        goto LABEL_3;
    }
  }
LABEL_25:
  v24 = *a10;
  v25 = *(unsigned int *)(a8 + 8);
  if ( !(_DWORD)v25 || *a11 != 1 && v24 )
    return 0;
  if ( *a9 == 1 || !v24 )
  {
    v28 = *(char **)a8;
    v30 = *(char **)a8;
    if ( v24 || *a9 )
    {
      if ( (_DWORD)v25 != 1 )
        *(_DWORD *)(a8 + 8) = 1;
      v32 = v28 + 8;
      goto LABEL_87;
    }
    v43 = &v28[8 * v25];
    do
    {
      v44 = *(_QWORD *)v28;
      if ( *(_BYTE *)(*(_QWORD *)(*(_QWORD *)v28 + 24LL) + 8LL) != 12 )
      {
        v45 = sub_BCB060(*(_QWORD *)v28);
        v46 = (__int64 *)sub_BCD140(*(_QWORD **)v44, v45);
        v47 = *(unsigned __int8 *)(v44 + 8);
        if ( (unsigned int)(v47 - 17) <= 1 )
        {
          BYTE4(v73) = (_BYTE)v47 == 18;
          LODWORD(v73) = *(_DWORD *)(v44 + 32);
          v46 = (__int64 *)sub_BCE1B0(v46, v73);
        }
        *(_QWORD *)v28 = v46;
      }
      v28 += 8;
    }
    while ( v43 != v28 );
    v28 = *(char **)a8;
    v48 = 8LL * *(unsigned int *)(a8 + 8);
    v30 = (char *)(*(_QWORD *)a8 + v48);
    if ( *(char **)a8 == v30 )
    {
      v55 = *(_QWORD *)a8;
    }
    else
    {
      _BitScanReverse64(&v49, v48 >> 3);
      sub_2912C10(*(char **)a8, (__int64 *)(*(_QWORD *)a8 + v48), 2LL * (int)(63 - (v49 ^ 0x3F)), a7);
      if ( (unsigned __int64)v48 <= 0x80 )
      {
        sub_2912570(v28, v30);
      }
      else
      {
        v50 = v28 + 128;
        sub_2912570(v28, v28 + 128);
        if ( v30 != v28 + 128 )
        {
          do
          {
            v51 = *(_QWORD *)v50;
            v52 = *((_QWORD *)v50 - 1);
            v53 = v50 - 8;
            if ( *(_DWORD *)(v52 + 32) <= *(_DWORD *)(*(_QWORD *)v50 + 32LL) )
            {
              v54 = v50;
            }
            else
            {
              do
              {
                *((_QWORD *)v53 + 1) = v52;
                v54 = v53;
                v52 = *((_QWORD *)v53 - 1);
                v53 -= 8;
              }
              while ( *(_DWORD *)(v51 + 32) < *(_DWORD *)(v52 + 32) );
            }
            v50 += 8;
            *(_QWORD *)v54 = v51;
          }
          while ( v30 != v50 );
        }
      }
      v28 = *(char **)a8;
      v55 = *(_QWORD *)a8 + 8LL * *(unsigned int *)(a8 + 8);
      v30 = *(char **)a8;
      if ( *(_QWORD *)a8 != v55 )
      {
        v56 = *(char **)a8;
        while ( 1 )
        {
          v58 = v56;
          v56 += 8;
          if ( v56 == (char *)v55 )
            break;
          v57 = *(_DWORD *)(*((_QWORD *)v56 - 1) + 32LL);
          if ( v57 == *(_DWORD *)(*(_QWORD *)v56 + 32LL) )
          {
            if ( v58 == (char *)v55 )
              break;
            v60 = v58 + 16;
            if ( v58 + 16 != (char *)v55 )
            {
              while ( 1 )
              {
                if ( *(_DWORD *)(*(_QWORD *)v60 + 32LL) != v57 )
                {
                  *((_QWORD *)v58 + 1) = *(_QWORD *)v60;
                  v58 += 8;
                }
                v60 += 8;
                if ( v60 == (char *)v55 )
                  break;
                v57 = *(_DWORD *)(*(_QWORD *)v58 + 32LL);
              }
              v28 = *(char **)a8;
              v30 = *(char **)a8;
              v61 = *(_QWORD *)a8 + 8LL * *(unsigned int *)(a8 + 8) - (_QWORD)v60;
              v56 = &v58[v61 + 8];
              if ( v60 != (char *)(*(_QWORD *)a8 + 8LL * *(unsigned int *)(a8 + 8)) )
              {
                memmove(v58 + 8, v60, v61);
                v28 = *(char **)a8;
                v30 = *(char **)a8;
              }
            }
            goto LABEL_72;
          }
        }
      }
    }
    v56 = (char *)v55;
LABEL_72:
    v59 = (v56 - v30) >> 3;
    *(_DWORD *)(a8 + 8) = v59;
    v29 = (unsigned int)v59;
  }
  else
  {
    v26 = *a12;
    *(_DWORD *)(a8 + 8) = 0;
    v27 = 0;
    if ( !*(_DWORD *)(a8 + 12) )
    {
      sub_C8D5F0(a8, (const void *)(a8 + 16), 1u, 8u, a5, a6);
      v27 = 8LL * *(unsigned int *)(a8 + 8);
    }
    *(_QWORD *)(*(_QWORD *)a8 + v27) = v26;
    v28 = *(char **)a8;
    v29 = (unsigned int)(*(_DWORD *)(a8 + 8) + 1);
    *(_DWORD *)(a8 + 8) = v29;
    v30 = v28;
  }
  v31 = 8 * v29;
  v32 = &v30[v31];
  v33 = v31 >> 5;
  if ( v33 )
  {
    v34 = &v30[32 * v33];
    while ( *(_DWORD *)(*(_QWORD *)v30 + 32LL) <= 0xFFFFu )
    {
      if ( *(_DWORD *)(*((_QWORD *)v30 + 1) + 32LL) > 0xFFFFu )
      {
        v30 += 8;
        goto LABEL_40;
      }
      if ( *(_DWORD *)(*((_QWORD *)v30 + 2) + 32LL) > 0xFFFFu )
      {
        v30 += 16;
        goto LABEL_40;
      }
      if ( *(_DWORD *)(*((_QWORD *)v30 + 3) + 32LL) > 0xFFFFu )
      {
        v30 += 24;
        goto LABEL_40;
      }
      v30 += 32;
      if ( v34 == v30 )
        goto LABEL_92;
    }
    goto LABEL_40;
  }
LABEL_92:
  v62 = (v32 - v30) >> 3;
  if ( v32 - v30 == 16 )
  {
LABEL_101:
    if ( *(_DWORD *)(*(_QWORD *)v30 + 32LL) > 0xFFFFu )
      goto LABEL_40;
    v30 += 8;
    goto LABEL_87;
  }
  if ( v62 == 3 )
  {
    if ( *(_DWORD *)(*(_QWORD *)v30 + 32LL) > 0xFFFFu )
      goto LABEL_40;
    v30 += 8;
    goto LABEL_101;
  }
  if ( v62 != 1 )
    goto LABEL_95;
LABEL_87:
  if ( *(_DWORD *)(*(_QWORD *)v30 + 32LL) <= 0xFFFFu )
  {
LABEL_95:
    v30 = v32;
    goto LABEL_46;
  }
LABEL_40:
  if ( v32 != v30 )
  {
    v35 = v30 + 8;
    if ( v32 != v30 + 8 )
    {
      do
      {
        if ( *(_DWORD *)(*(_QWORD *)v35 + 32LL) <= 0xFFFFu )
        {
          *(_QWORD *)v30 = *(_QWORD *)v35;
          v30 += 8;
        }
        v35 += 8;
      }
      while ( v32 != v35 );
      v28 = *(char **)a8;
    }
  }
LABEL_46:
  v36 = (__int64 *)v28;
  v37 = &v28[8 * *(unsigned int *)(a8 + 8)];
  v38 = v37 - v32;
  if ( v32 != v37 )
  {
    memmove(v30, v32, v38);
    v36 = *(__int64 **)a8;
  }
  v39 = (&v30[v38] - (char *)v36) >> 3;
  *(_DWORD *)(a8 + 8) = v39;
  v40 = &v36[(unsigned int)v39];
  if ( v40 == v36 )
    return 0;
  while ( 1 )
  {
    v41 = *v36;
    if ( (unsigned __int8)sub_2919680(v63, *v36, a7) )
      break;
    if ( v40 == ++v36 )
      return 0;
  }
  return v41;
}
