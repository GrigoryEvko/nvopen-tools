// Function: sub_2E58D20
// Address: 0x2e58d20
//
unsigned __int64 __fastcall sub_2E58D20(__int64 a1, __int64 a2)
{
  __int64 v3; // r13
  __int64 v4; // r13
  __int64 v5; // rax
  __int64 v6; // rbx
  unsigned int v7; // r14d
  __int64 v8; // r8
  __int64 *v9; // rax
  unsigned __int64 v10; // r13
  unsigned __int64 v11; // rdi
  unsigned __int64 v12; // rdi
  __int64 v13; // rdx
  __int64 v14; // rcx
  __int64 v15; // rbx
  __int64 v16; // r8
  __int64 v17; // r9
  __int64 *v18; // r13
  __int64 v19; // rdx
  unsigned int v20; // eax
  __int64 v21; // rdi
  int v22; // eax
  __int64 v23; // r10
  unsigned int v24; // ecx
  __int64 *v25; // rdx
  __int64 v26; // r11
  __int64 v27; // rdi
  __int64 v28; // rsi
  int v29; // eax
  unsigned int v30; // edi
  __int64 *v31; // rax
  __int64 v32; // r15
  unsigned int v33; // eax
  __int64 v34; // rax
  __int64 v35; // rdx
  __int64 v36; // rsi
  _QWORD *v37; // rdx
  __int64 v38; // rcx
  int v39; // edx
  int v40; // ecx
  unsigned __int64 v41; // rcx
  unsigned __int64 v42; // r8
  int v43; // eax
  __int64 v44; // r14
  unsigned __int64 v45; // r13
  unsigned __int64 result; // rax
  __int64 v47; // r14
  unsigned __int64 *v48; // rax
  unsigned __int64 *v49; // rdx
  __int64 v50; // r9
  size_t v51; // r15
  unsigned __int64 i; // r13
  int v53; // [rsp+0h] [rbp-60h]
  __int64 v54; // [rsp+0h] [rbp-60h]
  int v55; // [rsp+0h] [rbp-60h]
  __int64 v56; // [rsp+8h] [rbp-58h]
  const void *v57; // [rsp+8h] [rbp-58h]
  __int64 *s; // [rsp+10h] [rbp-50h]
  void *sa; // [rsp+10h] [rbp-50h]
  __int64 v60[2]; // [rsp+18h] [rbp-48h] BYREF
  __int64 v61[7]; // [rsp+28h] [rbp-38h] BYREF

  v3 = *(_QWORD *)(a1 + 96) - *(_QWORD *)(a1 + 88);
  v60[0] = a2;
  v4 = v3 >> 2;
  v5 = sub_22077B0(0xA8u);
  v6 = v5;
  if ( v5 )
  {
    *(_QWORD *)v5 = 0;
    v7 = (unsigned int)(v4 + 63) >> 6;
    *(_QWORD *)(v5 + 8) = 0;
    v8 = v5 + 112;
    *(_QWORD *)(v5 + 16) = 0;
    *(_QWORD *)(v5 + 24) = v5 + 40;
    *(_QWORD *)(v5 + 32) = 0x600000000LL;
    if ( v7 > 6 )
    {
      v57 = (const void *)(v5 + 112);
      sub_C8D5F0(v5 + 24, (const void *)(v5 + 40), v7, 8u, v8, 0x600000000LL);
      memset(*(void **)(v6 + 24), 0, 8LL * v7);
      *(_DWORD *)(v6 + 32) = v7;
      *(_DWORD *)(v6 + 88) = v4;
      *(_QWORD *)(v6 + 96) = v57;
      *(_QWORD *)(v6 + 104) = 0x600000000LL;
      sub_C8D5F0(v6 + 96, v57, v7, 8u, (__int64)v57, 0x600000000LL);
      memset(*(void **)(v6 + 96), 0, 8LL * v7);
      *(_DWORD *)(v6 + 104) = v7;
    }
    else
    {
      if ( v7 )
      {
        v51 = 8LL * v7;
        sa = (void *)(v5 + 112);
        memset((void *)(v5 + 40), 0, v51);
        *(_DWORD *)(v6 + 32) = v7;
        *(_DWORD *)(v6 + 88) = v4;
        *(_QWORD *)(v6 + 96) = sa;
        *(_DWORD *)(v6 + 108) = 6;
        memset(sa, 0, (size_t)sa + v51 - v6 - 112);
      }
      else
      {
        *(_DWORD *)(v5 + 32) = 0;
        *(_DWORD *)(v5 + 88) = v4;
        *(_QWORD *)(v5 + 96) = v8;
        *(_DWORD *)(v5 + 108) = 6;
      }
      *(_DWORD *)(v6 + 104) = v7;
    }
    *(_DWORD *)(v6 + 160) = v4;
  }
  v9 = sub_2E57C80(a1 + 112, v60);
  v10 = *v9;
  *v9 = v6;
  if ( v10 )
  {
    v11 = *(_QWORD *)(v10 + 96);
    if ( v11 != v10 + 112 )
      _libc_free(v11);
    v12 = *(_QWORD *)(v10 + 24);
    if ( v12 != v10 + 40 )
      _libc_free(v12);
    j_j___libc_free_0(v10);
  }
  v15 = *sub_2E57C80(a1 + 112, v60);
  v18 = *(__int64 **)(v60[0] + 112);
  s = &v18[*(unsigned int *)(v60[0] + 120)];
  v56 = v15 + 96;
  if ( s != v18 )
  {
    while ( 1 )
    {
      v61[0] = *v18;
      sub_307C930(a1);
      if ( v61[0] != v60[0] )
        break;
LABEL_33:
      if ( s == ++v18 )
        goto LABEL_47;
    }
    v16 = *(_QWORD *)(a1 + 16);
    if ( v60[0] )
    {
      v19 = (unsigned int)(*(_DWORD *)(v60[0] + 24) + 1);
      v20 = *(_DWORD *)(v60[0] + 24) + 1;
    }
    else
    {
      v19 = 0;
      v20 = 0;
    }
    v17 = *(unsigned int *)(v16 + 32);
    v21 = 0;
    if ( v20 < (unsigned int)v17 )
      v21 = *(_QWORD *)(*(_QWORD *)(v16 + 24) + 8 * v19);
    v22 = *(_DWORD *)(a1 + 168);
    v23 = *(_QWORD *)(a1 + 152);
    if ( v22 )
    {
      v24 = (v22 - 1) & (((unsigned int)v21 >> 9) ^ ((unsigned int)v21 >> 4));
      v25 = (__int64 *)(v23 + 16LL * v24);
      v26 = *v25;
      if ( v21 == *v25 )
      {
LABEL_21:
        v14 = *((unsigned int *)v25 + 2);
        goto LABEL_22;
      }
      v39 = 1;
      while ( v26 != -4096 )
      {
        v24 = (v22 - 1) & (v39 + v24);
        v55 = v39 + 1;
        v25 = (__int64 *)(v23 + 16LL * v24);
        v26 = *v25;
        if ( v21 == *v25 )
          goto LABEL_21;
        v39 = v55;
      }
    }
    v14 = 0;
LABEL_22:
    if ( v61[0] )
    {
      v27 = (unsigned int)(*(_DWORD *)(v61[0] + 24) + 1);
      v13 = v27;
    }
    else
    {
      v27 = 0;
      v13 = 0;
    }
    v28 = 0;
    if ( (unsigned int)v17 > (unsigned int)v13 )
    {
      v13 = *(_QWORD *)(v16 + 24);
      v28 = *(_QWORD *)(v13 + 8 * v27);
    }
    if ( v22 )
    {
      v29 = v22 - 1;
      v30 = v29 & (((unsigned int)v28 >> 9) ^ ((unsigned int)v28 >> 4));
      v13 = v23 + 16LL * v30;
      v17 = *(_QWORD *)v13;
      if ( v28 == *(_QWORD *)v13 )
      {
LABEL_28:
        if ( *(_DWORD *)(v13 + 8) > (unsigned int)v14 )
        {
          v31 = sub_2E57C80(a1 + 112, v61);
          v32 = *v31;
          v33 = *(_DWORD *)(*v31 + 88);
          if ( *(_DWORD *)(v15 + 160) < v33 )
          {
            v40 = *(_DWORD *)(v15 + 160) & 0x3F;
            if ( v40 )
              *(_QWORD *)(*(_QWORD *)(v15 + 96) + 8LL * *(unsigned int *)(v15 + 104) - 8) &= ~(-1LL << v40);
            v41 = *(unsigned int *)(v15 + 104);
            *(_DWORD *)(v15 + 160) = v33;
            v42 = (v33 + 63) >> 6;
            if ( v42 != v41 )
            {
              if ( v42 >= v41 )
              {
                v50 = v42 - v41;
                if ( v42 > *(unsigned int *)(v15 + 108) )
                {
                  v54 = v42 - v41;
                  sub_C8D5F0(v56, (const void *)(v15 + 112), v42, 8u, v42, v50);
                  v41 = *(unsigned int *)(v15 + 104);
                  v50 = v54;
                }
                if ( 8 * v50 )
                {
                  v53 = v50;
                  memset((void *)(*(_QWORD *)(v15 + 96) + 8 * v41), 0, 8 * v50);
                  LODWORD(v41) = *(_DWORD *)(v15 + 104);
                  LODWORD(v50) = v53;
                }
                v33 = *(_DWORD *)(v15 + 160);
                *(_DWORD *)(v15 + 104) = v50 + v41;
              }
              else
              {
                *(_DWORD *)(v15 + 104) = (v33 + 63) >> 6;
              }
            }
            v43 = v33 & 0x3F;
            if ( v43 )
              *(_QWORD *)(*(_QWORD *)(v15 + 96) + 8LL * *(unsigned int *)(v15 + 104) - 8) &= ~(-1LL << v43);
          }
          v34 = 0;
          v35 = *(unsigned int *)(v32 + 32);
          v36 = 8 * v35;
          if ( (_DWORD)v35 )
          {
            do
            {
              v37 = (_QWORD *)(v34 + *(_QWORD *)(v15 + 96));
              v38 = *(_QWORD *)(*(_QWORD *)(v32 + 24) + v34);
              v34 += 8;
              *v37 |= v38;
            }
            while ( v36 != v34 );
          }
          sub_307C290(a1, v60[0], v61[0]);
        }
      }
      else
      {
        v13 = 1;
        while ( v17 != -4096 )
        {
          v16 = (unsigned int)(v13 + 1);
          v30 = v29 & (v13 + v30);
          v13 = v23 + 16LL * v30;
          v17 = *(_QWORD *)v13;
          if ( v28 == *(_QWORD *)v13 )
            goto LABEL_28;
          v13 = (unsigned int)v16;
        }
      }
    }
    goto LABEL_33;
  }
LABEL_47:
  sub_2E4EFA0(v15 + 24, v56, v13, v14, v16, v17);
  v44 = v60[0];
  *(_DWORD *)(v15 + 88) = *(_DWORD *)(v15 + 160);
  v45 = *(_QWORD *)(v44 + 48) & 0xFFFFFFFFFFFFFFF8LL;
  if ( !v45 )
LABEL_75:
    BUG();
  result = *(_QWORD *)v45;
  if ( (*(_QWORD *)v45 & 4) == 0 && (*(_BYTE *)(v45 + 44) & 4) != 0 )
  {
    for ( i = *(_QWORD *)v45; ; i = *(_QWORD *)v45 )
    {
      v45 = i & 0xFFFFFFFFFFFFFFF8LL;
      if ( (*(_BYTE *)(v45 + 44) & 4) == 0 )
        break;
    }
  }
  v47 = v44 + 48;
  while ( v47 != v45 )
  {
    sub_307D460(a1, v45, v15);
    v48 = (unsigned __int64 *)(*(_QWORD *)v45 & 0xFFFFFFFFFFFFFFF8LL);
    v49 = v48;
    if ( !v48 )
      goto LABEL_75;
    v45 = *(_QWORD *)v45 & 0xFFFFFFFFFFFFFFF8LL;
    result = *v48;
    if ( (result & 4) == 0 && (*((_BYTE *)v49 + 44) & 4) != 0 )
    {
      while ( 1 )
      {
        result &= 0xFFFFFFFFFFFFFFF8LL;
        v45 = result;
        if ( (*(_BYTE *)(result + 44) & 4) == 0 )
          break;
        result = *(_QWORD *)result;
      }
    }
  }
  return result;
}
