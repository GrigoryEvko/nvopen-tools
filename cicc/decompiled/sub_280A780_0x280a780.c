// Function: sub_280A780
// Address: 0x280a780
//
__int64 __fastcall sub_280A780(__int64 a1, __int64 a2)
{
  __int64 *v4; // r12
  __int64 *v5; // r14
  _QWORD *v6; // rdi
  _QWORD *v7; // rsi
  __int64 v8; // rdx
  __int64 v9; // rcx
  __int64 v10; // r8
  __int64 v11; // r9
  int v12; // eax
  unsigned int v13; // esi
  __int64 v14; // r9
  _QWORD *v15; // r11
  int v16; // r15d
  unsigned int v17; // edx
  _QWORD *v18; // r8
  __int64 v19; // rdi
  int v20; // eax
  __int64 v21; // r15
  __int64 v22; // rax
  int v23; // eax
  __int64 v24; // rdx
  _QWORD *v25; // rax
  _QWORD *i; // rdx
  __int64 result; // rax
  unsigned int v28; // ecx
  unsigned int v29; // eax
  _QWORD *v30; // rdi
  __int64 v31; // r12
  _QWORD *v32; // rax
  int v33; // eax
  int v34; // ecx
  __int64 v35; // rdx
  __int64 v36; // rdi
  int v37; // r10d
  int v38; // eax
  int v39; // ecx
  int v40; // r10d
  __int64 v41; // rdx
  __int64 v42; // rdi
  unsigned __int64 v43; // rdx
  unsigned __int64 v44; // rax
  _QWORD *v45; // rax
  __int64 v46; // rdx
  _QWORD *j; // rdx
  const void *v48; // [rsp+8h] [rbp-38h]

  v4 = *(__int64 **)(a1 + 32);
  v5 = &v4[*(unsigned int *)(a1 + 40)];
  v48 = (const void *)(a2 + 48);
  if ( v4 != v5 )
  {
    while ( 1 )
    {
      v12 = *(_DWORD *)(a2 + 16);
      if ( !v12 )
        break;
      v13 = *(_DWORD *)(a2 + 24);
      if ( !v13 )
      {
        ++*(_QWORD *)a2;
        goto LABEL_40;
      }
      v14 = *(_QWORD *)(a2 + 8);
      v15 = 0;
      v16 = 1;
      v17 = (v13 - 1) & (((unsigned int)*v4 >> 9) ^ ((unsigned int)*v4 >> 4));
      v18 = (_QWORD *)(v14 + 8LL * v17);
      v19 = *v18;
      if ( *v4 == *v18 )
      {
LABEL_5:
        if ( ++v4 == v5 )
          goto LABEL_19;
      }
      else
      {
        while ( v19 != -4096 )
        {
          if ( v15 || v19 != -8192 )
            v18 = v15;
          v17 = (v13 - 1) & (v16 + v17);
          v19 = *(_QWORD *)(v14 + 8LL * v17);
          if ( *v4 == v19 )
            goto LABEL_5;
          ++v16;
          v15 = v18;
          v18 = (_QWORD *)(v14 + 8LL * v17);
        }
        if ( !v15 )
          v15 = v18;
        v20 = v12 + 1;
        ++*(_QWORD *)a2;
        if ( 4 * v20 < 3 * v13 )
        {
          if ( v13 - *(_DWORD *)(a2 + 20) - v20 > v13 >> 3 )
            goto LABEL_14;
          sub_CF4090(a2, v13);
          v38 = *(_DWORD *)(a2 + 24);
          if ( !v38 )
          {
LABEL_71:
            ++*(_DWORD *)(a2 + 16);
            BUG();
          }
          v39 = v38 - 1;
          v18 = *(_QWORD **)(a2 + 8);
          v14 = 0;
          v40 = 1;
          LODWORD(v41) = (v38 - 1) & (((unsigned int)*v4 >> 9) ^ ((unsigned int)*v4 >> 4));
          v15 = &v18[(unsigned int)v41];
          v42 = *v15;
          v20 = *(_DWORD *)(a2 + 16) + 1;
          if ( *v4 == *v15 )
            goto LABEL_14;
          while ( v42 != -4096 )
          {
            if ( v42 == -8192 && !v14 )
              v14 = (__int64)v15;
            v41 = v39 & (unsigned int)(v41 + v40);
            v15 = &v18[v41];
            v42 = *v15;
            if ( *v4 == *v15 )
              goto LABEL_14;
            ++v40;
          }
          goto LABEL_52;
        }
LABEL_40:
        sub_CF4090(a2, 2 * v13);
        v33 = *(_DWORD *)(a2 + 24);
        if ( !v33 )
          goto LABEL_71;
        v34 = v33 - 1;
        v18 = *(_QWORD **)(a2 + 8);
        LODWORD(v35) = (v33 - 1) & (((unsigned int)*v4 >> 9) ^ ((unsigned int)*v4 >> 4));
        v15 = &v18[(unsigned int)v35];
        v36 = *v15;
        v20 = *(_DWORD *)(a2 + 16) + 1;
        if ( *v15 == *v4 )
          goto LABEL_14;
        v37 = 1;
        v14 = 0;
        while ( v36 != -4096 )
        {
          if ( v36 == -8192 && !v14 )
            v14 = (__int64)v15;
          v35 = v34 & (unsigned int)(v35 + v37);
          v15 = &v18[v35];
          v36 = *v15;
          if ( *v4 == *v15 )
            goto LABEL_14;
          ++v37;
        }
LABEL_52:
        if ( v14 )
          v15 = (_QWORD *)v14;
LABEL_14:
        *(_DWORD *)(a2 + 16) = v20;
        if ( *v15 != -4096 )
          --*(_DWORD *)(a2 + 20);
        v21 = *v4;
        *v15 = *v4;
        v22 = *(unsigned int *)(a2 + 40);
        if ( v22 + 1 > (unsigned __int64)*(unsigned int *)(a2 + 44) )
        {
          sub_C8D5F0(a2 + 32, v48, v22 + 1, 8u, (__int64)v18, v14);
          v22 = *(unsigned int *)(a2 + 40);
        }
        ++v4;
        *(_QWORD *)(*(_QWORD *)(a2 + 32) + 8 * v22) = v21;
        ++*(_DWORD *)(a2 + 40);
        if ( v4 == v5 )
          goto LABEL_19;
      }
    }
    v6 = *(_QWORD **)(a2 + 32);
    v7 = &v6[*(unsigned int *)(a2 + 40)];
    if ( v7 == sub_28086A0(v6, (__int64)v7, v4) )
      sub_280A4F0(a2, *v4, v8, v9, v10, v11);
    goto LABEL_5;
  }
LABEL_19:
  v23 = *(_DWORD *)(a1 + 16);
  ++*(_QWORD *)a1;
  if ( !v23 )
  {
    if ( !*(_DWORD *)(a1 + 20) )
      goto LABEL_25;
    v24 = *(unsigned int *)(a1 + 24);
    if ( (unsigned int)v24 > 0x40 )
    {
      sub_C7D6A0(*(_QWORD *)(a1 + 8), 8 * v24, 8);
      *(_QWORD *)(a1 + 8) = 0;
      *(_QWORD *)(a1 + 16) = 0;
      *(_DWORD *)(a1 + 24) = 0;
      goto LABEL_25;
    }
    goto LABEL_22;
  }
  v28 = 4 * v23;
  v24 = *(unsigned int *)(a1 + 24);
  if ( (unsigned int)(4 * v23) < 0x40 )
    v28 = 64;
  if ( v28 >= (unsigned int)v24 )
  {
LABEL_22:
    v25 = *(_QWORD **)(a1 + 8);
    for ( i = &v25[v24]; i != v25; ++v25 )
      *v25 = -4096;
    *(_QWORD *)(a1 + 16) = 0;
    goto LABEL_25;
  }
  v29 = v23 - 1;
  if ( v29 )
  {
    _BitScanReverse(&v29, v29);
    v30 = *(_QWORD **)(a1 + 8);
    v31 = (unsigned int)(1 << (33 - (v29 ^ 0x1F)));
    if ( (int)v31 < 64 )
      v31 = 64;
    if ( (_DWORD)v31 == (_DWORD)v24 )
    {
      *(_QWORD *)(a1 + 16) = 0;
      v32 = &v30[v31];
      do
      {
        if ( v30 )
          *v30 = -4096;
        ++v30;
      }
      while ( v32 != v30 );
      goto LABEL_25;
    }
  }
  else
  {
    v30 = *(_QWORD **)(a1 + 8);
    LODWORD(v31) = 64;
  }
  sub_C7D6A0((__int64)v30, 8 * v24, 8);
  v43 = ((((((((4 * (int)v31 / 3u + 1) | ((unsigned __int64)(4 * (int)v31 / 3u + 1) >> 1)) >> 2)
           | (4 * (int)v31 / 3u + 1)
           | ((unsigned __int64)(4 * (int)v31 / 3u + 1) >> 1)) >> 4)
         | (((4 * (int)v31 / 3u + 1) | ((unsigned __int64)(4 * (int)v31 / 3u + 1) >> 1)) >> 2)
         | (4 * (int)v31 / 3u + 1)
         | ((unsigned __int64)(4 * (int)v31 / 3u + 1) >> 1)) >> 8)
       | (((((4 * (int)v31 / 3u + 1) | ((unsigned __int64)(4 * (int)v31 / 3u + 1) >> 1)) >> 2)
         | (4 * (int)v31 / 3u + 1)
         | ((unsigned __int64)(4 * (int)v31 / 3u + 1) >> 1)) >> 4)
       | (((4 * (int)v31 / 3u + 1) | ((unsigned __int64)(4 * (int)v31 / 3u + 1) >> 1)) >> 2)
       | (4 * (int)v31 / 3u + 1)
       | ((unsigned __int64)(4 * (int)v31 / 3u + 1) >> 1)) >> 16;
  v44 = (v43
       | (((((((4 * (int)v31 / 3u + 1) | ((unsigned __int64)(4 * (int)v31 / 3u + 1) >> 1)) >> 2)
           | (4 * (int)v31 / 3u + 1)
           | ((unsigned __int64)(4 * (int)v31 / 3u + 1) >> 1)) >> 4)
         | (((4 * (int)v31 / 3u + 1) | ((unsigned __int64)(4 * (int)v31 / 3u + 1) >> 1)) >> 2)
         | (4 * (int)v31 / 3u + 1)
         | ((unsigned __int64)(4 * (int)v31 / 3u + 1) >> 1)) >> 8)
       | (((((4 * (int)v31 / 3u + 1) | ((unsigned __int64)(4 * (int)v31 / 3u + 1) >> 1)) >> 2)
         | (4 * (int)v31 / 3u + 1)
         | ((unsigned __int64)(4 * (int)v31 / 3u + 1) >> 1)) >> 4)
       | (((4 * (int)v31 / 3u + 1) | ((unsigned __int64)(4 * (int)v31 / 3u + 1) >> 1)) >> 2)
       | (4 * (int)v31 / 3u + 1)
       | ((unsigned __int64)(4 * (int)v31 / 3u + 1) >> 1))
      + 1;
  *(_DWORD *)(a1 + 24) = v44;
  v45 = (_QWORD *)sub_C7D670(8 * v44, 8);
  v46 = *(unsigned int *)(a1 + 24);
  *(_QWORD *)(a1 + 16) = 0;
  *(_QWORD *)(a1 + 8) = v45;
  for ( j = &v45[v46]; j != v45; ++v45 )
  {
    if ( v45 )
      *v45 = -4096;
  }
LABEL_25:
  *(_DWORD *)(a1 + 40) = 0;
  result = *(unsigned __int8 *)(a1 + 112);
  *(_BYTE *)(a2 + 112) |= result;
  return result;
}
