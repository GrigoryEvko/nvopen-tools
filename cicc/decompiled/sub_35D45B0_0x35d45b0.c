// Function: sub_35D45B0
// Address: 0x35d45b0
//
__int64 __fastcall sub_35D45B0(__int64 a1, __int64 a2)
{
  __int64 v3; // rdi
  __int64 (*v4)(); // rax
  __int64 (*v5)(); // rdx
  __int64 v6; // rax
  __int64 result; // rax
  __int64 v8; // rax
  __int64 i; // rdx
  __int64 v10; // rcx
  __int64 v11; // r8
  __int64 v12; // r9
  int v13; // eax
  __int64 v14; // r13
  __int64 v15; // rbx
  __int64 v16; // rdx
  __int64 j; // r13
  unsigned __int64 v18; // rcx
  __int64 v19; // r15
  __int64 v20; // r14
  __int64 k; // r15
  __int64 v22; // rbx
  __int64 m; // r13
  _QWORD *v24; // rax
  __int64 v25; // rdx
  __int64 v26; // rcx
  unsigned int v27; // eax
  _QWORD *v28; // rdi
  int v29; // eax
  int v30; // ebx
  _QWORD *v31; // rax
  unsigned __int64 v32; // rdx
  unsigned __int64 v33; // rax
  _QWORD *v34; // rax
  __int64 v35; // rdx

  *(_QWORD *)a1 = a2;
  *(_QWORD *)(a1 + 8) = *(_QWORD *)a2;
  v3 = 0;
  v4 = *(__int64 (**)())(**(_QWORD **)(a2 + 16) + 144LL);
  if ( v4 != sub_2C8F680 )
  {
    v8 = ((__int64 (__fastcall *)(_QWORD))v4)(*(_QWORD *)(a2 + 16));
    a2 = *(_QWORD *)a1;
    v3 = v8;
  }
  *(_QWORD *)(a1 + 16) = v3;
  v5 = *(__int64 (**)())(**(_QWORD **)(a2 + 16) + 128LL);
  v6 = 0;
  if ( v5 != sub_2DAC790 )
  {
    v6 = ((__int64 (__fastcall *)(_QWORD))v5)(*(_QWORD *)(a2 + 16));
    v3 = *(_QWORD *)(a1 + 16);
  }
  *(_QWORD *)(a1 + 24) = v6;
  result = *(_QWORD *)(*(_QWORD *)v3 + 2216LL);
  if ( (__int64 (*)())result == sub_302E1B0 )
    return result;
  result = ((__int64 (*)(void))result)();
  if ( !(_BYTE)result )
    return result;
  *(_DWORD *)(a1 + 144) = 0;
  sub_35D43C0(a1 + 32);
  sub_35D43C0(a1 + 64);
  v13 = *(_DWORD *)(a1 + 112);
  ++*(_QWORD *)(a1 + 96);
  if ( v13 )
  {
    v10 = (unsigned int)(4 * v13);
    a2 = 64;
    i = *(unsigned int *)(a1 + 120);
    if ( (unsigned int)v10 < 0x40 )
      v10 = 64;
    if ( (unsigned int)i > (unsigned int)v10 )
    {
      v27 = v13 - 1;
      if ( v27 )
      {
        _BitScanReverse(&v27, v27);
        v28 = *(_QWORD **)(a1 + 104);
        v29 = v27 ^ 0x1F;
        v10 = (unsigned int)(33 - v29);
        v30 = 1 << (33 - v29);
        if ( v30 < 64 )
          v30 = 64;
        if ( v30 == (_DWORD)i )
        {
          *(_QWORD *)(a1 + 112) = 0;
          v31 = &v28[2 * (unsigned int)v30];
          do
          {
            if ( v28 )
              *v28 = -4;
            v28 += 2;
          }
          while ( v31 != v28 );
          goto LABEL_12;
        }
      }
      else
      {
        v28 = *(_QWORD **)(a1 + 104);
        v30 = 64;
      }
      sub_C7D6A0((__int64)v28, 16LL * *(unsigned int *)(a1 + 120), 8);
      a2 = 8;
      v32 = ((((((((4 * v30 / 3u + 1) | ((unsigned __int64)(4 * v30 / 3u + 1) >> 1)) >> 2)
               | (4 * v30 / 3u + 1)
               | ((unsigned __int64)(4 * v30 / 3u + 1) >> 1)) >> 4)
             | (((4 * v30 / 3u + 1) | ((unsigned __int64)(4 * v30 / 3u + 1) >> 1)) >> 2)
             | (4 * v30 / 3u + 1)
             | ((unsigned __int64)(4 * v30 / 3u + 1) >> 1)) >> 8)
           | (((((4 * v30 / 3u + 1) | ((unsigned __int64)(4 * v30 / 3u + 1) >> 1)) >> 2)
             | (4 * v30 / 3u + 1)
             | ((unsigned __int64)(4 * v30 / 3u + 1) >> 1)) >> 4)
           | (((4 * v30 / 3u + 1) | ((unsigned __int64)(4 * v30 / 3u + 1) >> 1)) >> 2)
           | (4 * v30 / 3u + 1)
           | ((unsigned __int64)(4 * v30 / 3u + 1) >> 1)) >> 16;
      v33 = (v32
           | (((((((4 * v30 / 3u + 1) | ((unsigned __int64)(4 * v30 / 3u + 1) >> 1)) >> 2)
               | (4 * v30 / 3u + 1)
               | ((unsigned __int64)(4 * v30 / 3u + 1) >> 1)) >> 4)
             | (((4 * v30 / 3u + 1) | ((unsigned __int64)(4 * v30 / 3u + 1) >> 1)) >> 2)
             | (4 * v30 / 3u + 1)
             | ((unsigned __int64)(4 * v30 / 3u + 1) >> 1)) >> 8)
           | (((((4 * v30 / 3u + 1) | ((unsigned __int64)(4 * v30 / 3u + 1) >> 1)) >> 2)
             | (4 * v30 / 3u + 1)
             | ((unsigned __int64)(4 * v30 / 3u + 1) >> 1)) >> 4)
           | (((4 * v30 / 3u + 1) | ((unsigned __int64)(4 * v30 / 3u + 1) >> 1)) >> 2)
           | (4 * v30 / 3u + 1)
           | ((unsigned __int64)(4 * v30 / 3u + 1) >> 1))
          + 1;
      *(_DWORD *)(a1 + 120) = v33;
      v34 = (_QWORD *)sub_C7D670(16 * v33, 8);
      v35 = *(unsigned int *)(a1 + 120);
      *(_QWORD *)(a1 + 112) = 0;
      *(_QWORD *)(a1 + 104) = v34;
      for ( i = (__int64)&v34[2 * v35]; (_QWORD *)i != v34; v34 += 2 )
      {
        if ( v34 )
          *v34 = -4;
      }
      goto LABEL_12;
    }
    goto LABEL_37;
  }
  if ( *(_DWORD *)(a1 + 116) )
  {
    i = *(unsigned int *)(a1 + 120);
    if ( (unsigned int)i > 0x40 )
    {
      a2 = 16LL * *(unsigned int *)(a1 + 120);
      sub_C7D6A0(*(_QWORD *)(a1 + 104), a2, 8);
      *(_QWORD *)(a1 + 104) = 0;
      *(_QWORD *)(a1 + 112) = 0;
      *(_DWORD *)(a1 + 120) = 0;
      goto LABEL_12;
    }
LABEL_37:
    v24 = *(_QWORD **)(a1 + 104);
    for ( i = (__int64)&v24[2 * i]; (_QWORD *)i != v24; v24 += 2 )
      *v24 = -4;
    *(_QWORD *)(a1 + 112) = 0;
  }
LABEL_12:
  v14 = *(_QWORD *)(a1 + 8);
  *(_QWORD *)(a1 + 128) = 0;
  if ( (*(_BYTE *)(v14 + 2) & 1) != 0 )
  {
    sub_B2C6D0(v14, a2, i, v10);
    v15 = *(_QWORD *)(v14 + 96);
    v14 = *(_QWORD *)(a1 + 8);
    if ( (*(_BYTE *)(v14 + 2) & 1) != 0 )
      sub_B2C6D0(*(_QWORD *)(a1 + 8), a2, v25, v26);
    v16 = *(_QWORD *)(v14 + 96);
  }
  else
  {
    v15 = *(_QWORD *)(v14 + 96);
    v16 = v15;
  }
  result = 5LL * *(_QWORD *)(v14 + 104);
  for ( j = v16 + 40LL * *(_QWORD *)(v14 + 104); v15 != j; ++*(_DWORD *)(a1 + 144) )
  {
    while ( 1 )
    {
      result = sub_B2D650(v15);
      if ( (_BYTE)result )
        break;
      v15 += 40;
      if ( v15 == j )
        goto LABEL_21;
    }
    result = *(unsigned int *)(a1 + 144);
    v18 = *(unsigned int *)(a1 + 148);
    *(_QWORD *)(a1 + 128) = v15;
    if ( result + 1 > v18 )
    {
      sub_C8D5F0(a1 + 136, (const void *)(a1 + 152), result + 1, 8u, v11, v12);
      result = *(unsigned int *)(a1 + 144);
    }
    *(_QWORD *)(*(_QWORD *)(a1 + 136) + 8 * result) = v15;
    v15 += 40;
  }
LABEL_21:
  v19 = *(_QWORD *)(a1 + 8);
  v20 = *(_QWORD *)(v19 + 80);
  for ( k = v19 + 72; k != v20; v20 = *(_QWORD *)(v20 + 8) )
  {
    if ( !v20 )
      BUG();
    v22 = *(_QWORD *)(v20 + 32);
    for ( m = v20 + 24; m != v22; v22 = *(_QWORD *)(v22 + 8) )
    {
      while ( 1 )
      {
        if ( !v22 )
          BUG();
        if ( *(_BYTE *)(v22 - 24) == 60 && *(char *)(v22 - 22) < 0 )
          break;
        v22 = *(_QWORD *)(v22 + 8);
        if ( m == v22 )
          goto LABEL_32;
      }
      result = *(unsigned int *)(a1 + 144);
      if ( result + 1 > (unsigned __int64)*(unsigned int *)(a1 + 148) )
      {
        sub_C8D5F0(a1 + 136, (const void *)(a1 + 152), result + 1, 8u, v11, v12);
        result = *(unsigned int *)(a1 + 144);
      }
      *(_QWORD *)(*(_QWORD *)(a1 + 136) + 8 * result) = v22 - 24;
      ++*(_DWORD *)(a1 + 144);
    }
LABEL_32:
    ;
  }
  return result;
}
