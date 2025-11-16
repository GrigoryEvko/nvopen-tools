// Function: sub_36FA670
// Address: 0x36fa670
//
__int64 __fastcall sub_36FA670(__int64 a1, __int64 *a2)
{
  int v4; // eax
  unsigned int v5; // ecx
  __int64 v6; // rdx
  _QWORD *v7; // rax
  _QWORD *i; // rdx
  unsigned int v9; // r15d
  __int64 v10; // r13
  __int64 *v11; // r12
  __int64 v12; // rcx
  _QWORD *v13; // rdx
  __int64 v14; // rdi
  __int64 (*v15)(void); // rax
  __int64 v16; // rbx
  unsigned __int64 v17; // rax
  char v18; // al
  _QWORD *v19; // rdx
  __int64 v20; // rcx
  __int64 v21; // rax
  unsigned int v22; // eax
  unsigned int v23; // eax
  __int64 *v24; // rbx
  __int64 *v25; // r12
  __int64 v26; // rdx
  __int64 v27; // rax
  __int64 v28; // rax
  unsigned int v29; // eax
  __int64 v30; // rdi
  int v31; // ebx
  unsigned __int64 v32; // rdx
  unsigned __int64 v33; // rax
  _QWORD *v34; // rax
  __int64 v35; // rdx
  _QWORD *v36; // rdx
  _QWORD *v37; // rax
  _QWORD *v38; // rdx
  __int64 *v39; // [rsp+8h] [rbp-58h]
  __int64 *v40; // [rsp+18h] [rbp-48h]
  __int64 v41; // [rsp+20h] [rbp-40h]
  __int64 v42; // [rsp+20h] [rbp-40h]
  _QWORD *v43; // [rsp+28h] [rbp-38h]
  _QWORD *v44; // [rsp+28h] [rbp-38h]

  if ( !*(_DWORD *)(a2[1] + 1280) )
    return sub_36F9CD0(a1, a2);
  v4 = *(_DWORD *)(a1 + 216);
  ++*(_QWORD *)(a1 + 200);
  if ( !v4 )
  {
    if ( !*(_DWORD *)(a1 + 220) )
      goto LABEL_10;
    v6 = *(unsigned int *)(a1 + 224);
    if ( (unsigned int)v6 > 0x40 )
    {
      sub_C7D6A0(*(_QWORD *)(a1 + 208), 8 * v6, 8);
      *(_QWORD *)(a1 + 208) = 0;
      *(_QWORD *)(a1 + 216) = 0;
      *(_DWORD *)(a1 + 224) = 0;
      goto LABEL_10;
    }
    goto LABEL_7;
  }
  v5 = 4 * v4;
  v6 = *(unsigned int *)(a1 + 224);
  if ( (unsigned int)(4 * v4) < 0x40 )
    v5 = 64;
  if ( v5 >= (unsigned int)v6 )
  {
LABEL_7:
    v7 = *(_QWORD **)(a1 + 208);
    for ( i = &v7[v6]; i != v7; ++v7 )
      *v7 = -4096;
    *(_QWORD *)(a1 + 216) = 0;
LABEL_10:
    v39 = a2 + 40;
    v40 = (__int64 *)a2[41];
    if ( a2 + 40 != v40 )
      goto LABEL_11;
    return 0;
  }
  v29 = v4 - 1;
  if ( v29 )
  {
    _BitScanReverse(&v29, v29);
    v30 = *(_QWORD *)(a1 + 208);
    v31 = 1 << (33 - (v29 ^ 0x1F));
    if ( v31 < 64 )
      v31 = 64;
    if ( (_DWORD)v6 == v31 )
    {
      *(_QWORD *)(a1 + 216) = 0;
      v37 = (_QWORD *)v30;
      v38 = (_QWORD *)(v30 + 8 * v6);
      do
      {
        if ( v37 )
          *v37 = -4096;
        ++v37;
      }
      while ( v38 != v37 );
      v39 = a2 + 40;
      v40 = (__int64 *)a2[41];
      if ( v40 == a2 + 40 )
      {
        v9 = 0;
LABEL_24:
        if ( *(_DWORD *)(a1 + 216) )
        {
          v24 = *(__int64 **)(a1 + 208);
          v25 = &v24[*(unsigned int *)(a1 + 224)];
          if ( v24 != v25 )
          {
            while ( *v24 == -8192 || *v24 == -4096 )
            {
              if ( ++v24 == v25 )
                return v9;
            }
LABEL_51:
            if ( v24 != v25 )
            {
              v26 = a2[4];
              v27 = *(unsigned int *)(*(_QWORD *)(*v24 + 32) + 8LL);
              if ( (int)v27 < 0 )
                v28 = *(_QWORD *)(*(_QWORD *)(v26 + 56) + 16 * (v27 & 0x7FFFFFFF) + 8);
              else
                v28 = *(_QWORD *)(*(_QWORD *)(v26 + 304) + 8 * v27);
              while ( 1 )
              {
                if ( !v28 )
                {
                  sub_2E88E20(*v24);
                  break;
                }
                if ( (*(_BYTE *)(v28 + 3) & 0x10) == 0 && (*(_BYTE *)(v28 + 4) & 8) == 0 )
                  break;
                v28 = *(_QWORD *)(v28 + 32);
              }
              while ( ++v24 != v25 )
              {
                if ( *v24 != -8192 && *v24 != -4096 )
                  goto LABEL_51;
              }
            }
          }
        }
        return v9;
      }
      goto LABEL_11;
    }
  }
  else
  {
    v30 = *(_QWORD *)(a1 + 208);
    v31 = 64;
  }
  sub_C7D6A0(v30, 8 * v6, 8);
  v32 = ((((((((4 * v31 / 3u + 1) | ((unsigned __int64)(4 * v31 / 3u + 1) >> 1)) >> 2)
           | (4 * v31 / 3u + 1)
           | ((unsigned __int64)(4 * v31 / 3u + 1) >> 1)) >> 4)
         | (((4 * v31 / 3u + 1) | ((unsigned __int64)(4 * v31 / 3u + 1) >> 1)) >> 2)
         | (4 * v31 / 3u + 1)
         | ((unsigned __int64)(4 * v31 / 3u + 1) >> 1)) >> 8)
       | (((((4 * v31 / 3u + 1) | ((unsigned __int64)(4 * v31 / 3u + 1) >> 1)) >> 2)
         | (4 * v31 / 3u + 1)
         | ((unsigned __int64)(4 * v31 / 3u + 1) >> 1)) >> 4)
       | (((4 * v31 / 3u + 1) | ((unsigned __int64)(4 * v31 / 3u + 1) >> 1)) >> 2)
       | (4 * v31 / 3u + 1)
       | ((unsigned __int64)(4 * v31 / 3u + 1) >> 1)) >> 16;
  v33 = (v32
       | (((((((4 * v31 / 3u + 1) | ((unsigned __int64)(4 * v31 / 3u + 1) >> 1)) >> 2)
           | (4 * v31 / 3u + 1)
           | ((unsigned __int64)(4 * v31 / 3u + 1) >> 1)) >> 4)
         | (((4 * v31 / 3u + 1) | ((unsigned __int64)(4 * v31 / 3u + 1) >> 1)) >> 2)
         | (4 * v31 / 3u + 1)
         | ((unsigned __int64)(4 * v31 / 3u + 1) >> 1)) >> 8)
       | (((((4 * v31 / 3u + 1) | ((unsigned __int64)(4 * v31 / 3u + 1) >> 1)) >> 2)
         | (4 * v31 / 3u + 1)
         | ((unsigned __int64)(4 * v31 / 3u + 1) >> 1)) >> 4)
       | (((4 * v31 / 3u + 1) | ((unsigned __int64)(4 * v31 / 3u + 1) >> 1)) >> 2)
       | (4 * v31 / 3u + 1)
       | ((unsigned __int64)(4 * v31 / 3u + 1) >> 1))
      + 1;
  *(_DWORD *)(a1 + 224) = v33;
  v34 = (_QWORD *)sub_C7D670(8 * v33, 8);
  v35 = *(unsigned int *)(a1 + 224);
  *(_QWORD *)(a1 + 216) = 0;
  *(_QWORD *)(a1 + 208) = v34;
  v36 = &v34[v35];
  if ( v34 == v36 )
    goto LABEL_10;
  do
  {
    if ( v34 )
      *v34 = -4096;
    ++v34;
  }
  while ( v36 != v34 );
  v39 = a2 + 40;
  v40 = (__int64 *)a2[41];
  if ( v40 != a2 + 40 )
  {
LABEL_11:
    v9 = 0;
    while ( 1 )
    {
      v10 = v40[7];
      v11 = v40 + 6;
      if ( (__int64 *)v10 != v40 + 6 )
        break;
LABEL_23:
      v40 = (__int64 *)v40[1];
      if ( v40 == v39 )
        goto LABEL_24;
    }
    while ( 1 )
    {
      v12 = *(_QWORD *)(v10 + 16);
      v13 = *(_QWORD **)(*(_QWORD *)(v10 + 24) + 32LL);
      v14 = v13[2];
      v15 = *(__int64 (**)(void))(*(_QWORD *)v14 + 128LL);
      if ( (char *)v15 == (char *)sub_30594F0 )
      {
        v16 = v14 + 376;
      }
      else
      {
        v42 = *(_QWORD *)(v10 + 16);
        v44 = *(_QWORD **)(*(_QWORD *)(v10 + 24) + 32LL);
        v21 = v15();
        v12 = v42;
        v13 = v44;
        v16 = v21;
      }
      v17 = *(_QWORD *)(v12 + 32);
      if ( (v17 & 0x40) != 0 )
        break;
      if ( (v17 & 0x180) != 0 )
      {
        v9 = sub_36F8260(a1, *(_QWORD *)(v10 + 32) + 40LL * (unsigned int)(1 << (((v17 >> 7) & 3) - 1)), v13);
        if ( (_BYTE)v9 )
        {
          v22 = sub_36F6540(*(unsigned __int16 *)(v10 + 68));
LABEL_39:
          sub_2E88D70(v10, (unsigned __int16 *)(*(_QWORD *)(v16 + 8) - 40LL * v22));
          goto LABEL_21;
        }
        goto LABEL_35;
      }
      if ( (v17 & 0x200) != 0 )
      {
        v9 = sub_36F8260(a1, *(_QWORD *)(v10 + 32), v13);
        if ( (_BYTE)v9 )
        {
          v22 = sub_36F6940(*(unsigned __int16 *)(v10 + 68));
          goto LABEL_39;
        }
        goto LABEL_35;
      }
      if ( (v17 & 0x400) != 0 )
      {
        v9 = sub_36F8260(a1, *(_QWORD *)(v10 + 32) + 40LL, v13);
        if ( (_BYTE)v9 )
        {
          v22 = sub_36F7480(*(unsigned __int16 *)(v10 + 68));
          goto LABEL_39;
        }
        goto LABEL_35;
      }
LABEL_21:
      if ( (*(_BYTE *)v10 & 4) != 0 )
      {
        v10 = *(_QWORD *)(v10 + 8);
        if ( v11 == (__int64 *)v10 )
          goto LABEL_23;
      }
      else
      {
        while ( (*(_BYTE *)(v10 + 44) & 8) != 0 )
          v10 = *(_QWORD *)(v10 + 8);
        v10 = *(_QWORD *)(v10 + 8);
        if ( v11 == (__int64 *)v10 )
          goto LABEL_23;
      }
    }
    v41 = v12;
    v43 = v13;
    v18 = sub_36F8260(a1, *(_QWORD *)(v10 + 32) + 160LL, v13);
    v19 = v43;
    v20 = v41;
    if ( v18 )
    {
      v23 = sub_36F6E50(*(unsigned __int16 *)(v10 + 68));
      sub_2E88D70(v10, (unsigned __int16 *)(*(_QWORD *)(v16 + 8) - 40LL * v23));
      v20 = v41;
      v19 = v43;
    }
    if ( (*(_BYTE *)(v20 + 33) & 8) == 0 )
    {
      v9 = sub_36F8260(a1, *(_QWORD *)(v10 + 32) + 200LL, v19);
      if ( (_BYTE)v9 )
      {
        v22 = sub_36F75C0(*(unsigned __int16 *)(v10 + 68));
        goto LABEL_39;
      }
    }
LABEL_35:
    v9 = 1;
    goto LABEL_21;
  }
  return 0;
}
