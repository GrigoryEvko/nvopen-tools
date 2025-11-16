// Function: sub_37F56A0
// Address: 0x37f56a0
//
__int64 __fastcall sub_37F56A0(__int64 a1, __int64 a2, unsigned int a3)
{
  int v5; // ecx
  __int64 v6; // rdi
  int v7; // r8d
  unsigned int v8; // esi
  __int64 *v9; // rcx
  __int64 v10; // r9
  int v11; // esi
  unsigned int v12; // r8d
  unsigned int v13; // ecx
  __int64 v14; // rdi
  __int64 v15; // rdx
  __int64 v16; // r9
  int v17; // r12d
  unsigned int i; // eax
  __int64 v19; // r10
  unsigned int v20; // eax
  __int64 v21; // r8
  __int64 v22; // rbx
  __int64 v23; // rax
  __int64 v24; // rdx
  unsigned int v25; // r8d
  unsigned int v26; // r11d
  __int16 *v27; // r9
  _QWORD *v28; // rax
  int *v29; // rdx
  unsigned __int64 v30; // rdi
  __int64 v31; // rdi
  int v32; // eax
  int v33; // edx
  int v35; // ecx
  int *v36; // rax
  int *v37; // rdx
  signed int v38; // edi
  int v39; // r11d

  v5 = *(_DWORD *)(a1 + 488);
  v6 = *(_QWORD *)(a1 + 472);
  if ( v5 )
  {
    v7 = v5 - 1;
    v8 = (v5 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v9 = (__int64 *)(v6 + 16LL * v8);
    v10 = *v9;
    if ( a2 == *v9 )
    {
LABEL_3:
      v11 = *((_DWORD *)v9 + 2);
      goto LABEL_4;
    }
    v35 = 1;
    while ( v10 != -4096 )
    {
      v39 = v35 + 1;
      v8 = v7 & (v35 + v8);
      v9 = (__int64 *)(v6 + 16LL * v8);
      v10 = *v9;
      if ( a2 == *v9 )
        goto LABEL_3;
      v35 = v39;
    }
  }
  v11 = 0;
LABEL_4:
  v12 = a3 - 0x40000000;
  v13 = *(_DWORD *)(a1 + 640);
  v14 = *(unsigned int *)(*(_QWORD *)(a2 + 24) + 24LL);
  if ( a3 - 0x40000000 > 0x3FFFFFFF )
  {
    v21 = *(_QWORD *)(a1 + 208);
    v22 = 24 * v14;
    v23 = *(_QWORD *)(v21 + 8) + 24LL * a3;
    v24 = *(_QWORD *)(v21 + 56);
    v25 = *(_DWORD *)(a1 + 640);
    LODWORD(v23) = *(_DWORD *)(v23 + 16);
    v26 = v23 & 0xFFF;
    v27 = (__int16 *)(v24 + 2LL * ((unsigned int)v23 >> 12));
    do
    {
      if ( !v27 )
        break;
      v28 = (_QWORD *)(v22 + *(_QWORD *)(a1 + 496));
      if ( v28[1] != *v28 )
      {
        v29 = (int *)(*v28 + 8LL * v26);
        v30 = *(_QWORD *)v29 & 0xFFFFFFFFFFFFFFFELL;
        if ( v30 )
        {
          if ( (*(_QWORD *)v29 & 1) != 0 )
          {
            v29 = *(int **)v30;
            v31 = *(_QWORD *)v30 + 8LL * *(unsigned int *)(v30 + 8);
          }
          else
          {
            v31 = (__int64)(v29 + 2);
          }
          if ( (int *)v31 != v29 )
          {
            v32 = v13;
            while ( 1 )
            {
              v13 = v32;
              v32 = *v29 >> 2;
              if ( v11 <= v32 )
                break;
              v29 += 2;
              if ( (int *)v31 == v29 )
              {
                v13 = v32;
                break;
              }
            }
          }
        }
      }
      v33 = *v27;
      if ( (int)v25 < (int)v13 )
        v25 = v13;
      ++v27;
      v26 += v33;
    }
    while ( (_WORD)v33 );
    return v25;
  }
  v15 = *(unsigned int *)(a1 + 632);
  v16 = *(_QWORD *)(a1 + 616);
  if ( !(_DWORD)v15 )
    return v13;
  v17 = 1;
  for ( i = (v15 - 1)
          & (((0xBF58476D1CE4E5B9LL * ((37 * v12) | ((unsigned __int64)(unsigned int)(37 * v14) << 32))) >> 31)
           ^ (756364221 * v12)); ; i = (v15 - 1) & v20 )
  {
    v19 = v16 + 72LL * i;
    if ( (_DWORD)v14 == *(_DWORD *)v19 && v12 == *(_DWORD *)(v19 + 4) )
      break;
    if ( *(_DWORD *)v19 == -1 && *(_DWORD *)(v19 + 4) == 0x7FFFFFFF )
      return v13;
    v20 = v17 + i;
    ++v17;
  }
  if ( v19 == v16 + 72 * v15 )
    return v13;
  v36 = *(int **)(v19 + 8);
  v25 = v13;
  v37 = &v36[*(unsigned int *)(v19 + 16)];
  if ( v37 == v36 )
    return v25;
  do
  {
    v38 = v25;
    v25 = *v36;
    if ( v11 <= *v36 )
    {
      if ( (int)v13 >= v38 )
        return v13;
      return (unsigned int)v38;
    }
    ++v36;
  }
  while ( v37 != v36 );
  if ( (int)v13 >= (int)v25 )
    return v13;
  return v25;
}
