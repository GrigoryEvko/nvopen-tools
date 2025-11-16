// Function: sub_F7CE70
// Address: 0xf7ce70
//
__int64 __fastcall sub_F7CE70(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // rcx
  __int64 v4; // r9
  __int64 v5; // rsi
  int v6; // r8d
  __int64 v7; // r10
  int v8; // r8d
  unsigned int v9; // edx
  __int64 *v10; // rax
  __int64 v11; // rdi
  _QWORD *v12; // rdi
  unsigned int v13; // edx
  __int64 *v14; // rax
  __int64 v15; // rcx
  _QWORD *v16; // rcx
  _QWORD *v17; // rax
  _QWORD *v18; // rax
  __int64 result; // rax
  __int64 v20; // rax
  __int64 v21; // rdx
  __int64 i; // r11
  unsigned int v23; // ebx
  __int64 *v24; // rdx
  __int64 v25; // r12
  int v26; // edx
  __int64 v27; // rdx
  __int64 v28; // rax
  __int64 v29; // rdi
  unsigned int v30; // r11d
  __int64 *v31; // rdx
  __int64 v32; // rbx
  _QWORD *v33; // rdx
  int v34; // eax
  int v35; // eax
  int v36; // r13d
  int v37; // edx
  int v38; // r11d
  int v39; // r11d
  int v40; // r12d

  v3 = *(_QWORD *)(a2 + 40);
  v4 = a2;
  v5 = *(_QWORD *)(a3 + 40);
  if ( v3 == v5 )
    return 1;
  v6 = *(_DWORD *)(a1 + 24);
  v7 = *(_QWORD *)(a1 + 8);
  if ( !v6 )
    return 1;
  v8 = v6 - 1;
  v9 = v8 & (((unsigned int)v3 >> 9) ^ ((unsigned int)v3 >> 4));
  v10 = (__int64 *)(v7 + 16LL * v9);
  v11 = *v10;
  if ( v3 == *v10 )
  {
LABEL_4:
    v12 = (_QWORD *)v10[1];
  }
  else
  {
    v35 = 1;
    while ( v11 != -4096 )
    {
      v38 = v35 + 1;
      v9 = v8 & (v35 + v9);
      v10 = (__int64 *)(v7 + 16LL * v9);
      v11 = *v10;
      if ( v3 == *v10 )
        goto LABEL_4;
      v35 = v38;
    }
    v12 = 0;
  }
  v13 = v8 & (((unsigned int)v5 >> 9) ^ ((unsigned int)v5 >> 4));
  v14 = (__int64 *)(v7 + 16LL * v13);
  v15 = *v14;
  if ( v5 != *v14 )
  {
    v34 = 1;
    while ( v15 != -4096 )
    {
      v39 = v34 + 1;
      v13 = v8 & (v34 + v13);
      v14 = (__int64 *)(v7 + 16LL * v13);
      v15 = *v14;
      if ( v5 == *v14 )
        goto LABEL_6;
      v34 = v39;
    }
    if ( !v12 )
      return 1;
    v16 = 0;
    goto LABEL_32;
  }
LABEL_6:
  v16 = (_QWORD *)v14[1];
  if ( v12 == v16 )
    return 1;
  if ( !v16 )
  {
    result = 1;
    if ( !v12 )
      return result;
LABEL_32:
    result = 0;
    if ( *(_BYTE *)v4 == 84 )
      return result;
    v27 = 32LL * (*(_DWORD *)(v4 + 4) & 0x7FFFFFF);
    if ( (*(_BYTE *)(v4 + 7) & 0x40) != 0 )
    {
      v28 = *(_QWORD *)(v4 - 8);
      v4 = v28 + v27;
    }
    else
    {
      v28 = v4 - v27;
    }
    if ( v4 != v28 )
    {
      while ( **(_BYTE **)v28 > 0x1Cu )
      {
        v29 = *(_QWORD *)(*(_QWORD *)v28 + 40LL);
        if ( v5 != v29 )
        {
          v30 = v8 & (((unsigned int)v29 >> 9) ^ ((unsigned int)v29 >> 4));
          v31 = (__int64 *)(v7 + 16LL * v30);
          v32 = *v31;
          if ( v29 == *v31 )
          {
LABEL_39:
            v33 = (_QWORD *)v31[1];
          }
          else
          {
            v37 = 1;
            while ( v32 != -4096 )
            {
              v40 = v37 + 1;
              v30 = v8 & (v37 + v30);
              v31 = (__int64 *)(v7 + 16LL * v30);
              v32 = *v31;
              if ( v29 == *v31 )
                goto LABEL_39;
              v37 = v40;
            }
            v33 = 0;
          }
          if ( v16 != v33 )
            break;
        }
        v28 += 32;
        if ( v4 == v28 )
          return 1;
      }
      return 0;
    }
    return 1;
  }
  if ( !v12 )
  {
    v20 = *(_QWORD *)(v4 + 16);
    if ( v20 )
      goto LABEL_26;
    return 1;
  }
  v17 = v12;
  do
  {
    v17 = (_QWORD *)*v17;
    if ( v17 == v16 )
      goto LABEL_12;
  }
  while ( v17 );
  v20 = *(_QWORD *)(v4 + 16);
  if ( !v20 )
  {
LABEL_12:
    if ( v12 != v16 )
    {
      v18 = v16;
      while ( 1 )
      {
        v18 = (_QWORD *)*v18;
        if ( v12 == v18 )
          break;
        if ( !v18 )
          goto LABEL_32;
      }
    }
    return 1;
  }
  v21 = *(_QWORD *)(v20 + 24);
  if ( *(_BYTE *)v21 == 84 )
    goto LABEL_27;
LABEL_21:
  for ( i = *(_QWORD *)(v21 + 40);
        ;
        i = *(_QWORD *)(*(_QWORD *)(v21 - 8)
                      + 32LL * *(unsigned int *)(v21 + 72)
                      + 8LL * (unsigned int)((v20 - *(_QWORD *)(v21 - 8)) >> 5)) )
  {
    if ( v5 != i )
    {
      v23 = v8 & (((unsigned int)i >> 9) ^ ((unsigned int)i >> 4));
      v24 = (__int64 *)(v7 + 16LL * v23);
      v25 = *v24;
      if ( *v24 != i )
      {
        v26 = 1;
        while ( v25 != -4096 )
        {
          v36 = v26 + 1;
          v23 = v8 & (v26 + v23);
          v24 = (__int64 *)(v7 + 16LL * v23);
          v25 = *v24;
          if ( *v24 == i )
            goto LABEL_24;
          v26 = v36;
        }
        return 0;
      }
LABEL_24:
      if ( v16 != (_QWORD *)v24[1] )
        return 0;
    }
    v20 = *(_QWORD *)(v20 + 8);
    if ( !v20 )
      break;
LABEL_26:
    v21 = *(_QWORD *)(v20 + 24);
    if ( *(_BYTE *)v21 != 84 )
      goto LABEL_21;
LABEL_27:
    ;
  }
  result = 1;
  if ( v12 )
    goto LABEL_12;
  return result;
}
