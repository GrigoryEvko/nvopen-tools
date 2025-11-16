// Function: sub_B1CE50
// Address: 0xb1ce50
//
__int64 __fastcall sub_B1CE50(__int64 a1, __int64 a2, __int64 a3)
{
  _QWORD *v4; // rsi
  __int64 v6; // rbx
  __int64 v7; // r14
  __int64 v8; // rcx
  __int64 v9; // rax
  int v10; // r15d
  __int64 v11; // rdx
  int v12; // ecx
  char v13; // dl
  __int64 v14; // rax
  __int64 v15; // rsi
  int v16; // ecx
  unsigned int v17; // eax
  __int64 *v18; // r15
  __int64 v19; // rdi
  __int64 v20; // rax
  __int64 *v21; // rbx
  __int64 *v22; // r13
  __int64 v23; // rsi
  __int64 v25; // rax
  __int64 v26; // rdx
  __int64 v27; // rbx
  __int64 v28; // r13
  __int64 v29; // r14
  int v30; // r14d
  __int64 v31; // r15
  int v32; // r8d

  v4 = (_QWORD *)(a1 + 16);
  if ( a3 )
  {
    v6 = *(_QWORD *)(a2 + 16);
    v7 = *(_QWORD *)(a3 + 8);
    if ( v6 )
    {
      while ( (unsigned __int8)(**(_BYTE **)(v6 + 24) - 30) > 0xAu )
      {
        v6 = *(_QWORD *)(v6 + 8);
        if ( !v6 )
          goto LABEL_29;
      }
      *(_QWORD *)a1 = v4;
      v8 = 0;
      *(_QWORD *)(a1 + 8) = 0x800000000LL;
      v9 = v6;
      while ( 1 )
      {
        v9 = *(_QWORD *)(v9 + 8);
        if ( !v9 )
          break;
        while ( (unsigned __int8)(**(_BYTE **)(v9 + 24) - 30) <= 0xAu )
        {
          v9 = *(_QWORD *)(v9 + 8);
          ++v8;
          if ( !v9 )
            goto LABEL_8;
        }
      }
LABEL_8:
      v10 = v8 + 1;
      if ( v8 + 1 > 8 )
      {
        sub_C8D5F0(a1, v4, v8 + 1, 8);
        v4 = (_QWORD *)(*(_QWORD *)a1 + 8LL * *(unsigned int *)(a1 + 8));
      }
      v11 = *(_QWORD *)(v6 + 24);
LABEL_13:
      if ( v4 )
        *v4 = *(_QWORD *)(v11 + 40);
      while ( 1 )
      {
        v6 = *(_QWORD *)(v6 + 8);
        if ( !v6 )
          break;
        v11 = *(_QWORD *)(v6 + 24);
        if ( (unsigned __int8)(*(_BYTE *)v11 - 30) <= 0xAu )
        {
          ++v4;
          goto LABEL_13;
        }
      }
      v12 = v10 + *(_DWORD *)(a1 + 8);
    }
    else
    {
LABEL_29:
      *(_QWORD *)a1 = v4;
      v12 = 0;
      *(_DWORD *)(a1 + 12) = 8;
    }
    *(_DWORD *)(a1 + 8) = v12;
    sub_B1C8F0(a1);
    v13 = *(_BYTE *)(v7 + 8) & 1;
    if ( v13 )
    {
      v15 = v7 + 16;
      v16 = 3;
    }
    else
    {
      v14 = *(unsigned int *)(v7 + 24);
      v15 = *(_QWORD *)(v7 + 16);
      if ( !(_DWORD)v14 )
        goto LABEL_46;
      v16 = v14 - 1;
    }
    v17 = v16 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v18 = (__int64 *)(v15 + 72LL * v17);
    v19 = *v18;
    if ( a2 == *v18 )
    {
LABEL_21:
      v20 = 288;
      if ( !v13 )
        v20 = 72LL * *(unsigned int *)(v7 + 24);
      if ( v18 != (__int64 *)(v15 + v20) )
      {
        v21 = (__int64 *)v18[1];
        v22 = &v21[*((unsigned int *)v18 + 4)];
        while ( v22 != v21 )
        {
          v23 = *v21++;
          sub_B1CA60(a1, v23);
        }
        sub_B1CB00(a1, (__int64)(v18 + 5));
      }
      return a1;
    }
    v32 = 1;
    while ( v19 != -4096 )
    {
      v17 = v16 & (v32 + v17);
      v18 = (__int64 *)(v15 + 72LL * v17);
      v19 = *v18;
      if ( a2 == *v18 )
        goto LABEL_21;
      ++v32;
    }
    if ( v13 )
    {
      v31 = 288;
      goto LABEL_47;
    }
    v14 = *(unsigned int *)(v7 + 24);
LABEL_46:
    v31 = 72 * v14;
LABEL_47:
    v18 = (__int64 *)(v15 + v31);
    goto LABEL_21;
  }
  v25 = sub_B18F80(a2);
  *(_QWORD *)a1 = v4;
  v27 = v25;
  *(_QWORD *)(a1 + 8) = 0x800000000LL;
  if ( v25 == v26 )
  {
    v30 = 0;
  }
  else
  {
    v28 = v25;
    v29 = 0;
    do
    {
      do
        v28 = *(_QWORD *)(v28 + 8);
      while ( v28 && (unsigned __int8)(**(_BYTE **)(v28 + 24) - 30) > 0xAu );
      ++v29;
    }
    while ( v26 != v28 );
    if ( v29 > 8 )
    {
      sub_C8D5F0(a1, v4, v29, 8);
      v4 = (_QWORD *)(*(_QWORD *)a1 + 8LL * *(unsigned int *)(a1 + 8));
    }
    do
    {
      if ( v4 )
        *v4 = *(_QWORD *)(*(_QWORD *)(v27 + 24) + 40LL);
      do
        v27 = *(_QWORD *)(v27 + 8);
      while ( v27 && (unsigned __int8)(**(_BYTE **)(v27 + 24) - 30) > 0xAu );
      ++v4;
    }
    while ( v28 != v27 );
    v30 = *(_DWORD *)(a1 + 8) + v29;
  }
  *(_DWORD *)(a1 + 8) = v30;
  sub_B1C8F0(a1);
  return a1;
}
