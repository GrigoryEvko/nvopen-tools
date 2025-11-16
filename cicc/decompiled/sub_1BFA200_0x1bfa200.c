// Function: sub_1BFA200
// Address: 0x1bfa200
//
unsigned __int64 __fastcall sub_1BFA200(unsigned __int64 a1, __int64 a2, _BYTE *a3)
{
  unsigned int v6; // esi
  __int64 v7; // r8
  __int64 v8; // rdi
  int v9; // r10d
  unsigned __int64 *v10; // r9
  unsigned int v11; // ecx
  unsigned __int64 *v12; // rdx
  unsigned __int64 result; // rax
  int v14; // eax
  int v15; // edx
  __int64 v16; // rcx
  __int64 v17; // r14
  __int64 v18; // r15
  unsigned __int64 i; // rax
  __int64 *v20; // rax
  __int64 v21; // rdx
  __int64 v22; // r14
  __int64 v23; // r15
  unsigned __int64 j; // rax
  int v25; // eax
  __int64 v26; // rdi
  unsigned int v27; // eax
  unsigned __int64 v28; // rsi
  int v29; // r10d
  int v30; // eax
  int v31; // eax
  __int64 v32; // rsi
  unsigned int v33; // r14d
  unsigned __int64 *v34; // rdi

  v6 = *(_DWORD *)(a2 + 24);
  if ( v6 )
  {
    v7 = v6 - 1;
    v8 = *(_QWORD *)(a2 + 8);
    v9 = 1;
    v10 = 0;
    v11 = v7 & (((unsigned int)a1 >> 9) ^ ((unsigned int)a1 >> 4));
    v12 = (unsigned __int64 *)(v8 + 8LL * v11);
    result = *v12;
    if ( *v12 == a1 )
      return result;
    while ( result != -8 )
    {
      if ( result != -16 || v10 )
        v12 = v10;
      v11 = v7 & (v9 + v11);
      result = *(_QWORD *)(v8 + 8LL * v11);
      if ( result == a1 )
        return result;
      ++v9;
      v10 = v12;
      v12 = (unsigned __int64 *)(v8 + 8LL * v11);
    }
    v14 = *(_DWORD *)(a2 + 16);
    if ( !v10 )
      v10 = v12;
    ++*(_QWORD *)a2;
    v15 = v14 + 1;
    if ( 4 * (v14 + 1) < 3 * v6 )
    {
      v16 = v6 >> 3;
      if ( v6 - *(_DWORD *)(a2 + 20) - v15 > (unsigned int)v16 )
        goto LABEL_13;
      sub_1353F00(a2, v6);
      v30 = *(_DWORD *)(a2 + 24);
      if ( v30 )
      {
        v31 = v30 - 1;
        v32 = *(_QWORD *)(a2 + 8);
        v7 = 1;
        v33 = v31 & (((unsigned int)a1 >> 9) ^ ((unsigned int)a1 >> 4));
        v10 = (unsigned __int64 *)(v32 + 8LL * v33);
        v15 = *(_DWORD *)(a2 + 16) + 1;
        v34 = 0;
        v16 = *v10;
        if ( *v10 != a1 )
        {
          while ( v16 != -8 )
          {
            if ( !v34 && v16 == -16 )
              v34 = v10;
            v33 = v31 & (v7 + v33);
            v10 = (unsigned __int64 *)(v32 + 8LL * v33);
            v16 = *v10;
            if ( *v10 == a1 )
              goto LABEL_13;
            v7 = (unsigned int)(v7 + 1);
          }
          if ( v34 )
            v10 = v34;
        }
        goto LABEL_13;
      }
LABEL_65:
      ++*(_DWORD *)(a2 + 16);
      BUG();
    }
  }
  else
  {
    ++*(_QWORD *)a2;
  }
  sub_1353F00(a2, 2 * v6);
  v25 = *(_DWORD *)(a2 + 24);
  if ( !v25 )
    goto LABEL_65;
  v16 = (unsigned int)(v25 - 1);
  v26 = *(_QWORD *)(a2 + 8);
  v27 = v16 & (((unsigned int)a1 >> 9) ^ ((unsigned int)a1 >> 4));
  v10 = (unsigned __int64 *)(v26 + 8LL * v27);
  v28 = *v10;
  v15 = *(_DWORD *)(a2 + 16) + 1;
  if ( *v10 != a1 )
  {
    v29 = 1;
    v7 = 0;
    while ( v28 != -8 )
    {
      if ( v28 == -16 && !v7 )
        v7 = (__int64)v10;
      v27 = v16 & (v29 + v27);
      v10 = (unsigned __int64 *)(v26 + 8LL * v27);
      v28 = *v10;
      if ( *v10 == a1 )
        goto LABEL_13;
      ++v29;
    }
    if ( v7 )
      v10 = (unsigned __int64 *)v7;
  }
LABEL_13:
  *(_DWORD *)(a2 + 16) = v15;
  if ( *v10 != -8 )
    --*(_DWORD *)(a2 + 20);
  *v10 = a1;
  result = *(unsigned __int8 *)(a1 + 16);
  if ( (unsigned __int8)result <= 0x17u )
    return result;
  if ( (_BYTE)result == 62 )
  {
    if ( (*(_BYTE *)(a1 + 23) & 0x40) != 0 )
      v20 = *(__int64 **)(a1 - 8);
    else
      v20 = (__int64 *)(a1 - 24LL * (*(_DWORD *)(a1 + 20) & 0xFFFFFFF));
    v21 = *v20;
    result = *(unsigned __int8 *)(*v20 + 16);
    v16 = (unsigned int)(result - 35);
    if ( (unsigned __int8)(result - 35) > 0x11u )
      return result;
    if ( (unsigned __int8)result <= 0x2Fu )
    {
      v16 = 0x80A800000000LL;
      if ( _bittest64(&v16, result) )
      {
        if ( (*(_BYTE *)(v21 + 17) & 4) == 0 )
          *a3 = 0;
        return result;
      }
    }
  }
  else if ( (_BYTE)result == 77 )
  {
    result = *(_DWORD *)(a1 + 20) & 0xFFFFFFF;
    if ( (*(_DWORD *)(a1 + 20) & 0xFFFFFFF) != 0 )
    {
      v17 = 0;
      v18 = 24LL * (unsigned int)result;
      if ( (*(_BYTE *)(a1 + 23) & 0x40) == 0 )
        goto LABEL_24;
LABEL_20:
      for ( i = *(_QWORD *)(a1 - 8); ; i = a1 - 24LL * (*(_DWORD *)(a1 + 20) & 0xFFFFFFF) )
      {
        result = sub_1BFA200(*(_QWORD *)(i + v17), a2, a3, v16, v7, v10);
        if ( !*a3 )
          break;
        v17 += 24;
        if ( v17 == v18 )
          break;
        if ( (*(_BYTE *)(a1 + 23) & 0x40) != 0 )
          goto LABEL_20;
LABEL_24:
        ;
      }
    }
    return result;
  }
  result = *(_DWORD *)(a1 + 20) & 0xFFFFFFF;
  if ( (*(_DWORD *)(a1 + 20) & 0xFFFFFFF) != 0 )
  {
    v22 = 0;
    v23 = 24LL * (unsigned int)result;
    if ( (*(_BYTE *)(a1 + 23) & 0x40) == 0 )
      goto LABEL_35;
LABEL_31:
    for ( j = *(_QWORD *)(a1 - 8); ; j = a1 - 24LL * (*(_DWORD *)(a1 + 20) & 0xFFFFFFF) )
    {
      result = sub_1BFA200(*(_QWORD *)(j + v22), a2, a3, v16, v7, v10);
      if ( !*a3 )
        break;
      v22 += 24;
      if ( v23 == v22 )
        break;
      if ( (*(_BYTE *)(a1 + 23) & 0x40) != 0 )
        goto LABEL_31;
LABEL_35:
      ;
    }
  }
  return result;
}
