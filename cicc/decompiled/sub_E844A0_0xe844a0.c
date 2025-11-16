// Function: sub_E844A0
// Address: 0xe844a0
//
__int64 __fastcall sub_E844A0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 result; // rax
  __int64 v6; // rsi
  __int64 v7; // r8
  int v8; // r11d
  __int64 v9; // rdi
  __int64 v10; // rdx
  __int64 v11; // rcx
  __int64 v12; // r9
  __int64 v13; // r13
  unsigned int v14; // esi
  __int64 v15; // rcx
  __int64 v16; // r8
  int v17; // r11d
  unsigned int v18; // edx
  __int64 *v19; // rax
  __int64 v20; // r10
  int v21; // eax
  int v22; // eax
  __int64 v23; // rdi
  int v24; // r10d
  __int64 v25; // r9
  int v26; // eax
  __int64 v27; // rdi
  unsigned int v28; // r13d
  int v29; // r9d
  int v30; // eax
  int v31; // edx
  int v32; // ecx
  int v33; // ecx
  int v34; // esi
  unsigned int j; // edx
  __int64 v36; // rax
  int v37; // edx
  int v38; // edx
  __int64 v39; // rdi
  int v40; // ecx
  unsigned int i; // r15d
  __int64 *v42; // rax
  __int64 v43; // rsi
  unsigned int v44; // r15d
  unsigned int v45; // edx

  result = sub_E8CEC0(a1, a2, a3);
  if ( !*(_BYTE *)(a1 + 440) )
    return result;
  v6 = *(unsigned int *)(a1 + 472);
  if ( (_DWORD)v6 )
  {
    v7 = (unsigned int)(v6 - 1);
    v8 = 1;
    v9 = *(_QWORD *)(a1 + 456);
    v10 = 0;
    v11 = (unsigned int)v7 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    result = v9 + 16 * v11;
    v12 = *(_QWORD *)result;
    if ( *(_QWORD *)result == a2 )
    {
LABEL_5:
      if ( *(_BYTE *)(result + 8) )
        return result;
      v13 = *(_QWORD *)(a2 + 16);
      if ( v13 )
        return result;
LABEL_7:
      *(_QWORD *)(a2 + 16) = sub_E6C350(*(_QWORD *)(a1 + 8), v6, v10, v11, v7);
      v14 = *(_DWORD *)(a1 + 472);
      if ( v14 )
      {
        v15 = *(_QWORD *)(a1 + 456);
        v16 = 0;
        v17 = 1;
        v18 = (v14 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
        v19 = (__int64 *)(v15 + 16LL * v18);
        v20 = *v19;
        if ( *v19 == a2 )
        {
LABEL_9:
          result = (__int64)(v19 + 1);
LABEL_10:
          *(_BYTE *)result = 1;
          return result;
        }
        while ( v20 != -4096 )
        {
          if ( !v16 && v20 == -8192 )
            v16 = (__int64)v19;
          v18 = (v14 - 1) & (v17 + v18);
          v19 = (__int64 *)(v15 + 16LL * v18);
          v20 = *v19;
          if ( *v19 == a2 )
            goto LABEL_9;
          ++v17;
        }
        if ( !v16 )
          v16 = (__int64)v19;
        v30 = *(_DWORD *)(a1 + 464);
        ++*(_QWORD *)(a1 + 448);
        v31 = v30 + 1;
        if ( 4 * (v30 + 1) < 3 * v14 )
        {
          if ( v14 - *(_DWORD *)(a1 + 468) - v31 > v14 >> 3 )
          {
LABEL_49:
            *(_DWORD *)(a1 + 464) = v31;
            if ( *(_QWORD *)v16 != -4096 )
              --*(_DWORD *)(a1 + 468);
            *(_QWORD *)v16 = a2;
            result = v16 + 8;
            *(_BYTE *)(v16 + 8) = 0;
            goto LABEL_10;
          }
          sub_E842C0(a1 + 448, v14);
          v37 = *(_DWORD *)(a1 + 472);
          if ( v37 )
          {
            v38 = v37 - 1;
            v16 = 0;
            v40 = 1;
            for ( i = v38 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4)); ; i = v38 & v44 )
            {
              v39 = *(_QWORD *)(a1 + 456);
              v42 = (__int64 *)(v39 + 16LL * i);
              v43 = *v42;
              if ( *v42 == a2 )
              {
                v16 = v39 + 16LL * i;
                v31 = *(_DWORD *)(a1 + 464) + 1;
                goto LABEL_49;
              }
              if ( v43 == -4096 )
                break;
              if ( v43 != -8192 || v16 )
                v42 = (__int64 *)v16;
              v44 = v40 + i;
              v16 = (__int64)v42;
              ++v40;
            }
            if ( !v16 )
              v16 = v39 + 16LL * i;
            v31 = *(_DWORD *)(a1 + 464) + 1;
            goto LABEL_49;
          }
LABEL_87:
          ++*(_DWORD *)(a1 + 464);
          BUG();
        }
      }
      else
      {
        ++*(_QWORD *)(a1 + 448);
      }
      sub_E842C0(a1 + 448, 2 * v14);
      v32 = *(_DWORD *)(a1 + 472);
      if ( v32 )
      {
        v33 = v32 - 1;
        v34 = 1;
        for ( j = v33 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4)); ; j = v33 & v45 )
        {
          v16 = *(_QWORD *)(a1 + 456) + 16LL * j;
          v36 = *(_QWORD *)v16;
          if ( *(_QWORD *)v16 == a2 )
          {
            v31 = *(_DWORD *)(a1 + 464) + 1;
            goto LABEL_49;
          }
          if ( v36 == -4096 )
            break;
          if ( v13 || v36 != -8192 )
            v16 = v13;
          v45 = v34 + j;
          v13 = v16;
          ++v34;
        }
        if ( v13 )
          v16 = v13;
        v31 = *(_DWORD *)(a1 + 464) + 1;
        goto LABEL_49;
      }
      goto LABEL_87;
    }
    while ( v12 != -4096 )
    {
      if ( !v10 && v12 == -8192 )
        v10 = result;
      v11 = (unsigned int)v7 & (v8 + (_DWORD)v11);
      result = v9 + 16LL * (unsigned int)v11;
      v12 = *(_QWORD *)result;
      if ( *(_QWORD *)result == a2 )
        goto LABEL_5;
      ++v8;
    }
    if ( !v10 )
      v10 = result;
    v21 = *(_DWORD *)(a1 + 464);
    ++*(_QWORD *)(a1 + 448);
    v11 = (unsigned int)(v21 + 1);
    if ( 4 * (int)v11 < (unsigned int)(3 * v6) )
    {
      result = (unsigned int)(v6 - *(_DWORD *)(a1 + 468) - v11);
      if ( (unsigned int)result > (unsigned int)v6 >> 3 )
        goto LABEL_21;
      sub_E842C0(a1 + 448, v6);
      v26 = *(_DWORD *)(a1 + 472);
      if ( v26 )
      {
        result = (unsigned int)(v26 - 1);
        v27 = *(_QWORD *)(a1 + 456);
        v7 = 0;
        v28 = result & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
        v29 = 1;
        v11 = (unsigned int)(*(_DWORD *)(a1 + 464) + 1);
        v10 = v27 + 16LL * v28;
        v6 = *(_QWORD *)v10;
        if ( *(_QWORD *)v10 != a2 )
        {
          while ( v6 != -4096 )
          {
            if ( v6 == -8192 && !v7 )
              v7 = v10;
            v28 = result & (v29 + v28);
            v10 = v27 + 16LL * v28;
            v6 = *(_QWORD *)v10;
            if ( *(_QWORD *)v10 == a2 )
              goto LABEL_21;
            ++v29;
          }
          if ( v7 )
            v10 = v7;
        }
        goto LABEL_21;
      }
LABEL_88:
      ++*(_DWORD *)(a1 + 464);
      BUG();
    }
  }
  else
  {
    ++*(_QWORD *)(a1 + 448);
  }
  sub_E842C0(a1 + 448, 2 * v6);
  v22 = *(_DWORD *)(a1 + 472);
  if ( !v22 )
    goto LABEL_88;
  v6 = (unsigned int)(v22 - 1);
  v7 = *(_QWORD *)(a1 + 456);
  result = (unsigned int)v6 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v11 = (unsigned int)(*(_DWORD *)(a1 + 464) + 1);
  v10 = v7 + 16 * result;
  v23 = *(_QWORD *)v10;
  if ( *(_QWORD *)v10 != a2 )
  {
    v24 = 1;
    v25 = 0;
    while ( v23 != -4096 )
    {
      if ( !v25 && v23 == -8192 )
        v25 = v10;
      result = (unsigned int)v6 & (v24 + (_DWORD)result);
      v10 = v7 + 16LL * (unsigned int)result;
      v23 = *(_QWORD *)v10;
      if ( *(_QWORD *)v10 == a2 )
        goto LABEL_21;
      ++v24;
    }
    if ( v25 )
      v10 = v25;
  }
LABEL_21:
  *(_DWORD *)(a1 + 464) = v11;
  if ( *(_QWORD *)v10 != -4096 )
    --*(_DWORD *)(a1 + 468);
  *(_QWORD *)v10 = a2;
  *(_BYTE *)(v10 + 8) = 0;
  v13 = *(_QWORD *)(a2 + 16);
  if ( !v13 )
    goto LABEL_7;
  return result;
}
