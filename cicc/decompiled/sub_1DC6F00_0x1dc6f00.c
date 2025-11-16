// Function: sub_1DC6F00
// Address: 0x1dc6f00
//
__int64 *__fastcall sub_1DC6F00(__int64 a1, __int64 a2)
{
  int v4; // eax
  __int64 v5; // rdx
  _QWORD *v6; // rax
  _QWORD *i; // rdx
  __int64 *result; // rax
  int v9; // r13d
  __int64 *v10; // rdx
  unsigned int v11; // esi
  int v12; // r15d
  __int64 v13; // r8
  unsigned int v14; // edi
  __int64 v15; // rcx
  int v16; // r11d
  __int64 *v17; // r10
  int v18; // edi
  int v19; // edi
  unsigned int v20; // ecx
  _QWORD *v21; // rdi
  unsigned int v22; // eax
  int v23; // eax
  unsigned __int64 v24; // rax
  unsigned __int64 v25; // rax
  int v26; // r14d
  __int64 v27; // r13
  _QWORD *v28; // rax
  __int64 v29; // rdx
  _QWORD *j; // rdx
  int v31; // eax
  int v32; // r9d
  __int64 v33; // r8
  unsigned int v34; // ecx
  __int64 v35; // rsi
  int v36; // r11d
  __int64 *v37; // r10
  int v38; // eax
  int v39; // esi
  __int64 v40; // r8
  __int64 *v41; // r9
  unsigned int v42; // r14d
  int v43; // r10d
  __int64 v44; // rcx
  _QWORD *v45; // rax
  __int64 *v46; // [rsp+8h] [rbp-38h]
  __int64 *v47; // [rsp+8h] [rbp-38h]

  v4 = *(_DWORD *)(a2 + 16);
  ++*(_QWORD *)a2;
  if ( !v4 )
  {
    if ( !*(_DWORD *)(a2 + 20) )
      goto LABEL_7;
    v5 = *(unsigned int *)(a2 + 24);
    if ( (unsigned int)v5 > 0x40 )
    {
      j___libc_free_0(*(_QWORD *)(a2 + 8));
      *(_QWORD *)(a2 + 8) = 0;
      *(_QWORD *)(a2 + 16) = 0;
      *(_DWORD *)(a2 + 24) = 0;
      goto LABEL_7;
    }
    goto LABEL_4;
  }
  v20 = 4 * v4;
  v5 = *(unsigned int *)(a2 + 24);
  if ( (unsigned int)(4 * v4) < 0x40 )
    v20 = 64;
  if ( (unsigned int)v5 <= v20 )
  {
LABEL_4:
    v6 = *(_QWORD **)(a2 + 8);
    for ( i = &v6[2 * v5]; i != v6; v6 += 2 )
      *v6 = -8;
    *(_QWORD *)(a2 + 16) = 0;
    goto LABEL_7;
  }
  v21 = *(_QWORD **)(a2 + 8);
  v22 = v4 - 1;
  if ( !v22 )
  {
    v27 = 2048;
    v26 = 128;
LABEL_36:
    j___libc_free_0(v21);
    *(_DWORD *)(a2 + 24) = v26;
    v28 = (_QWORD *)sub_22077B0(v27);
    v29 = *(unsigned int *)(a2 + 24);
    *(_QWORD *)(a2 + 16) = 0;
    *(_QWORD *)(a2 + 8) = v28;
    for ( j = &v28[2 * v29]; j != v28; v28 += 2 )
    {
      if ( v28 )
        *v28 = -8;
    }
    goto LABEL_7;
  }
  _BitScanReverse(&v22, v22);
  v23 = 1 << (33 - (v22 ^ 0x1F));
  if ( v23 < 64 )
    v23 = 64;
  if ( (_DWORD)v5 != v23 )
  {
    v24 = (4 * v23 / 3u + 1) | ((unsigned __int64)(4 * v23 / 3u + 1) >> 1);
    v25 = ((v24 | (v24 >> 2)) >> 4) | v24 | (v24 >> 2) | ((((v24 | (v24 >> 2)) >> 4) | v24 | (v24 >> 2)) >> 8);
    v26 = (v25 | (v25 >> 16)) + 1;
    v27 = 16 * ((v25 | (v25 >> 16)) + 1);
    goto LABEL_36;
  }
  *(_QWORD *)(a2 + 16) = 0;
  v45 = &v21[2 * (unsigned int)v5];
  do
  {
    if ( v21 )
      *v21 = -8;
    v21 += 2;
  }
  while ( v45 != v21 );
LABEL_7:
  result = *(__int64 **)(a1 + 24);
  v9 = 0;
  v10 = result + 3;
  if ( result + 3 != (__int64 *)a1 )
  {
    while ( 1 )
    {
      v11 = *(_DWORD *)(a2 + 24);
      v12 = v9++;
      if ( !v11 )
        break;
      v13 = *(_QWORD *)(a2 + 8);
      v14 = (v11 - 1) & (((unsigned int)a1 >> 9) ^ ((unsigned int)a1 >> 4));
      result = (__int64 *)(v13 + 16LL * v14);
      v15 = *result;
      if ( *result != a1 )
      {
        v16 = 1;
        v17 = 0;
        while ( v15 != -8 )
        {
          if ( v15 == -16 && !v17 )
            v17 = result;
          v14 = (v11 - 1) & (v16 + v14);
          result = (__int64 *)(v13 + 16LL * v14);
          v15 = *result;
          if ( *result == a1 )
            goto LABEL_12;
          ++v16;
        }
        v18 = *(_DWORD *)(a2 + 16);
        if ( v17 )
          result = v17;
        ++*(_QWORD *)a2;
        v19 = v18 + 1;
        if ( 4 * v19 < 3 * v11 )
        {
          if ( v11 - *(_DWORD *)(a2 + 20) - v19 <= v11 >> 3 )
          {
            v47 = v10;
            sub_1DC6D40(a2, v11);
            v38 = *(_DWORD *)(a2 + 24);
            if ( !v38 )
            {
LABEL_78:
              ++*(_DWORD *)(a2 + 16);
              BUG();
            }
            v39 = v38 - 1;
            v40 = *(_QWORD *)(a2 + 8);
            v41 = 0;
            v42 = (v38 - 1) & (((unsigned int)a1 >> 9) ^ ((unsigned int)a1 >> 4));
            v43 = 1;
            v19 = *(_DWORD *)(a2 + 16) + 1;
            v10 = v47;
            result = (__int64 *)(v40 + 16LL * v42);
            v44 = *result;
            if ( *result != a1 )
            {
              while ( v44 != -8 )
              {
                if ( v44 == -16 && !v41 )
                  v41 = result;
                v42 = v39 & (v43 + v42);
                result = (__int64 *)(v40 + 16LL * v42);
                v44 = *result;
                if ( *result == a1 )
                  goto LABEL_25;
                ++v43;
              }
              if ( v41 )
                result = v41;
            }
          }
          goto LABEL_25;
        }
LABEL_42:
        v46 = v10;
        sub_1DC6D40(a2, 2 * v11);
        v31 = *(_DWORD *)(a2 + 24);
        if ( !v31 )
          goto LABEL_78;
        v32 = v31 - 1;
        v33 = *(_QWORD *)(a2 + 8);
        v34 = (v31 - 1) & (((unsigned int)a1 >> 9) ^ ((unsigned int)a1 >> 4));
        v19 = *(_DWORD *)(a2 + 16) + 1;
        v10 = v46;
        result = (__int64 *)(v33 + 16LL * v34);
        v35 = *result;
        if ( *result != a1 )
        {
          v36 = 1;
          v37 = 0;
          while ( v35 != -8 )
          {
            if ( v35 == -16 && !v37 )
              v37 = result;
            v34 = v32 & (v36 + v34);
            result = (__int64 *)(v33 + 16LL * v34);
            v35 = *result;
            if ( *result == a1 )
              goto LABEL_25;
            ++v36;
          }
          if ( v37 )
            result = v37;
        }
LABEL_25:
        *(_DWORD *)(a2 + 16) = v19;
        if ( *result != -8 )
          --*(_DWORD *)(a2 + 20);
        *result = a1;
        *((_DWORD *)result + 2) = 0;
      }
LABEL_12:
      *((_DWORD *)result + 2) = v12;
      if ( !a1 )
        BUG();
      if ( (*(_BYTE *)a1 & 4) != 0 )
      {
        a1 = *(_QWORD *)(a1 + 8);
        if ( v10 == (__int64 *)a1 )
          return result;
      }
      else
      {
        while ( (*(_BYTE *)(a1 + 46) & 8) != 0 )
          a1 = *(_QWORD *)(a1 + 8);
        a1 = *(_QWORD *)(a1 + 8);
        if ( v10 == (__int64 *)a1 )
          return result;
      }
    }
    ++*(_QWORD *)a2;
    goto LABEL_42;
  }
  return result;
}
