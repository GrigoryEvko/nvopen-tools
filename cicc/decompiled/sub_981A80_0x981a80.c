// Function: sub_981A80
// Address: 0x981a80
//
__int64 __fastcall sub_981A80(__int64 a1, unsigned int a2, _BYTE *a3, size_t a4)
{
  __int64 result; // rax
  unsigned int v9; // esi
  int v10; // r11d
  __int64 v11; // rdx
  __int64 v12; // r9
  unsigned int v13; // edi
  int v14; // ecx
  __int64 v15; // rdx
  unsigned __int8 *v16; // rdi
  __int64 *v17; // rbx
  __int64 v18; // rcx
  int v19; // eax
  int v20; // esi
  __int64 v21; // r8
  int v22; // ecx
  int v23; // edi
  size_t v24; // rdx
  int v25; // eax
  int v26; // eax
  int v27; // r9d
  __int64 v28; // r8
  __int64 v29; // rdi
  unsigned int v30; // r13d
  int v31; // esi
  int v32; // r10d
  __int64 v33; // r9
  unsigned __int8 *v34; // [rsp+0h] [rbp-50h] BYREF
  size_t n; // [rsp+8h] [rbp-48h]
  unsigned __int8 src[64]; // [rsp+10h] [rbp-40h] BYREF

  if ( a4 == qword_4977328[2 * a2] && (!a4 || !memcmp((&off_4977320)[2 * a2], a3, a4)) )
  {
    result = a2 >> 2;
    *(_BYTE *)(a1 + result) |= 3 << (2 * (a2 & 3));
    return result;
  }
  v34 = src;
  *(_BYTE *)(a1 + (a2 >> 2)) = (1 << (2 * (a2 & 3))) | *(_BYTE *)(a1 + (a2 >> 2)) & ~(3 << (2 * (a2 & 3)));
  sub_97E470((__int64 *)&v34, a3, (__int64)&a3[a4]);
  v9 = *(_DWORD *)(a1 + 160);
  if ( !v9 )
  {
    ++*(_QWORD *)(a1 + 136);
    goto LABEL_15;
  }
  v10 = 1;
  v11 = 0;
  v12 = *(_QWORD *)(a1 + 144);
  v13 = (v9 - 1) & (37 * a2);
  result = v12 + 40LL * v13;
  v14 = *(_DWORD *)result;
  if ( a2 != *(_DWORD *)result )
  {
    while ( v14 != -1 )
    {
      if ( v14 == -2 && !v11 )
        v11 = result;
      v13 = (v9 - 1) & (v10 + v13);
      result = v12 + 40LL * v13;
      v14 = *(_DWORD *)result;
      if ( a2 == *(_DWORD *)result )
        goto LABEL_8;
      ++v10;
    }
    if ( !v11 )
      v11 = result;
    v25 = *(_DWORD *)(a1 + 152);
    ++*(_QWORD *)(a1 + 136);
    v22 = v25 + 1;
    if ( 4 * (v25 + 1) < 3 * v9 )
    {
      result = v9 - *(_DWORD *)(a1 + 156) - v22;
      if ( (unsigned int)result > v9 >> 3 )
        goto LABEL_17;
      sub_981820(a1 + 136, v9);
      v26 = *(_DWORD *)(a1 + 160);
      if ( v26 )
      {
        result = (unsigned int)(v26 - 1);
        v27 = 1;
        v28 = 0;
        v29 = *(_QWORD *)(a1 + 144);
        v30 = result & (37 * a2);
        v22 = *(_DWORD *)(a1 + 152) + 1;
        v11 = v29 + 40LL * v30;
        v31 = *(_DWORD *)v11;
        if ( a2 != *(_DWORD *)v11 )
        {
          while ( v31 != -1 )
          {
            if ( v31 == -2 && !v28 )
              v28 = v11;
            v30 = result & (v27 + v30);
            v11 = v29 + 40LL * v30;
            v31 = *(_DWORD *)v11;
            if ( a2 == *(_DWORD *)v11 )
              goto LABEL_17;
            ++v27;
          }
          if ( v28 )
            v11 = v28;
        }
        goto LABEL_17;
      }
      goto LABEL_59;
    }
LABEL_15:
    sub_981820(a1 + 136, 2 * v9);
    v19 = *(_DWORD *)(a1 + 160);
    if ( v19 )
    {
      v20 = v19 - 1;
      v21 = *(_QWORD *)(a1 + 144);
      result = (v19 - 1) & (37 * a2);
      v22 = *(_DWORD *)(a1 + 152) + 1;
      v11 = v21 + 40 * result;
      v23 = *(_DWORD *)v11;
      if ( a2 != *(_DWORD *)v11 )
      {
        v32 = 1;
        v33 = 0;
        while ( v23 != -1 )
        {
          if ( !v33 && v23 == -2 )
            v33 = v11;
          result = v20 & (unsigned int)(v32 + result);
          v11 = v21 + 40LL * (unsigned int)result;
          v23 = *(_DWORD *)v11;
          if ( a2 == *(_DWORD *)v11 )
            goto LABEL_17;
          ++v32;
        }
        if ( v33 )
          v11 = v33;
      }
LABEL_17:
      *(_DWORD *)(a1 + 152) = v22;
      if ( *(_DWORD *)v11 != -1 )
        --*(_DWORD *)(a1 + 156);
      v16 = (unsigned __int8 *)(v11 + 24);
      *(_DWORD *)v11 = a2;
      v17 = (__int64 *)(v11 + 8);
      *(_QWORD *)(v11 + 8) = v11 + 24;
      *(_QWORD *)(v11 + 16) = 0;
      *(_BYTE *)(v11 + 24) = 0;
      v15 = (__int64)v34;
      if ( v34 != src )
        goto LABEL_25;
LABEL_20:
      v24 = n;
      if ( n )
      {
        if ( n == 1 )
        {
          result = src[0];
          *v16 = src[0];
        }
        else
        {
          result = (__int64)memcpy(v16, src, n);
        }
        v24 = n;
        v16 = (unsigned __int8 *)*v17;
      }
      v17[1] = v24;
      v16[v24] = 0;
      v16 = v34;
      goto LABEL_12;
    }
LABEL_59:
    ++*(_DWORD *)(a1 + 152);
    BUG();
  }
LABEL_8:
  v15 = (__int64)v34;
  v16 = *(unsigned __int8 **)(result + 8);
  v17 = (__int64 *)(result + 8);
  if ( v34 == src )
    goto LABEL_20;
  if ( (unsigned __int8 *)(result + 24) == v16 )
  {
LABEL_25:
    *v17 = v15;
    v17[1] = n;
    result = *(_QWORD *)src;
    v17[2] = *(_QWORD *)src;
    goto LABEL_26;
  }
  *(_QWORD *)(result + 8) = v34;
  v18 = *(_QWORD *)(result + 24);
  *(_QWORD *)(result + 16) = n;
  *(_QWORD *)(result + 24) = *(_QWORD *)src;
  if ( v16 )
  {
    v34 = v16;
    *(_QWORD *)src = v18;
    goto LABEL_12;
  }
LABEL_26:
  v34 = src;
  v16 = src;
LABEL_12:
  n = 0;
  *v16 = 0;
  if ( v34 != src )
    return j_j___libc_free_0(v34, *(_QWORD *)src + 1LL);
  return result;
}
