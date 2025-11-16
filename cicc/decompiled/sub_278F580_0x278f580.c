// Function: sub_278F580
// Address: 0x278f580
//
__int64 __fastcall sub_278F580(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  int v7; // eax
  __int64 v8; // rdx
  _DWORD *v9; // rax
  _DWORD *i; // rdx
  __int64 *v11; // rbx
  __int64 *v12; // r13
  __int64 v13; // rsi
  __int64 v14; // rdi
  __int64 v15; // rdx
  __int64 result; // rax
  unsigned int v17; // ecx
  unsigned int v18; // eax
  _DWORD *v19; // rdi
  int v20; // ebx
  unsigned __int64 v21; // rdx
  unsigned __int64 v22; // rax
  _DWORD *v23; // rax
  __int64 v24; // rdx
  _DWORD *j; // rdx
  __int64 *v26; // rax
  __int64 v27; // rcx
  __int64 *v28; // rbx
  __int64 *v29; // r14
  __int64 v30; // rdi
  unsigned int v31; // ecx
  __int64 v32; // rsi
  _DWORD *v33; // rax

  sub_278EBA0(a1 + 136, a2, a3, a4, a5, a6);
  v7 = *(_DWORD *)(a1 + 368);
  ++*(_QWORD *)(a1 + 352);
  if ( !v7 )
  {
    if ( !*(_DWORD *)(a1 + 372) )
      goto LABEL_7;
    v8 = *(unsigned int *)(a1 + 376);
    if ( (unsigned int)v8 > 0x40 )
    {
      sub_C7D6A0(*(_QWORD *)(a1 + 360), 40 * v8, 8);
      *(_QWORD *)(a1 + 360) = 0;
      *(_QWORD *)(a1 + 368) = 0;
      *(_DWORD *)(a1 + 376) = 0;
      goto LABEL_7;
    }
    goto LABEL_4;
  }
  v17 = 4 * v7;
  v8 = *(unsigned int *)(a1 + 376);
  if ( (unsigned int)(4 * v7) < 0x40 )
    v17 = 64;
  if ( (unsigned int)v8 <= v17 )
  {
LABEL_4:
    v9 = *(_DWORD **)(a1 + 360);
    for ( i = &v9[10 * v8]; i != v9; v9 += 10 )
      *v9 = -1;
    *(_QWORD *)(a1 + 368) = 0;
    goto LABEL_7;
  }
  v18 = v7 - 1;
  if ( !v18 )
  {
    v19 = *(_DWORD **)(a1 + 360);
    v20 = 64;
LABEL_18:
    sub_C7D6A0((__int64)v19, 40 * v8, 8);
    v21 = ((((((((4 * v20 / 3u + 1) | ((unsigned __int64)(4 * v20 / 3u + 1) >> 1)) >> 2)
             | (4 * v20 / 3u + 1)
             | ((unsigned __int64)(4 * v20 / 3u + 1) >> 1)) >> 4)
           | (((4 * v20 / 3u + 1) | ((unsigned __int64)(4 * v20 / 3u + 1) >> 1)) >> 2)
           | (4 * v20 / 3u + 1)
           | ((unsigned __int64)(4 * v20 / 3u + 1) >> 1)) >> 8)
         | (((((4 * v20 / 3u + 1) | ((unsigned __int64)(4 * v20 / 3u + 1) >> 1)) >> 2)
           | (4 * v20 / 3u + 1)
           | ((unsigned __int64)(4 * v20 / 3u + 1) >> 1)) >> 4)
         | (((4 * v20 / 3u + 1) | ((unsigned __int64)(4 * v20 / 3u + 1) >> 1)) >> 2)
         | (4 * v20 / 3u + 1)
         | ((unsigned __int64)(4 * v20 / 3u + 1) >> 1)) >> 16;
    v22 = (v21
         | (((((((4 * v20 / 3u + 1) | ((unsigned __int64)(4 * v20 / 3u + 1) >> 1)) >> 2)
             | (4 * v20 / 3u + 1)
             | ((unsigned __int64)(4 * v20 / 3u + 1) >> 1)) >> 4)
           | (((4 * v20 / 3u + 1) | ((unsigned __int64)(4 * v20 / 3u + 1) >> 1)) >> 2)
           | (4 * v20 / 3u + 1)
           | ((unsigned __int64)(4 * v20 / 3u + 1) >> 1)) >> 8)
         | (((((4 * v20 / 3u + 1) | ((unsigned __int64)(4 * v20 / 3u + 1) >> 1)) >> 2)
           | (4 * v20 / 3u + 1)
           | ((unsigned __int64)(4 * v20 / 3u + 1) >> 1)) >> 4)
         | (((4 * v20 / 3u + 1) | ((unsigned __int64)(4 * v20 / 3u + 1) >> 1)) >> 2)
         | (4 * v20 / 3u + 1)
         | ((unsigned __int64)(4 * v20 / 3u + 1) >> 1))
        + 1;
    *(_DWORD *)(a1 + 376) = v22;
    v23 = (_DWORD *)sub_C7D670(40 * v22, 8);
    v24 = *(unsigned int *)(a1 + 376);
    *(_QWORD *)(a1 + 368) = 0;
    *(_QWORD *)(a1 + 360) = v23;
    for ( j = &v23[10 * v24]; j != v23; v23 += 10 )
    {
      if ( v23 )
        *v23 = -1;
    }
    goto LABEL_7;
  }
  _BitScanReverse(&v18, v18);
  v19 = *(_DWORD **)(a1 + 360);
  v20 = 1 << (33 - (v18 ^ 0x1F));
  if ( v20 < 64 )
    v20 = 64;
  if ( (_DWORD)v8 != v20 )
    goto LABEL_18;
  *(_QWORD *)(a1 + 368) = 0;
  v33 = &v19[10 * v8];
  do
  {
    if ( v19 )
      *v19 = -1;
    v19 += 10;
  }
  while ( v33 != v19 );
LABEL_7:
  v11 = *(__int64 **)(a1 + 448);
  v12 = &v11[2 * *(unsigned int *)(a1 + 456)];
  while ( v12 != v11 )
  {
    v13 = v11[1];
    v14 = *v11;
    v11 += 2;
    sub_C7D6A0(v14, v13, 16);
  }
  v15 = *(unsigned int *)(a1 + 408);
  *(_DWORD *)(a1 + 456) = 0;
  if ( (_DWORD)v15 )
  {
    v26 = *(__int64 **)(a1 + 400);
    *(_QWORD *)(a1 + 464) = 0;
    v27 = *v26;
    v28 = &v26[v15];
    v29 = v26 + 1;
    *(_QWORD *)(a1 + 384) = *v26;
    *(_QWORD *)(a1 + 392) = v27 + 4096;
    if ( v28 != v26 + 1 )
    {
      while ( 1 )
      {
        v30 = *v29;
        v31 = (unsigned int)(v29 - v26) >> 7;
        v32 = 4096LL << v31;
        if ( v31 >= 0x1E )
          v32 = 0x40000000000LL;
        ++v29;
        sub_C7D6A0(v30, v32, 16);
        if ( v28 == v29 )
          break;
        v26 = *(__int64 **)(a1 + 400);
      }
    }
    *(_DWORD *)(a1 + 408) = 1;
  }
  sub_278E0A0(a1 + 728);
  result = sub_30EC4F0(*(_QWORD *)(a1 + 104));
  *(_BYTE *)(a1 + 760) = 1;
  return result;
}
