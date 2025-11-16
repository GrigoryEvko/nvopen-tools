// Function: sub_3212810
// Address: 0x3212810
//
__int64 __fastcall sub_3212810(__int64 a1, __int64 *a2)
{
  __int64 v3; // rax
  unsigned __int8 v4; // dl
  __int64 v5; // rax
  __int64 v6; // r13
  __int64 v7; // rbx
  unsigned __int64 v8; // rdi
  int v9; // eax
  __int64 result; // rax
  __int64 v11; // rdx
  __int64 i; // rdx
  unsigned int v13; // ecx
  unsigned int v14; // eax
  _QWORD *v15; // rdi
  int v16; // ebx
  unsigned __int64 v17; // rax
  unsigned __int64 v18; // rdi
  __int64 v19; // rdx
  __int64 j; // rdx

  if ( *(_QWORD *)(a1 + 8) )
  {
    v3 = sub_B92180(*a2);
    if ( v3 )
    {
      v4 = *(_BYTE *)(v3 - 16);
      v5 = (v4 & 2) != 0 ? *(_QWORD *)(v3 - 32) : v3 - 16 - 8LL * ((v4 >> 2) & 0xF);
      if ( *(_DWORD *)(*(_QWORD *)(v5 + 40) + 32LL) )
        (*(void (__fastcall **)(__int64, __int64 *))(*(_QWORD *)a1 + 128LL))(a1, a2);
    }
  }
  sub_3212620(a1 + 336);
  v6 = *(_QWORD *)(a1 + 368);
  v7 = v6 + 96LL * *(unsigned int *)(a1 + 376);
  while ( v6 != v7 )
  {
    while ( 1 )
    {
      v7 -= 96;
      v8 = *(_QWORD *)(v7 + 16);
      if ( v8 == v7 + 32 )
        break;
      _libc_free(v8);
      if ( v6 == v7 )
        goto LABEL_11;
    }
  }
LABEL_11:
  *(_DWORD *)(a1 + 376) = 0;
  sub_3212620(a1 + 384);
  *(_DWORD *)(a1 + 424) = 0;
  sub_3212450(a1 + 432);
  sub_3212450(a1 + 464);
  v9 = *(_DWORD *)(a1 + 512);
  ++*(_QWORD *)(a1 + 496);
  if ( !v9 )
  {
    result = *(unsigned int *)(a1 + 516);
    if ( !(_DWORD)result )
      return result;
    v11 = *(unsigned int *)(a1 + 520);
    if ( (unsigned int)v11 > 0x40 )
    {
      result = sub_C7D6A0(*(_QWORD *)(a1 + 504), 16LL * (unsigned int)v11, 8);
      *(_QWORD *)(a1 + 504) = 0;
      *(_QWORD *)(a1 + 512) = 0;
      *(_DWORD *)(a1 + 520) = 0;
      return result;
    }
    goto LABEL_14;
  }
  v13 = 4 * v9;
  v11 = *(unsigned int *)(a1 + 520);
  if ( (unsigned int)(4 * v9) < 0x40 )
    v13 = 64;
  if ( (unsigned int)v11 <= v13 )
  {
LABEL_14:
    result = *(_QWORD *)(a1 + 504);
    for ( i = result + 16 * v11; i != result; result += 16 )
      *(_QWORD *)result = -4096;
    *(_QWORD *)(a1 + 512) = 0;
    return result;
  }
  v14 = v9 - 1;
  if ( v14 )
  {
    _BitScanReverse(&v14, v14);
    v15 = *(_QWORD **)(a1 + 504);
    v16 = 1 << (33 - (v14 ^ 0x1F));
    if ( v16 < 64 )
      v16 = 64;
    if ( v16 == (_DWORD)v11 )
    {
      *(_QWORD *)(a1 + 512) = 0;
      result = (__int64)&v15[2 * (unsigned int)v16];
      do
      {
        if ( v15 )
          *v15 = -4096;
        v15 += 2;
      }
      while ( (_QWORD *)result != v15 );
      return result;
    }
  }
  else
  {
    v15 = *(_QWORD **)(a1 + 504);
    v16 = 64;
  }
  sub_C7D6A0((__int64)v15, 16LL * (unsigned int)v11, 8);
  v17 = ((((((((4 * v16 / 3u + 1) | ((unsigned __int64)(4 * v16 / 3u + 1) >> 1)) >> 2)
           | (4 * v16 / 3u + 1)
           | ((unsigned __int64)(4 * v16 / 3u + 1) >> 1)) >> 4)
         | (((4 * v16 / 3u + 1) | ((unsigned __int64)(4 * v16 / 3u + 1) >> 1)) >> 2)
         | (4 * v16 / 3u + 1)
         | ((unsigned __int64)(4 * v16 / 3u + 1) >> 1)) >> 8)
       | (((((4 * v16 / 3u + 1) | ((unsigned __int64)(4 * v16 / 3u + 1) >> 1)) >> 2)
         | (4 * v16 / 3u + 1)
         | ((unsigned __int64)(4 * v16 / 3u + 1) >> 1)) >> 4)
       | (((4 * v16 / 3u + 1) | ((unsigned __int64)(4 * v16 / 3u + 1) >> 1)) >> 2)
       | (4 * v16 / 3u + 1)
       | ((unsigned __int64)(4 * v16 / 3u + 1) >> 1)) >> 16;
  v18 = (v17
       | (((((((4 * v16 / 3u + 1) | ((unsigned __int64)(4 * v16 / 3u + 1) >> 1)) >> 2)
           | (4 * v16 / 3u + 1)
           | ((unsigned __int64)(4 * v16 / 3u + 1) >> 1)) >> 4)
         | (((4 * v16 / 3u + 1) | ((unsigned __int64)(4 * v16 / 3u + 1) >> 1)) >> 2)
         | (4 * v16 / 3u + 1)
         | ((unsigned __int64)(4 * v16 / 3u + 1) >> 1)) >> 8)
       | (((((4 * v16 / 3u + 1) | ((unsigned __int64)(4 * v16 / 3u + 1) >> 1)) >> 2)
         | (4 * v16 / 3u + 1)
         | ((unsigned __int64)(4 * v16 / 3u + 1) >> 1)) >> 4)
       | (((4 * v16 / 3u + 1) | ((unsigned __int64)(4 * v16 / 3u + 1) >> 1)) >> 2)
       | (4 * v16 / 3u + 1)
       | ((unsigned __int64)(4 * v16 / 3u + 1) >> 1))
      + 1;
  *(_DWORD *)(a1 + 520) = v18;
  result = sub_C7D670(16 * v18, 8);
  v19 = *(unsigned int *)(a1 + 520);
  *(_QWORD *)(a1 + 512) = 0;
  *(_QWORD *)(a1 + 504) = result;
  for ( j = result + 16 * v19; j != result; result += 16 )
  {
    if ( result )
      *(_QWORD *)result = -4096;
  }
  return result;
}
