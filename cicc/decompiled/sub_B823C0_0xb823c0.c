// Function: sub_B823C0
// Address: 0xb823c0
//
__int64 __fastcall sub_B823C0(__int64 a1)
{
  __int64 v2; // rbx
  int v3; // eax
  __int64 result; // rax
  __int64 v5; // rdx
  __int64 i; // rdx
  unsigned int v7; // ecx
  unsigned int v8; // eax
  _QWORD *v9; // rdi
  int v10; // r13d
  unsigned __int64 v11; // rax
  unsigned __int64 v12; // rdi
  __int64 v13; // rdx
  __int64 j; // rdx

  v2 = *(_QWORD *)(*(_QWORD *)(a1 + 8) - 8LL);
  v3 = *(_DWORD *)(v2 + 224);
  ++*(_QWORD *)(v2 + 208);
  if ( !v3 )
  {
    result = *(unsigned int *)(v2 + 228);
    if ( !(_DWORD)result )
      goto LABEL_7;
    v5 = *(unsigned int *)(v2 + 232);
    if ( (unsigned int)v5 > 0x40 )
    {
      result = sub_C7D6A0(*(_QWORD *)(v2 + 216), 16LL * (unsigned int)v5, 8);
      *(_QWORD *)(v2 + 216) = 0;
      *(_QWORD *)(v2 + 224) = 0;
      *(_DWORD *)(v2 + 232) = 0;
      goto LABEL_7;
    }
    goto LABEL_4;
  }
  v7 = 4 * v3;
  v5 = *(unsigned int *)(v2 + 232);
  if ( (unsigned int)(4 * v3) < 0x40 )
    v7 = 64;
  if ( (unsigned int)v5 <= v7 )
  {
LABEL_4:
    result = *(_QWORD *)(v2 + 216);
    for ( i = result + 16 * v5; i != result; result += 16 )
      *(_QWORD *)result = -4096;
    *(_QWORD *)(v2 + 224) = 0;
    goto LABEL_7;
  }
  v8 = v3 - 1;
  if ( !v8 )
  {
    v9 = *(_QWORD **)(v2 + 216);
    v10 = 64;
LABEL_15:
    sub_C7D6A0(v9, 16LL * (unsigned int)v5, 8);
    v11 = ((((((((4 * v10 / 3u + 1) | ((unsigned __int64)(4 * v10 / 3u + 1) >> 1)) >> 2)
             | (4 * v10 / 3u + 1)
             | ((unsigned __int64)(4 * v10 / 3u + 1) >> 1)) >> 4)
           | (((4 * v10 / 3u + 1) | ((unsigned __int64)(4 * v10 / 3u + 1) >> 1)) >> 2)
           | (4 * v10 / 3u + 1)
           | ((unsigned __int64)(4 * v10 / 3u + 1) >> 1)) >> 8)
         | (((((4 * v10 / 3u + 1) | ((unsigned __int64)(4 * v10 / 3u + 1) >> 1)) >> 2)
           | (4 * v10 / 3u + 1)
           | ((unsigned __int64)(4 * v10 / 3u + 1) >> 1)) >> 4)
         | (((4 * v10 / 3u + 1) | ((unsigned __int64)(4 * v10 / 3u + 1) >> 1)) >> 2)
         | (4 * v10 / 3u + 1)
         | ((unsigned __int64)(4 * v10 / 3u + 1) >> 1)) >> 16;
    v12 = (v11
         | (((((((4 * v10 / 3u + 1) | ((unsigned __int64)(4 * v10 / 3u + 1) >> 1)) >> 2)
             | (4 * v10 / 3u + 1)
             | ((unsigned __int64)(4 * v10 / 3u + 1) >> 1)) >> 4)
           | (((4 * v10 / 3u + 1) | ((unsigned __int64)(4 * v10 / 3u + 1) >> 1)) >> 2)
           | (4 * v10 / 3u + 1)
           | ((unsigned __int64)(4 * v10 / 3u + 1) >> 1)) >> 8)
         | (((((4 * v10 / 3u + 1) | ((unsigned __int64)(4 * v10 / 3u + 1) >> 1)) >> 2)
           | (4 * v10 / 3u + 1)
           | ((unsigned __int64)(4 * v10 / 3u + 1) >> 1)) >> 4)
         | (((4 * v10 / 3u + 1) | ((unsigned __int64)(4 * v10 / 3u + 1) >> 1)) >> 2)
         | (4 * v10 / 3u + 1)
         | ((unsigned __int64)(4 * v10 / 3u + 1) >> 1))
        + 1;
    *(_DWORD *)(v2 + 232) = v12;
    result = sub_C7D670(16 * v12, 8);
    v13 = *(unsigned int *)(v2 + 232);
    *(_QWORD *)(v2 + 224) = 0;
    *(_QWORD *)(v2 + 216) = result;
    for ( j = result + 16 * v13; j != result; result += 16 )
    {
      if ( result )
        *(_QWORD *)result = -4096;
    }
    goto LABEL_7;
  }
  _BitScanReverse(&v8, v8);
  v9 = *(_QWORD **)(v2 + 216);
  v10 = 1 << (33 - (v8 ^ 0x1F));
  if ( v10 < 64 )
    v10 = 64;
  if ( (_DWORD)v5 != v10 )
    goto LABEL_15;
  *(_QWORD *)(v2 + 224) = 0;
  result = (__int64)&v9[2 * (unsigned int)v5];
  do
  {
    if ( v9 )
      *v9 = -4096;
    v9 += 2;
  }
  while ( (_QWORD *)result != v9 );
LABEL_7:
  *(_OWORD *)(v2 + 160) = 0;
  *(_OWORD *)(v2 + 176) = 0;
  *(_OWORD *)(v2 + 192) = 0;
  *(_QWORD *)(a1 + 8) -= 8LL;
  return result;
}
