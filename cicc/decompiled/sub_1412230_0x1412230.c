// Function: sub_1412230
// Address: 0x1412230
//
__int64 __fastcall sub_1412230(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  _QWORD *v6; // rax
  int v7; // eax
  __int64 result; // rax
  __int64 v9; // rdx
  __int64 i; // rdx
  unsigned int v11; // ecx
  _QWORD *v12; // rdi
  unsigned int v13; // eax
  int v14; // eax
  unsigned __int64 v15; // rax
  __int64 v16; // rax
  int v17; // r13d
  __int64 v18; // r12
  __int64 v19; // rdx
  __int64 j; // rdx

  *(_QWORD *)a1 = 0;
  *(_QWORD *)(a1 + 8) = 0;
  *(_QWORD *)(a1 + 16) = 0;
  *(_DWORD *)(a1 + 24) = 0;
  *(_QWORD *)(a1 + 32) = 0;
  *(_QWORD *)(a1 + 40) = 0;
  *(_QWORD *)(a1 + 48) = 0;
  *(_DWORD *)(a1 + 56) = 0;
  *(_QWORD *)(a1 + 64) = 0;
  *(_QWORD *)(a1 + 72) = 0;
  *(_QWORD *)(a1 + 80) = 0;
  *(_DWORD *)(a1 + 88) = 0;
  *(_QWORD *)(a1 + 96) = 0;
  *(_QWORD *)(a1 + 104) = 0;
  *(_QWORD *)(a1 + 112) = 0;
  *(_DWORD *)(a1 + 120) = 0;
  *(_QWORD *)(a1 + 128) = 0;
  *(_QWORD *)(a1 + 136) = 0;
  *(_QWORD *)(a1 + 144) = 0;
  *(_DWORD *)(a1 + 152) = 0;
  *(_QWORD *)(a1 + 160) = 0;
  *(_QWORD *)(a1 + 168) = 0;
  *(_QWORD *)(a1 + 176) = 0;
  *(_DWORD *)(a1 + 184) = 0;
  *(_QWORD *)(a1 + 192) = 0;
  *(_QWORD *)(a1 + 200) = 0;
  *(_QWORD *)(a1 + 208) = 0;
  *(_DWORD *)(a1 + 216) = 0;
  *(_QWORD *)(a1 + 376) = a1 + 392;
  *(_QWORD *)(a1 + 224) = 0;
  *(_QWORD *)(a1 + 232) = 0;
  *(_QWORD *)(a1 + 240) = 0;
  *(_DWORD *)(a1 + 248) = 0;
  *(_QWORD *)(a1 + 256) = a2;
  *(_QWORD *)(a1 + 272) = a4;
  *(_QWORD *)(a1 + 280) = a5;
  *(_QWORD *)(a1 + 288) = a6;
  *(_QWORD *)(a1 + 296) = 0;
  *(_QWORD *)(a1 + 304) = 0;
  *(_QWORD *)(a1 + 312) = 0;
  *(_DWORD *)(a1 + 320) = 0;
  *(_QWORD *)(a1 + 328) = 0;
  *(_QWORD *)(a1 + 336) = 0;
  *(_QWORD *)(a1 + 344) = 0;
  *(_DWORD *)(a1 + 352) = 0;
  *(_QWORD *)(a1 + 360) = 0;
  *(_QWORD *)(a1 + 368) = 0;
  *(_QWORD *)(a1 + 432) = 0;
  *(_QWORD *)(a1 + 440) = 0;
  *(_QWORD *)(a1 + 448) = 1;
  *(_QWORD *)(a1 + 464) = 0;
  *(_QWORD *)(a1 + 472) = 1;
  *(_QWORD *)(a1 + 384) = 0x400000000LL;
  *(_QWORD *)(a1 + 264) = a3;
  *(_QWORD *)(a1 + 424) = a1 + 440;
  v6 = (_QWORD *)(a1 + 480);
  do
  {
    if ( v6 )
      *v6 = -8;
    v6 += 11;
  }
  while ( v6 != (_QWORD *)(a1 + 832) );
  *(_QWORD *)(a1 + 832) = 0;
  *(_QWORD *)(a1 + 840) = a1 + 872;
  *(_QWORD *)(a1 + 848) = a1 + 872;
  *(_QWORD *)(a1 + 856) = 8;
  *(_DWORD *)(a1 + 864) = 0;
  *(_BYTE *)(a1 + 936) = 0;
  v7 = *(_DWORD *)(a2 + 24);
  ++*(_QWORD *)(a2 + 8);
  *(_BYTE *)(a2 + 1) = 1;
  if ( !v7 )
  {
    result = *(unsigned int *)(a2 + 28);
    if ( !(_DWORD)result )
      return result;
    v9 = *(unsigned int *)(a2 + 32);
    if ( (unsigned int)v9 > 0x40 )
    {
      result = j___libc_free_0(*(_QWORD *)(a2 + 16));
      *(_QWORD *)(a2 + 16) = 0;
      *(_QWORD *)(a2 + 24) = 0;
      *(_DWORD *)(a2 + 32) = 0;
      return result;
    }
    goto LABEL_8;
  }
  v11 = 4 * v7;
  v9 = *(unsigned int *)(a2 + 32);
  if ( (unsigned int)(4 * v7) < 0x40 )
    v11 = 64;
  if ( (unsigned int)v9 <= v11 )
  {
LABEL_8:
    result = *(_QWORD *)(a2 + 16);
    for ( i = result + 24 * v9; i != result; *(_QWORD *)(result - 16) = -8 )
    {
      *(_QWORD *)result = -8;
      result += 24;
    }
    *(_QWORD *)(a2 + 24) = 0;
    return result;
  }
  v12 = *(_QWORD **)(a2 + 16);
  v13 = v7 - 1;
  if ( !v13 )
  {
    v18 = 3072;
    v17 = 128;
LABEL_20:
    j___libc_free_0(v12);
    *(_DWORD *)(a2 + 32) = v17;
    result = sub_22077B0(v18);
    v19 = *(unsigned int *)(a2 + 32);
    *(_QWORD *)(a2 + 24) = 0;
    *(_QWORD *)(a2 + 16) = result;
    for ( j = result + 24 * v19; j != result; result += 24 )
    {
      if ( result )
      {
        *(_QWORD *)result = -8;
        *(_QWORD *)(result + 8) = -8;
      }
    }
    return result;
  }
  _BitScanReverse(&v13, v13);
  v14 = 1 << (33 - (v13 ^ 0x1F));
  if ( v14 < 64 )
    v14 = 64;
  if ( (_DWORD)v9 != v14 )
  {
    v15 = (4 * v14 / 3u + 1) | ((unsigned __int64)(4 * v14 / 3u + 1) >> 1);
    v16 = ((((v15 >> 2) | v15 | (((v15 >> 2) | v15) >> 4)) >> 8)
         | (v15 >> 2)
         | v15
         | (((v15 >> 2) | v15) >> 4)
         | (((((v15 >> 2) | v15 | (((v15 >> 2) | v15) >> 4)) >> 8) | (v15 >> 2) | v15 | (((v15 >> 2) | v15) >> 4)) >> 16))
        + 1;
    v17 = v16;
    v18 = 24 * v16;
    goto LABEL_20;
  }
  *(_QWORD *)(a2 + 24) = 0;
  result = (__int64)&v12[3 * v9];
  do
  {
    if ( v12 )
    {
      *v12 = -8;
      v12[1] = -8;
    }
    v12 += 3;
  }
  while ( (_QWORD *)result != v12 );
  return result;
}
