// Function: sub_2A72C90
// Address: 0x2a72c90
//
__int64 __fastcall sub_2A72C90(__int64 a1, unsigned __int8 *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  int v6; // r13d
  __int64 result; // rax
  __int64 v9; // rsi
  __int64 i; // rdx
  unsigned __int8 **v11; // rbx
  unsigned __int8 **v12; // r12
  unsigned int v13; // eax
  unsigned int v14; // eax
  __int64 v15; // rbx
  _QWORD *v16; // rdi
  unsigned __int64 v17; // rdx
  unsigned __int64 v18; // rax
  __int64 v19; // rdx
  __int64 j; // rdx

  while ( 1 )
  {
    sub_2A72020(a1, (__int64)a2, a3, a4, a5, a6);
    a3 = *(unsigned int *)(a1 + 344);
    if ( !(_DWORD)a3 )
      break;
    v11 = *(unsigned __int8 ***)(a1 + 336);
    v12 = &v11[*(unsigned int *)(a1 + 352)];
    if ( v11 == v12 )
      break;
    while ( 1 )
    {
      LOBYTE(v6) = *v11 + 4096 == 0 || *v11 + 0x2000 == 0;
      if ( !(_BYTE)v6 )
        break;
      if ( v12 == ++v11 )
        goto LABEL_2;
    }
    if ( v11 == v12 )
      break;
    do
    {
      a2 = *v11;
      if ( **v11 > 0x1Cu )
        v6 |= sub_2A6CBA0(a1, a2, a3, a4);
      if ( ++v11 == v12 )
        break;
      while ( *v11 == (unsigned __int8 *)-8192LL || *v11 == (unsigned __int8 *)-4096LL )
      {
        if ( v12 == ++v11 )
          goto LABEL_20;
      }
    }
    while ( v12 != v11 );
LABEL_20:
    if ( !(_BYTE)v6 )
    {
      LODWORD(a3) = *(_DWORD *)(a1 + 344);
      break;
    }
  }
LABEL_2:
  ++*(_QWORD *)(a1 + 328);
  if ( !(_DWORD)a3 )
  {
    result = *(unsigned int *)(a1 + 348);
    if ( !(_DWORD)result )
      return result;
    v9 = *(unsigned int *)(a1 + 352);
    if ( (unsigned int)v9 > 0x40 )
    {
      result = sub_C7D6A0(*(_QWORD *)(a1 + 336), 8 * v9, 8);
      *(_QWORD *)(a1 + 336) = 0;
      *(_QWORD *)(a1 + 344) = 0;
      *(_DWORD *)(a1 + 352) = 0;
      return result;
    }
    goto LABEL_5;
  }
  v13 = 4 * a3;
  v9 = *(unsigned int *)(a1 + 352);
  if ( (unsigned int)(4 * a3) < 0x40 )
    v13 = 64;
  if ( v13 >= (unsigned int)v9 )
  {
LABEL_5:
    result = *(_QWORD *)(a1 + 336);
    for ( i = result + 8 * v9; i != result; result += 8 )
      *(_QWORD *)result = -4096;
    *(_QWORD *)(a1 + 344) = 0;
    return result;
  }
  if ( (_DWORD)a3 == 1 )
  {
    v16 = *(_QWORD **)(a1 + 336);
    LODWORD(v15) = 64;
LABEL_31:
    sub_C7D6A0((__int64)v16, 8 * v9, 8);
    v17 = ((((((((4 * (int)v15 / 3u + 1) | ((unsigned __int64)(4 * (int)v15 / 3u + 1) >> 1)) >> 2)
             | (4 * (int)v15 / 3u + 1)
             | ((unsigned __int64)(4 * (int)v15 / 3u + 1) >> 1)) >> 4)
           | (((4 * (int)v15 / 3u + 1) | ((unsigned __int64)(4 * (int)v15 / 3u + 1) >> 1)) >> 2)
           | (4 * (int)v15 / 3u + 1)
           | ((unsigned __int64)(4 * (int)v15 / 3u + 1) >> 1)) >> 8)
         | (((((4 * (int)v15 / 3u + 1) | ((unsigned __int64)(4 * (int)v15 / 3u + 1) >> 1)) >> 2)
           | (4 * (int)v15 / 3u + 1)
           | ((unsigned __int64)(4 * (int)v15 / 3u + 1) >> 1)) >> 4)
         | (((4 * (int)v15 / 3u + 1) | ((unsigned __int64)(4 * (int)v15 / 3u + 1) >> 1)) >> 2)
         | (4 * (int)v15 / 3u + 1)
         | ((unsigned __int64)(4 * (int)v15 / 3u + 1) >> 1)) >> 16;
    v18 = (v17
         | (((((((4 * (int)v15 / 3u + 1) | ((unsigned __int64)(4 * (int)v15 / 3u + 1) >> 1)) >> 2)
             | (4 * (int)v15 / 3u + 1)
             | ((unsigned __int64)(4 * (int)v15 / 3u + 1) >> 1)) >> 4)
           | (((4 * (int)v15 / 3u + 1) | ((unsigned __int64)(4 * (int)v15 / 3u + 1) >> 1)) >> 2)
           | (4 * (int)v15 / 3u + 1)
           | ((unsigned __int64)(4 * (int)v15 / 3u + 1) >> 1)) >> 8)
         | (((((4 * (int)v15 / 3u + 1) | ((unsigned __int64)(4 * (int)v15 / 3u + 1) >> 1)) >> 2)
           | (4 * (int)v15 / 3u + 1)
           | ((unsigned __int64)(4 * (int)v15 / 3u + 1) >> 1)) >> 4)
         | (((4 * (int)v15 / 3u + 1) | ((unsigned __int64)(4 * (int)v15 / 3u + 1) >> 1)) >> 2)
         | (4 * (int)v15 / 3u + 1)
         | ((unsigned __int64)(4 * (int)v15 / 3u + 1) >> 1))
        + 1;
    *(_DWORD *)(a1 + 352) = v18;
    result = sub_C7D670(8 * v18, 8);
    v19 = *(unsigned int *)(a1 + 352);
    *(_QWORD *)(a1 + 344) = 0;
    *(_QWORD *)(a1 + 336) = result;
    for ( j = result + 8 * v19; j != result; result += 8 )
    {
      if ( result )
        *(_QWORD *)result = -4096;
    }
    return result;
  }
  _BitScanReverse(&v14, a3 - 1);
  v15 = (unsigned int)(1 << (33 - (v14 ^ 0x1F)));
  if ( (int)v15 < 64 )
    v15 = 64;
  v16 = *(_QWORD **)(a1 + 336);
  if ( (_DWORD)v15 != (_DWORD)v9 )
    goto LABEL_31;
  *(_QWORD *)(a1 + 344) = 0;
  result = (__int64)&v16[v15];
  do
  {
    if ( v16 )
      *v16 = -4096;
    ++v16;
  }
  while ( (_QWORD *)result != v16 );
  return result;
}
