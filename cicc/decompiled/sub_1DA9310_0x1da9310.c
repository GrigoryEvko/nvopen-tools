// Function: sub_1DA9310
// Address: 0x1da9310
//
__int64 __fastcall sub_1DA9310(__int64 a1, __int64 a2)
{
  unsigned __int64 v4; // r10
  __int64 v5; // rdx
  __int64 *v7; // rdi
  __int64 v8; // r11
  __int64 *v9; // rbx
  __int64 v10; // rdx
  unsigned int v11; // r9d
  __int64 v12; // rsi
  __int64 *v13; // rcx

  v4 = a2 & 0xFFFFFFFFFFFFFFF8LL;
  if ( (a2 & 0xFFFFFFFFFFFFFFF8LL) != 0 )
  {
    v5 = *(_QWORD *)(v4 + 16);
    if ( v5 )
      return *(_QWORD *)(v5 + 24);
  }
  v7 = *(__int64 **)(a1 + 536);
  v8 = *(unsigned int *)(a1 + 544);
  v9 = &v7[2 * v8];
  v10 = (16 * v8) >> 4;
  if ( 16 * v8 )
  {
    v11 = *(_DWORD *)(v4 + 24) | (a2 >> 1) & 3;
    do
    {
      while ( 1 )
      {
        v12 = v10 >> 1;
        v13 = &v7[2 * (v10 >> 1)];
        if ( (*(_DWORD *)((*v13 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(*v13 >> 1) & 3) >= v11 )
          break;
        v7 = v13 + 2;
        v10 = v10 - v12 - 1;
        if ( v10 <= 0 )
          goto LABEL_9;
      }
      v10 >>= 1;
    }
    while ( v12 > 0 );
  }
LABEL_9:
  if ( v9 == v7 )
  {
    if ( !(_DWORD)v8 )
      return v7[1];
    goto LABEL_11;
  }
  if ( (*(_DWORD *)((*v7 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(*v7 >> 1) & 3) > (*(_DWORD *)(v4 + 24)
                                                                                        | (unsigned int)(a2 >> 1) & 3) )
LABEL_11:
    v7 -= 2;
  return v7[1];
}
