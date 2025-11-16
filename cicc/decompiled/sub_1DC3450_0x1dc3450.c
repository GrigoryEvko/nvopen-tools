// Function: sub_1DC3450
// Address: 0x1dc3450
//
__int64 *__fastcall sub_1DC3450(__int64 *a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 *v4; // r8
  __int64 v7; // r9
  __int64 v8; // rax
  unsigned __int64 v10; // rsi
  unsigned int v11; // edx
  __int64 v12; // rcx
  __int64 *v13; // r9
  unsigned int v14; // eax
  unsigned int v16; // eax
  unsigned int v17; // eax
  unsigned int v18; // eax
  unsigned int v19; // eax
  unsigned int v20; // edx
  unsigned int v21; // edx
  unsigned int v22; // edx

  v4 = a1;
  v7 = (a2 - (__int64)a1) >> 5;
  v8 = (a2 - (__int64)a1) >> 3;
  if ( v7 <= 0 )
  {
LABEL_17:
    switch ( v8 )
    {
      case 2LL:
        v19 = *(_DWORD *)((a3 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (a3 >> 1) & 3;
        break;
      case 3LL:
        v19 = *(_DWORD *)((a3 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (a3 >> 1) & 3;
        v21 = *(_DWORD *)((*v4 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (*v4 >> 1) & 3;
        if ( v19 <= v21 && v21 < (*(_DWORD *)((a4 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(a4 >> 1) & 3) )
          return v4;
        ++v4;
        break;
      case 1LL:
        v19 = *(_DWORD *)((a3 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (a3 >> 1) & 3;
LABEL_22:
        v20 = *(_DWORD *)((*v4 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (*v4 >> 1) & 3;
        if ( v19 <= v20 )
        {
          if ( v20 >= (*(_DWORD *)((a4 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(a4 >> 1) & 3) )
            return (__int64 *)a2;
          return v4;
        }
        return (__int64 *)a2;
      default:
        return (__int64 *)a2;
    }
    v22 = *(_DWORD *)((*v4 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (*v4 >> 1) & 3;
    if ( v19 <= v22 && v22 < (*(_DWORD *)((a4 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(a4 >> 1) & 3) )
      return v4;
    ++v4;
    goto LABEL_22;
  }
  v10 = a4 & 0xFFFFFFFFFFFFFFF8LL;
  v11 = *(_DWORD *)((a3 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (a3 >> 1) & 3;
  v12 = (a4 >> 1) & 3;
  v13 = &a1[4 * v7];
  while ( 1 )
  {
    v14 = *(_DWORD *)((*v4 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (*v4 >> 1) & 3;
    if ( v11 <= v14 && v14 < ((unsigned int)v12 | *(_DWORD *)(v10 + 24)) )
      return v4;
    v16 = *(_DWORD *)((v4[1] & 0xFFFFFFFFFFFFFFF8LL) + 24) | (v4[1] >> 1) & 3;
    if ( v11 <= v16 && v16 < ((unsigned int)v12 | *(_DWORD *)(v10 + 24)) )
      return v4 + 1;
    v17 = *(_DWORD *)((v4[2] & 0xFFFFFFFFFFFFFFF8LL) + 24) | (v4[2] >> 1) & 3;
    if ( v11 <= v17 && v17 < ((unsigned int)v12 | *(_DWORD *)(v10 + 24)) )
    {
      v4 += 2;
      return v4;
    }
    v18 = *(_DWORD *)((v4[3] & 0xFFFFFFFFFFFFFFF8LL) + 24) | (v4[3] >> 1) & 3;
    if ( v11 <= v18 && v18 < ((unsigned int)v12 | *(_DWORD *)(v10 + 24)) )
    {
      v4 += 3;
      return v4;
    }
    v4 += 4;
    if ( v13 == v4 )
    {
      v8 = (a2 - (__int64)v4) >> 3;
      goto LABEL_17;
    }
  }
}
