// Function: sub_228C270
// Address: 0x228c270
//
__int64 __fastcall sub_228C270(
        unsigned __int64 *a1,
        unsigned __int64 *a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6)
{
  unsigned __int64 v7; // rsi
  unsigned __int64 v8; // rsi
  unsigned __int64 v9; // rax
  unsigned __int64 v10; // rax
  __int64 result; // rax
  __int64 v12; // r9
  unsigned __int64 v13; // rbx
  unsigned __int64 v14; // r14
  unsigned __int64 v15; // rsi
  unsigned __int64 v16; // rcx
  __int64 v17; // r10
  unsigned __int64 v18; // rdx
  char v19; // r11
  char v20; // r10
  __int64 *v21; // rcx
  __int64 v22; // rdx
  unsigned __int64 v23; // r10
  unsigned __int64 v24; // rdx
  unsigned __int64 v25; // rdx
  __int64 v26; // rsi
  __int64 v27; // rsi
  _QWORD *v28; // rdx
  __int64 v29; // rcx
  int v30; // ecx
  unsigned __int64 v31; // rcx
  unsigned __int64 v32; // r8
  __int64 v33; // r12

  v7 = *a2;
  if ( (v7 & 1) != 0 )
    v8 = v7 >> 58;
  else
    v8 = *(unsigned int *)(v7 + 64);
  v9 = *a1;
  if ( (*a1 & 1) != 0 )
    v10 = v9 >> 58;
  else
    v10 = *(unsigned int *)(v9 + 64);
  if ( v8 < v10 )
    LODWORD(v8) = v10;
  result = sub_228BF90(a1, v8, 0, a4, a5, a6);
  v13 = *a1;
  if ( (*a1 & 1) != 0 )
  {
    v25 = *a2;
    if ( (*a2 & 1) != 0 )
    {
      result = 2
             * ((v13 >> 58 << 57)
              | ~(-1LL << (v13 >> 58)) & (~(-1LL << (v13 >> 58)) & (v13 >> 1) | (v25 >> 1) & ~(-1LL << (*a2 >> 58))))
             + 1;
      *a1 = result;
      return result;
    }
    v15 = *(unsigned int *)(v25 + 64);
  }
  else
  {
    v14 = *a2;
    v15 = *a2 >> 58;
    if ( (*a2 & 1) == 0 )
    {
      result = *(unsigned int *)(v14 + 64);
      if ( *(_DWORD *)(v13 + 64) < (unsigned int)result )
      {
        v30 = *(_DWORD *)(v13 + 64) & 0x3F;
        if ( v30 )
          *(_QWORD *)(*(_QWORD *)v13 + 8LL * *(unsigned int *)(v13 + 8) - 8) &= ~(-1LL << v30);
        v31 = *(unsigned int *)(v13 + 8);
        *(_DWORD *)(v13 + 64) = result;
        v32 = (unsigned int)(result + 63) >> 6;
        if ( v32 != v31 )
        {
          if ( v32 >= v31 )
          {
            v33 = v32 - v31;
            if ( v32 > *(unsigned int *)(v13 + 12) )
            {
              sub_C8D5F0(v13, (const void *)(v13 + 16), v32, 8u, v32, v12);
              v31 = *(unsigned int *)(v13 + 8);
            }
            if ( 8 * v33 )
            {
              memset((void *)(*(_QWORD *)v13 + 8 * v31), 0, 8 * v33);
              LODWORD(v31) = *(_DWORD *)(v13 + 8);
            }
            LODWORD(result) = *(_DWORD *)(v13 + 64);
            *(_DWORD *)(v13 + 8) = v33 + v31;
          }
          else
          {
            *(_DWORD *)(v13 + 8) = (unsigned int)(result + 63) >> 6;
          }
        }
        result &= 0x3Fu;
        if ( (_DWORD)result )
          *(_QWORD *)(*(_QWORD *)v13 + 8LL * *(unsigned int *)(v13 + 8) - 8) &= ~(-1LL << result);
      }
      v26 = *(unsigned int *)(v14 + 8);
      if ( (_DWORD)v26 )
      {
        v27 = 8 * v26;
        result = 0;
        do
        {
          v28 = (_QWORD *)(result + *(_QWORD *)v13);
          v29 = *(_QWORD *)(*(_QWORD *)v14 + result);
          result += 8;
          *v28 |= v29;
        }
        while ( v27 != result );
      }
      return result;
    }
  }
  if ( v15 )
  {
    result = 0;
    while ( 1 )
    {
      v19 = v13 & 1;
      if ( (v13 & 1) != 0 )
      {
        v16 = v13 >> 58;
        v17 = ~(-1LL << (v13 >> 58));
        v18 = v17 & (v13 >> 1);
        if ( _bittest64((const __int64 *)&v18, result) )
          goto LABEL_12;
      }
      else
      {
        v20 = result & 0x3F;
        v21 = (__int64 *)(*(_QWORD *)v13 + 8LL * ((unsigned int)result >> 6));
        v22 = *v21;
        if ( _bittest64(&v22, result) )
          goto LABEL_17;
      }
      v23 = *a2;
      if ( (*a2 & 1) != 0 )
        v24 = (((v23 >> 1) & ~(-1LL << (*a2 >> 58))) >> result) & 1;
      else
        v24 = (*(_QWORD *)(*(_QWORD *)v23 + 8LL * ((unsigned int)result >> 6)) >> result) & 1LL;
      if ( !(_BYTE)v24 )
      {
        if ( v19 )
          *a1 = 2 * ((v13 >> 58 << 57) | ~((-1LL << (v13 >> 58)) | (1LL << result)) & (v13 >> 1)) + 1;
        else
          *(_QWORD *)(*(_QWORD *)v13 + 8LL * ((unsigned int)result >> 6)) &= ~(1LL << result);
        goto LABEL_13;
      }
      if ( v19 )
      {
        v16 = v13 >> 58;
        v17 = ~(-1LL << (v13 >> 58));
        v18 = v17 & (v13 >> 1);
LABEL_12:
        *a1 = 2 * ((v16 << 57) | v17 & (v18 | (1LL << result))) + 1;
LABEL_13:
        if ( v15 == ++result )
          return result;
        goto LABEL_14;
      }
      v20 = result & 0x3F;
      v21 = (__int64 *)(*(_QWORD *)v13 + 8LL * ((unsigned int)result >> 6));
      v22 = *v21;
LABEL_17:
      ++result;
      *v21 = v22 | (1LL << v20);
      if ( v15 == result )
        return result;
LABEL_14:
      v13 = *a1;
    }
  }
  return result;
}
