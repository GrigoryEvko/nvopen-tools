// Function: sub_13A5430
// Address: 0x13a5430
//
unsigned __int64 __fastcall sub_13A5430(
        unsigned __int64 *a1,
        unsigned __int64 *a2,
        __int64 a3,
        __int64 a4,
        int a5,
        int a6)
{
  unsigned __int64 v7; // rsi
  unsigned __int64 v8; // rsi
  unsigned __int64 v9; // rax
  unsigned __int64 v10; // rax
  unsigned __int64 result; // rax
  int v12; // ecx
  int v13; // r8d
  int v14; // r9d
  unsigned __int64 v15; // rbx
  unsigned __int64 v16; // r14
  unsigned __int64 v17; // rsi
  unsigned __int64 v18; // rcx
  __int64 v19; // r10
  unsigned __int64 v20; // rdx
  char v21; // r11
  char v22; // r10
  __int64 *v23; // rcx
  __int64 v24; // rdx
  unsigned __int64 v25; // r10
  unsigned __int64 v26; // rdx
  unsigned __int64 v27; // rdx
  unsigned int v28; // esi
  bool v29; // zf
  unsigned __int64 v30; // rsi

  v7 = *a2;
  if ( (v7 & 1) != 0 )
    v8 = v7 >> 58;
  else
    v8 = *(unsigned int *)(v7 + 16);
  v9 = *a1;
  if ( (*a1 & 1) != 0 )
    v10 = v9 >> 58;
  else
    v10 = *(unsigned int *)(v9 + 16);
  if ( v8 < v10 )
    LODWORD(v8) = v10;
  result = sub_13A5100(a1, v8, 0, a4, a5, a6);
  v15 = *a1;
  if ( (*a1 & 1) != 0 )
  {
    v27 = *a2;
    if ( (*a2 & 1) != 0 )
    {
      result = 2
             * ((v15 >> 58 << 57)
              | ~(-1LL << (v15 >> 58)) & (~(-1LL << (v15 >> 58)) & (v15 >> 1) | (v27 >> 1) & ~(-1LL << (*a2 >> 58))))
             + 1;
      *a1 = result;
      return result;
    }
    v17 = *(unsigned int *)(v27 + 16);
  }
  else
  {
    v16 = *a2;
    v17 = *a2 >> 58;
    if ( (*a2 & 1) == 0 )
    {
      v28 = *(_DWORD *)(v16 + 16);
      if ( *(_DWORD *)(v15 + 16) < v28 )
      {
        sub_13A49F0(*a1, v28, 0, v12, v13, v14);
        v28 = *(_DWORD *)(v16 + 16);
      }
      v29 = (v28 + 63) >> 6 == 0;
      result = (v28 + 63) >> 6;
      v30 = result;
      if ( !v29 )
      {
        result = 0;
        do
        {
          *(_QWORD *)(*(_QWORD *)v15 + 8 * result) |= *(_QWORD *)(*(_QWORD *)v16 + 8 * result);
          ++result;
        }
        while ( v30 != result );
      }
      return result;
    }
  }
  if ( v17 )
  {
    result = 0;
    while ( 1 )
    {
      v21 = v15 & 1;
      if ( (v15 & 1) != 0 )
      {
        v18 = v15 >> 58;
        v19 = ~(-1LL << (v15 >> 58));
        v20 = v19 & (v15 >> 1);
        if ( _bittest64((const __int64 *)&v20, result) )
          goto LABEL_12;
      }
      else
      {
        v22 = result & 0x3F;
        v23 = (__int64 *)(*(_QWORD *)v15 + 8LL * ((unsigned int)result >> 6));
        v24 = *v23;
        if ( _bittest64(&v24, result) )
          goto LABEL_17;
      }
      v25 = *a2;
      if ( (*a2 & 1) != 0 )
        v26 = (((v25 >> 1) & ~(-1LL << (*a2 >> 58))) >> result) & 1;
      else
        v26 = (*(_QWORD *)(*(_QWORD *)v25 + 8LL * ((unsigned int)result >> 6)) >> result) & 1LL;
      if ( !(_BYTE)v26 )
      {
        if ( v21 )
          *a1 = 2 * ((v15 >> 58 << 57) | ~((-1LL << (v15 >> 58)) | (1LL << result)) & (v15 >> 1)) + 1;
        else
          *(_QWORD *)(*(_QWORD *)v15 + 8LL * ((unsigned int)result >> 6)) &= ~(1LL << result);
        goto LABEL_13;
      }
      if ( v21 )
      {
        v18 = v15 >> 58;
        v19 = ~(-1LL << (v15 >> 58));
        v20 = v19 & (v15 >> 1);
LABEL_12:
        *a1 = 2 * ((v18 << 57) | v19 & (v20 | (1LL << result))) + 1;
LABEL_13:
        if ( v17 == ++result )
          return result;
        goto LABEL_14;
      }
      v22 = result & 0x3F;
      v23 = (__int64 *)(*(_QWORD *)v15 + 8LL * ((unsigned int)result >> 6));
      v24 = *v23;
LABEL_17:
      ++result;
      *v23 = v24 | (1LL << v22);
      if ( v17 == result )
        return result;
LABEL_14:
      v15 = *a1;
    }
  }
  return result;
}
