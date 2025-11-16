// Function: sub_228C5E0
// Address: 0x228c5e0
//
unsigned __int64 __fastcall sub_228C5E0(
        unsigned __int64 *a1,
        unsigned __int64 *a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6)
{
  unsigned __int64 v8; // rsi
  unsigned __int64 v9; // rax
  unsigned __int64 v10; // rsi
  unsigned __int64 v11; // rax
  unsigned __int64 result; // rax
  unsigned __int64 v13; // rsi
  unsigned __int64 v14; // r8
  unsigned __int64 v15; // rsi
  unsigned __int64 v16; // rdx
  unsigned __int64 v17; // r13
  __int64 v18; // rdi
  char v19; // r8
  char v20; // r9
  __int64 *v21; // rdi
  __int64 v22; // r13
  char v23; // cl
  unsigned __int64 v24; // r13
  unsigned __int64 v25; // rcx
  unsigned __int64 v26; // rdx
  unsigned __int64 v27; // rdi
  unsigned int v28; // r9d
  unsigned int v29; // ecx
  __int64 v30; // rdx
  _QWORD *v31; // rdi
  __int64 v32; // r8
  __int64 v33; // rsi

  v8 = *a2;
  if ( (v8 & 1) != 0 )
  {
    v9 = *a1;
    v10 = v8 >> 58;
    if ( (*a1 & 1) != 0 )
    {
LABEL_3:
      v11 = v9 >> 58;
      goto LABEL_4;
    }
  }
  else
  {
    v9 = *a1;
    v10 = *(unsigned int *)(v8 + 64);
    if ( (*a1 & 1) != 0 )
      goto LABEL_3;
  }
  v11 = *(unsigned int *)(v9 + 64);
LABEL_4:
  if ( v10 < v11 )
    LODWORD(v10) = v11;
  sub_228BF90(a1, v10, 0, a4, a5, a6);
  result = *a1;
  if ( (*a1 & 1) != 0 )
  {
    v26 = *a2;
    if ( (*a2 & 1) != 0 )
    {
      result = 2 * ((*a1 >> 58 << 57) | ~(-1LL << (v26 >> 58)) & ~(-1LL << (*a1 >> 58)) & ((v26 & result) >> 1)) + 1;
      *a1 = result;
      return result;
    }
    v15 = *(unsigned int *)(v26 + 64);
    v14 = result >> 58;
    if ( v15 > result >> 58 )
      v15 = result >> 58;
    if ( v15 )
      goto LABEL_11;
    goto LABEL_30;
  }
  v13 = *a2;
  if ( (*a2 & 1) != 0 )
  {
    v14 = *(unsigned int *)(result + 64);
    v15 = v13 >> 58;
    if ( v15 > v14 )
      v15 = *(unsigned int *)(result + 64);
    if ( v15 )
    {
LABEL_11:
      v16 = 0;
      while ( 1 )
      {
        v19 = v16;
        v20 = result & 1;
        if ( (result & 1) != 0 )
        {
          v17 = result >> 58;
          v18 = (result >> 1) & ~(-1LL << (result >> 58));
          if ( _bittest64(&v18, v16) )
            goto LABEL_17;
LABEL_13:
          result = 2 * (~(1LL << v16) & v18 | (v17 << 57)) + 1;
          *a1 = result;
LABEL_14:
          if ( ++v16 == v15 )
            goto LABEL_38;
        }
        else
        {
          v21 = (__int64 *)(*(_QWORD *)result + 8LL * ((unsigned int)v16 >> 6));
          v22 = *v21;
          v23 = v16 & 0x3F;
          if ( !_bittest64(&v22, v16) )
            goto LABEL_43;
LABEL_17:
          v24 = *a2;
          if ( (*a2 & 1) != 0 )
            v25 = (((v24 >> 1) & ~(-1LL << (*a2 >> 58))) >> v16) & 1;
          else
            v25 = (*(_QWORD *)(*(_QWORD *)v24 + 8LL * ((unsigned int)v16 >> 6)) >> v16) & 1LL;
          if ( !(_BYTE)v25 )
          {
            if ( !v20 )
            {
              v21 = (__int64 *)(*(_QWORD *)result + 8LL * ((unsigned int)v16 >> 6));
              v22 = *v21;
              v23 = v16 & 0x3F;
LABEL_43:
              *v21 = ~(1LL << v23) & v22;
              result = *a1;
              goto LABEL_14;
            }
            v17 = result >> 58;
            v18 = (result >> 1) & ~(-1LL << (result >> 58));
            goto LABEL_13;
          }
          if ( !v20 )
          {
            *(_QWORD *)(*(_QWORD *)result + 8LL * ((unsigned int)v16 >> 6)) |= 1LL << v16;
            result = *a1;
            goto LABEL_14;
          }
          ++v16;
          result = 2
                 * ((result >> 58 << 57)
                  | ~(-1LL << (result >> 58)) & (~(-1LL << (result >> 58)) & (result >> 1) | (1LL << v19)))
                 + 1;
          *a1 = result;
          if ( v16 == v15 )
          {
LABEL_38:
            if ( (result & 1) != 0 )
            {
              result >>= 58;
              v14 = result;
            }
            else
            {
              v14 = *(unsigned int *)(result + 64);
            }
            break;
          }
        }
      }
    }
LABEL_30:
    while ( v14 != v15 )
    {
      v27 = *a1;
      if ( (*a1 & 1) != 0 )
      {
        result = 2 * ((*a1 >> 58 << 57) | ~(1LL << v15) & (v27 >> 1) & ~(-1LL << (*a1 >> 58))) + 1;
        *a1 = result;
      }
      else
      {
        result = (unsigned int)v15 >> 6;
        *(_QWORD *)(*(_QWORD *)v27 + 8 * result) &= ~(1LL << v15);
      }
      ++v15;
    }
    return result;
  }
  v28 = *(_DWORD *)(result + 8);
  v29 = v28;
  if ( *(_DWORD *)(v13 + 8) <= v28 )
    v29 = *(_DWORD *)(v13 + 8);
  v30 = 0;
  if ( !v29 )
    goto LABEL_51;
  do
  {
    v31 = (_QWORD *)(v30 + *(_QWORD *)result);
    v32 = *(_QWORD *)(*(_QWORD *)v13 + v30);
    v30 += 8;
    *v31 &= v32;
  }
  while ( 8LL * v29 != v30 );
  while ( v28 != v29 )
  {
    v33 = v29++;
    *(_QWORD *)(*(_QWORD *)result + 8 * v33) = 0;
LABEL_51:
    ;
  }
  return result;
}
