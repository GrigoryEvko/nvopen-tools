// Function: sub_DC2B70
// Address: 0xdc2b70
//
_QWORD *__fastcall sub_DC2B70(__int64 a1, __int64 a2, __int64 a3, unsigned int a4)
{
  __int64 v6; // rax
  __int64 v7; // r9
  __int64 v8; // rcx
  __int64 v9; // r8
  __int64 v10; // rsi
  int v11; // r10d
  unsigned int i; // eax
  __int64 v13; // rdx
  unsigned int v14; // eax
  _QWORD *result; // rax
  _QWORD *v16; // [rsp+8h] [rbp-48h]
  __m128i v17; // [rsp+10h] [rbp-40h] BYREF
  __int16 v18; // [rsp+20h] [rbp-30h]

  v6 = sub_D97090(a1, a3);
  v8 = *(unsigned int *)(a1 + 184);
  v17.m128i_i64[0] = a2;
  v17.m128i_i64[1] = v6;
  v9 = v6;
  v10 = *(_QWORD *)(a1 + 168);
  v18 = 3;
  if ( (_DWORD)v8 )
  {
    v11 = 1;
    for ( i = (v8 - 1)
            & (((0xBF58476D1CE4E5B9LL
               * ((484763065 * (_DWORD)v6)
                ^ (unsigned int)((0xBF58476D1CE4E5B9LL * ((unsigned int)v6 | (unsigned __int64)(a2 << 32))) >> 31)
                | 0x300000000LL)) >> 31)
             ^ (484763065
              * ((484763065 * v6) ^ ((0xBF58476D1CE4E5B9LL * ((unsigned int)v6 | (unsigned __int64)(a2 << 32))) >> 31))));
          ;
          i = (v8 - 1) & v14 )
    {
      v13 = v10 + 32LL * i;
      v7 = *(_QWORD *)v13;
      if ( a2 == *(_QWORD *)v13 && v9 == *(_QWORD *)(v13 + 8) && *(_WORD *)(v13 + 16) == 3 )
        break;
      if ( !v7 && !*(_QWORD *)(v13 + 8) && !*(_WORD *)(v13 + 16) )
        goto LABEL_8;
      v14 = v11 + i;
      ++v11;
    }
    if ( v13 != v10 + 32 * v8 )
      return *(_QWORD **)(v13 + 24);
  }
LABEL_8:
  result = sub_DC1A10(a1, a2, v9, a4, v9, v7);
  if ( *((_WORD *)result + 12) != 3 )
  {
    v16 = result;
    sub_DB1930(&v17, (__int64)result, a1 + 160, a1 + 192);
    return v16;
  }
  return result;
}
