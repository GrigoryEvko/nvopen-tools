// Function: sub_DC5000
// Address: 0xdc5000
//
_QWORD *__fastcall sub_DC5000(__int64 a1, __int64 a2, __int64 a3, unsigned int a4)
{
  __int64 v6; // rax
  __int64 v7; // rcx
  __int64 v8; // r8
  __int64 v9; // rsi
  int v10; // r10d
  unsigned int i; // eax
  __int64 v12; // rdx
  unsigned int v13; // eax
  _QWORD *result; // rax
  _QWORD *v15; // [rsp+8h] [rbp-48h]
  __m128i v16; // [rsp+10h] [rbp-40h] BYREF
  __int16 v17; // [rsp+20h] [rbp-30h]

  v6 = sub_D97090(a1, a3);
  v7 = *(unsigned int *)(a1 + 184);
  v16.m128i_i64[0] = a2;
  v16.m128i_i64[1] = v6;
  v8 = v6;
  v9 = *(_QWORD *)(a1 + 168);
  v17 = 4;
  if ( (_DWORD)v7 )
  {
    v10 = 1;
    for ( i = (v7 - 1)
            & (((0xBF58476D1CE4E5B9LL
               * ((484763065 * (_DWORD)v6)
                ^ (unsigned int)((0xBF58476D1CE4E5B9LL * ((unsigned int)v6 | (unsigned __int64)(a2 << 32))) >> 31)
                | 0x400000000LL)) >> 31)
             ^ (484763065
              * ((484763065 * v6) ^ ((0xBF58476D1CE4E5B9LL * ((unsigned int)v6 | (unsigned __int64)(a2 << 32))) >> 31))));
          ;
          i = (v7 - 1) & v13 )
    {
      v12 = v9 + 32LL * i;
      if ( a2 == *(_QWORD *)v12 && v8 == *(_QWORD *)(v12 + 8) && *(_WORD *)(v12 + 16) == 4 )
        break;
      if ( !*(_QWORD *)v12 && !*(_QWORD *)(v12 + 8) && !*(_WORD *)(v12 + 16) )
        goto LABEL_8;
      v13 = v10 + i;
      ++v10;
    }
    if ( v12 != v9 + 32 * v7 )
      return *(_QWORD **)(v12 + 24);
  }
LABEL_8:
  result = sub_DC42C0(a1, a2, v8, a4);
  if ( *((_WORD *)result + 12) != 4 )
  {
    v15 = result;
    sub_DB1930(&v16, (__int64)result, a1 + 160, a1 + 192);
    return v15;
  }
  return result;
}
