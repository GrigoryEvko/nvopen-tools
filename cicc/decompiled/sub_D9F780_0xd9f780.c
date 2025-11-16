// Function: sub_D9F780
// Address: 0xd9f780
//
__int64 __fastcall sub_D9F780(__int64 a1, __int64 *a2, _QWORD *a3)
{
  int v4; // r8d
  __int64 v5; // r13
  __int64 v6; // r14
  int v7; // r8d
  __int64 v8; // r15
  int v9; // ecx
  __int64 v10; // r9
  int v11; // r10d
  unsigned int i; // ebx
  __int64 v13; // rdx
  __int64 result; // rax
  bool v15; // al
  __int64 v16; // rdx
  int v17; // [rsp+14h] [rbp-6Ch]
  __int64 v18; // [rsp+18h] [rbp-68h]
  int v19; // [rsp+20h] [rbp-60h]
  int v20; // [rsp+24h] [rbp-5Ch]
  _QWORD v21[2]; // [rsp+30h] [rbp-50h] BYREF
  __int16 v22; // [rsp+40h] [rbp-40h]

  v4 = *(_DWORD *)(a1 + 24);
  if ( v4 )
  {
    v5 = *a2;
    v6 = a2[1];
    v7 = v4 - 1;
    v8 = *(_QWORD *)(a1 + 8);
    v9 = *((unsigned __int16 *)a2 + 8);
    v22 = 1;
    v10 = 0;
    v11 = 1;
    v21[0] = 0;
    v21[1] = 0;
    for ( i = v7
            & (((0xBF58476D1CE4E5B9LL
               * (((unsigned __int64)(unsigned __int16)v9 << 32)
                | (484763065 * (_DWORD)v6)
                ^ (unsigned int)((0xBF58476D1CE4E5B9LL * ((unsigned int)v6 | (unsigned __int64)(v5 << 32))) >> 31))) >> 31)
             ^ (484763065
              * ((484763065 * v6) ^ ((0xBF58476D1CE4E5B9LL * ((unsigned int)v6 | (unsigned __int64)(v5 << 32))) >> 31))));
          ;
          i = v20 & (v19 + i) )
    {
      v13 = v8 + 32LL * i;
      result = *(_QWORD *)v13;
      if ( v5 == *(_QWORD *)v13 && v6 == *(_QWORD *)(v13 + 8) && (_WORD)v9 == *(_WORD *)(v13 + 16) )
      {
        *a3 = v13;
        return 1;
      }
      if ( !result && !*(_QWORD *)(v13 + 8) && !*(_WORD *)(v13 + 16) )
        break;
      v17 = v9;
      v19 = v11;
      v18 = v10;
      v20 = v7;
      v15 = sub_D95440(v8 + 32LL * i, (__int64)v21);
      v7 = v20;
      v9 = v17;
      if ( v18 || (v16 = v8 + 32LL * i, !v15) )
        v16 = v18;
      v10 = v16;
      v11 = v19 + 1;
    }
    if ( !v10 )
      v10 = v8 + 32LL * i;
    *a3 = v10;
  }
  else
  {
    *a3 = 0;
    return 0;
  }
  return result;
}
