// Function: sub_37C5A70
// Address: 0x37c5a70
//
_WORD *__fastcall sub_37C5A70(__int64 a1, int a2)
{
  __int64 v3; // r13
  __int64 v4; // r12
  unsigned int v5; // eax
  _WORD *result; // rax
  __int64 v7; // rdx
  unsigned __int16 *v8; // r8
  _WORD *i; // rdx
  unsigned __int16 *v10; // rdx
  unsigned __int16 v11; // cx
  int v12; // r9d
  unsigned __int16 v13; // si
  int v14; // r9d
  __int64 v15; // rdi
  int v16; // r15d
  unsigned __int16 *v17; // r14
  unsigned int j; // eax
  unsigned __int16 *v19; // r10
  unsigned __int16 v20; // r11
  unsigned __int16 v21; // ax
  unsigned int v22; // eax
  __int64 v23; // rdx
  _WORD *k; // rdx

  v3 = *(unsigned int *)(a1 + 24);
  v4 = *(_QWORD *)(a1 + 8);
  v5 = sub_AF1560((unsigned int)(a2 - 1));
  if ( v5 < 0x40 )
    v5 = 64;
  *(_DWORD *)(a1 + 24) = v5;
  result = (_WORD *)sub_C7D670(8LL * v5, 4);
  *(_QWORD *)(a1 + 8) = result;
  if ( v4 )
  {
    v7 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    v8 = (unsigned __int16 *)(v4 + 8 * v3);
    for ( i = &result[4 * v7]; i != result; result += 4 )
    {
      if ( result )
      {
        *result = -1;
        result[1] = -1;
      }
    }
    if ( v8 != (unsigned __int16 *)v4 )
    {
      v10 = (unsigned __int16 *)v4;
      while ( 1 )
      {
        v11 = *v10;
        if ( *v10 == 0xFFFF )
        {
          if ( v10[1] != 0xFFFF )
            goto LABEL_12;
          v10 += 4;
          if ( v8 == v10 )
            return (_WORD *)sub_C7D6A0(v4, 8 * v3, 4);
        }
        else if ( v11 == 0xFFFE && v10[1] == 0xFFFE )
        {
          v10 += 4;
          if ( v8 == v10 )
            return (_WORD *)sub_C7D6A0(v4, 8 * v3, 4);
        }
        else
        {
LABEL_12:
          v12 = *(_DWORD *)(a1 + 24);
          if ( !v12 )
          {
            MEMORY[0] = 0;
            BUG();
          }
          v13 = v10[1];
          v14 = v12 - 1;
          v16 = 1;
          v17 = 0;
          for ( j = v14
                  & (((0xBF58476D1CE4E5B9LL
                     * ((37 * (unsigned int)v13) | ((unsigned __int64)(37 * (unsigned int)v11) << 32))) >> 31)
                   ^ (756364221 * v13)); ; j = v14 & v22 )
          {
            v15 = *(_QWORD *)(a1 + 8);
            v19 = (unsigned __int16 *)(v15 + 8LL * j);
            v20 = *v19;
            if ( v11 == *v19 && v13 == v19[1] )
              break;
            if ( v20 == 0xFFFF )
            {
              if ( v19[1] == 0xFFFF )
              {
                if ( v17 )
                  v19 = v17;
                break;
              }
            }
            else if ( v20 == 0xFFFE && v19[1] == 0xFFFE && !v17 )
            {
              v17 = (unsigned __int16 *)(v15 + 8LL * j);
            }
            v22 = v16 + j;
            ++v16;
          }
          *v19 = v11;
          v21 = v10[1];
          v10 += 4;
          v19[1] = v21;
          *((_DWORD *)v19 + 1) = *((_DWORD *)v10 - 1);
          ++*(_DWORD *)(a1 + 16);
          if ( v8 == v10 )
            return (_WORD *)sub_C7D6A0(v4, 8 * v3, 4);
        }
      }
    }
    return (_WORD *)sub_C7D6A0(v4, 8 * v3, 4);
  }
  else
  {
    v23 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    for ( k = &result[4 * v23]; k != result; result += 4 )
    {
      if ( result )
      {
        *result = -1;
        result[1] = -1;
      }
    }
  }
  return result;
}
