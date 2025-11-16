// Function: sub_FF3180
// Address: 0xff3180
//
__int64 __fastcall sub_FF3180(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v5; // rbx
  bool v6; // zf
  __int64 v7; // rax
  unsigned __int64 v8; // rdx
  __int64 i; // rdx
  __int64 result; // rax
  __int64 v11; // rcx
  __int64 v12; // r8
  __int64 v13; // rdx
  __int64 v14; // rsi
  int v15; // r10d
  unsigned __int64 v16; // r11
  unsigned int j; // eax
  unsigned __int64 v18; // rdi
  __int64 v19; // r9
  int v20; // edx
  int v21; // eax
  __int64 v22; // rdi
  unsigned int v23; // eax
  __int64 v24; // [rsp+8h] [rbp-38h]

  v5 = a2;
  v6 = (*(_QWORD *)(a1 + 8) & 1LL) == 0;
  *(_QWORD *)(a1 + 8) &= 1uLL;
  if ( v6 )
  {
    v7 = *(_QWORD *)(a1 + 16);
    v8 = (unsigned __int64)*(unsigned int *)(a1 + 24) << 6;
  }
  else
  {
    v7 = a1 + 16;
    v8 = 256;
  }
  for ( i = v7 + v8; i != v7; v7 += 64 )
  {
    if ( v7 )
    {
      *(_QWORD *)v7 = -4096;
      *(_DWORD *)(v7 + 8) = 0x7FFFFFFF;
    }
  }
  result = a1 + 16;
  v24 = a1 + 16;
  if ( a2 != a3 )
  {
    do
    {
      while ( 1 )
      {
        v11 = *(_QWORD *)v5;
        if ( *(_QWORD *)v5 != -4096 )
          break;
        if ( *(_DWORD *)(v5 + 8) == 0x7FFFFFFF )
        {
LABEL_23:
          v5 += 64;
          if ( a3 == v5 )
            return result;
        }
        else
        {
LABEL_10:
          if ( (*(_BYTE *)(a1 + 8) & 1) != 0 )
          {
            v12 = v24;
            v13 = 3;
          }
          else
          {
            v20 = *(_DWORD *)(a1 + 24);
            v12 = *(_QWORD *)(a1 + 16);
            if ( !v20 )
            {
              MEMORY[0] = *(_QWORD *)v5;
              BUG();
            }
            v13 = (unsigned int)(v20 - 1);
          }
          v14 = *(unsigned int *)(v5 + 8);
          v15 = 1;
          v16 = 0;
          for ( j = v13
                  & (((0xBF58476D1CE4E5B9LL
                     * ((unsigned int)(37 * v14)
                      | ((unsigned __int64)(((unsigned int)v11 >> 9) ^ ((unsigned int)v11 >> 4)) << 32))) >> 31)
                   ^ (756364221 * v14)); ; j = v13 & v23 )
          {
            v18 = v12 + ((unsigned __int64)j << 6);
            v19 = *(_QWORD *)v18;
            if ( v11 == *(_QWORD *)v18 && *(_DWORD *)(v18 + 8) == (_DWORD)v14 )
              break;
            if ( v19 == -4096 )
            {
              if ( *(_DWORD *)(v18 + 8) == 0x7FFFFFFF )
              {
                if ( v16 )
                  v18 = v16;
                break;
              }
            }
            else if ( v19 == -8192 && *(_DWORD *)(v18 + 8) == 0x80000000 && !v16 )
            {
              v16 = v12 + ((unsigned __int64)j << 6);
            }
            v23 = v15 + j;
            ++v15;
          }
          *(_QWORD *)v18 = v11;
          v21 = *(_DWORD *)(v5 + 8);
          *(_QWORD *)(v18 + 24) = 0x400000000LL;
          *(_DWORD *)(v18 + 8) = v21;
          *(_QWORD *)(v18 + 16) = v18 + 32;
          if ( *(_DWORD *)(v5 + 24) )
          {
            v14 = v5 + 16;
            sub_FEE2C0(v18 + 16, (char **)(v5 + 16), v13, v11, v12, v19);
          }
          *(_DWORD *)(a1 + 8) = (2 * (*(_DWORD *)(a1 + 8) >> 1) + 2) | *(_DWORD *)(a1 + 8) & 1;
          v22 = *(_QWORD *)(v5 + 16);
          result = v5 + 32;
          if ( v22 == v5 + 32 )
            goto LABEL_23;
          result = _libc_free(v22, v14);
          v5 += 64;
          if ( a3 == v5 )
            return result;
        }
      }
      if ( v11 != -8192 || *(_DWORD *)(v5 + 8) != 0x80000000 )
        goto LABEL_10;
      v5 += 64;
    }
    while ( a3 != v5 );
  }
  return result;
}
