// Function: sub_2D75210
// Address: 0x2d75210
//
void __fastcall sub_2D75210(__int64 a1)
{
  __int64 *v1; // rbx
  __int64 *i; // r13
  unsigned int v4; // esi
  __int64 v5; // rcx
  int v6; // r11d
  __int64 *v7; // r10
  __int64 v8; // rdi
  unsigned int j; // eax
  __int64 *v10; // r8
  __int64 v11; // r15
  int v12; // edx
  __int64 v13; // rax
  unsigned int v14; // eax
  int v15; // eax
  __int64 *v16; // [rsp+8h] [rbp-38h] BYREF

  v1 = *(__int64 **)(a1 + 32);
  for ( i = &v1[2 * *(unsigned int *)(a1 + 40)]; i != v1; v7[1] = *(v1 - 1) )
  {
LABEL_2:
    v4 = *(_DWORD *)(a1 + 24);
    if ( v4 )
    {
      v5 = v1[1];
      v6 = 1;
      v7 = 0;
      for ( j = (v4 - 1)
              & (((0xBF58476D1CE4E5B9LL
                 * (((unsigned int)v5 >> 9) ^ ((unsigned int)v5 >> 4)
                  | ((unsigned __int64)(((unsigned int)*v1 >> 9) ^ ((unsigned int)*v1 >> 4)) << 32))) >> 31)
               ^ (484763065 * (((unsigned int)v5 >> 9) ^ ((unsigned int)v5 >> 4)))); ; j = (v4 - 1) & v14 )
      {
        v8 = *(_QWORD *)(a1 + 8);
        v10 = (__int64 *)(v8 + 16LL * j);
        v11 = *v10;
        if ( *v10 == *v1 && v10[1] == v5 )
        {
          v1 += 2;
          if ( i != v1 )
            goto LABEL_2;
          return;
        }
        if ( v11 == -4096 )
        {
          if ( v10[1] == -4096 )
          {
            v15 = *(_DWORD *)(a1 + 16);
            if ( !v7 )
              v7 = v10;
            ++*(_QWORD *)a1;
            v12 = v15 + 1;
            v16 = v7;
            if ( 4 * (v15 + 1) >= 3 * v4 )
              goto LABEL_15;
            if ( v4 - *(_DWORD *)(a1 + 20) - v12 > v4 >> 3 )
              goto LABEL_17;
            goto LABEL_16;
          }
        }
        else if ( v11 == -8192 && v10[1] == -8192 && !v7 )
        {
          v7 = (__int64 *)(v8 + 16LL * j);
        }
        v14 = v6 + j;
        ++v6;
      }
    }
    ++*(_QWORD *)a1;
    v16 = 0;
LABEL_15:
    v4 *= 2;
LABEL_16:
    sub_2D74F50(a1, v4);
    sub_2D6B120(a1, v1, &v16);
    v7 = v16;
    v12 = *(_DWORD *)(a1 + 16) + 1;
LABEL_17:
    *(_DWORD *)(a1 + 16) = v12;
    if ( *v7 != -4096 || v7[1] != -4096 )
      --*(_DWORD *)(a1 + 20);
    v13 = *v1;
    v1 += 2;
    *v7 = v13;
  }
}
