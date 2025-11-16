// Function: sub_D091D0
// Address: 0xd091d0
//
__int64 **__fastcall sub_D091D0(__int64 **a1, __int64 *a2, unsigned __int64 *a3)
{
  char v5; // dl
  __int64 *v6; // rbx
  int v7; // r9d
  unsigned __int64 v8; // r12
  unsigned __int64 v9; // r10
  __int64 v10; // rcx
  int v11; // edi
  unsigned int i; // r11d
  __int64 *v13; // rcx
  unsigned int v14; // r11d
  __int64 v15; // r9
  __int64 v16; // rax
  __int64 *v17; // rdx
  __int64 v19; // rax
  __int64 *v20; // rbx
  __int64 *v21; // rax

  v5 = a2[1] & 1;
  if ( v5 )
  {
    v6 = a2 + 2;
    v7 = 7;
  }
  else
  {
    v15 = *((unsigned int *)a2 + 6);
    v6 = (__int64 *)a2[2];
    if ( !(_DWORD)v15 )
    {
LABEL_16:
      v19 = 5 * v15;
      goto LABEL_17;
    }
    v7 = v15 - 1;
  }
  v8 = a3[2];
  v9 = *a3;
  v10 = (unsigned int)v8
      ^ (unsigned int)(v8 >> 9)
      ^ (unsigned int)((0xBF58476D1CE4E5B9LL * a3[3]) >> 31)
      ^ (484763065 * *((_DWORD *)a3 + 6));
  v11 = 1;
  for ( i = v7
          & (((0xBF58476D1CE4E5B9LL
             * (v10
              | ((unsigned __int64)((unsigned int)((0xBF58476D1CE4E5B9LL * a3[1]) >> 31)
                                  ^ (484763065 * *((_DWORD *)a3 + 2))
                                  ^ (unsigned int)v9
                                  ^ (unsigned int)(v9 >> 9)) << 32))) >> 31)
           ^ (484763065 * v10)); ; i = v7 & v14 )
  {
    v13 = &v6[5 * i];
    if ( v9 == *v13 && a3[1] == v13[1] && v8 == v13[2] && a3[3] == v13[3] )
    {
      v16 = 40;
      if ( !v5 )
        v16 = 5LL * *((unsigned int *)a2 + 6);
      v17 = (__int64 *)*a2;
      *a1 = a2;
      a1[3] = &v6[v16];
      a1[1] = v17;
      a1[2] = v13;
      return a1;
    }
    if ( *v13 == -4 && v13[1] == -3 && v13[2] == -4 && v13[3] == -3 )
      break;
    v14 = v11 + i;
    ++v11;
  }
  if ( !v5 )
  {
    v15 = *((unsigned int *)a2 + 6);
    goto LABEL_16;
  }
  v19 = 40;
LABEL_17:
  v20 = &v6[v19];
  v21 = (__int64 *)*a2;
  *a1 = a2;
  a1[2] = v20;
  a1[1] = v21;
  a1[3] = v20;
  return a1;
}
