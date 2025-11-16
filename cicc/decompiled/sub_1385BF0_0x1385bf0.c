// Function: sub_1385BF0
// Address: 0x1385bf0
//
void __fastcall sub_1385BF0(__int64 *a1, unsigned __int8 a2)
{
  __int64 v2; // rax
  __int64 v3; // rdx
  int v4; // r10d
  __int64 *v7; // rcx
  __int64 v8; // rdi
  int v9; // esi
  __int64 v10; // rcx
  unsigned __int64 v11; // r8
  unsigned __int64 v12; // r8
  unsigned int i; // eax
  __int64 v14; // r8
  unsigned int v15; // eax
  __int64 *v16; // rbx
  __int64 *v17; // r14
  __int64 v18; // rdx
  __int64 v19; // rcx
  __int64 v20; // rax

  v2 = *a1;
  v3 = *(unsigned int *)(*a1 + 24);
  if ( (_DWORD)v3 )
  {
    v4 = 1;
    v7 = (__int64 *)a1[1];
    v8 = *v7;
    v9 = *((_DWORD *)v7 + 2);
    v10 = *(_QWORD *)(v2 + 8);
    v11 = ((((unsigned int)(37 * v9) | ((unsigned __int64)(((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4)) << 32))
          - 1
          - ((unsigned __int64)(unsigned int)(37 * v9) << 32)) >> 22)
        ^ (((unsigned int)(37 * v9) | ((unsigned __int64)(((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4)) << 32))
         - 1
         - ((unsigned __int64)(unsigned int)(37 * v9) << 32));
    v12 = ((9 * (((v11 - 1 - (v11 << 13)) >> 8) ^ (v11 - 1 - (v11 << 13)))) >> 15)
        ^ (9 * (((v11 - 1 - (v11 << 13)) >> 8) ^ (v11 - 1 - (v11 << 13))));
    for ( i = (v3 - 1) & (((v12 - 1 - (v12 << 27)) >> 31) ^ (v12 - 1 - ((_DWORD)v12 << 27))); ; i = (v3 - 1) & v15 )
    {
      v14 = v10 + 48LL * i;
      if ( v8 == *(_QWORD *)v14 && v9 == *(_DWORD *)(v14 + 8) )
        break;
      if ( *(_QWORD *)v14 == -8 && *(_DWORD *)(v14 + 8) == -1 )
        return;
      v15 = v4 + i;
      ++v4;
    }
    if ( v14 != 48 * v3 + v10 )
    {
      v16 = *(__int64 **)(v14 + 24);
      v17 = &v16[2 * *(unsigned int *)(v14 + 40)];
      if ( *(_DWORD *)(v14 + 32) )
      {
        if ( v17 != v16 )
        {
          while ( 1 )
          {
            if ( *v16 == -8 )
            {
              if ( *((_DWORD *)v16 + 2) != -1 )
                goto LABEL_14;
              goto LABEL_24;
            }
            if ( *v16 != -16 || *((_DWORD *)v16 + 2) != -2 )
              break;
LABEL_24:
            v16 += 2;
            if ( v17 == v16 )
              return;
          }
          while ( 1 )
          {
LABEL_14:
            if ( v17 == v16 )
              return;
            v18 = *v16;
            v19 = v16[1];
            v16 += 2;
            sub_1385450(*(_QWORD *)a1[2], *(_QWORD *)(a1[2] + 8), v18, v19, a2, a1[3], a1[4]);
            if ( v16 == v17 )
              return;
            v20 = *v16;
            if ( *v16 == -8 )
              goto LABEL_21;
            while ( v20 == -16 && *((_DWORD *)v16 + 2) == -2 )
            {
              while ( 1 )
              {
                v16 += 2;
                if ( v17 == v16 )
                  return;
                v20 = *v16;
                if ( *v16 != -8 )
                  break;
LABEL_21:
                if ( *((_DWORD *)v16 + 2) != -1 )
                  goto LABEL_14;
              }
            }
          }
        }
      }
    }
  }
}
