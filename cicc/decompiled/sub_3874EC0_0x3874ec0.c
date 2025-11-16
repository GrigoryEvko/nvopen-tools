// Function: sub_3874EC0
// Address: 0x3874ec0
//
void __fastcall sub_3874EC0(__int64 a1, int a2)
{
  __int64 v3; // rbx
  unsigned __int64 v4; // r12
  unsigned __int64 v5; // rax
  _QWORD *v6; // rax
  __int64 *v7; // rbx
  _QWORD *i; // rdx
  __int64 *j; // r13
  __int64 v10; // rdx
  int v11; // ecx
  __int64 v12; // rsi
  int v13; // ecx
  __int64 v14; // rdi
  _QWORD *v15; // r11
  int v16; // r10d
  __int64 v17; // r8
  unsigned __int64 v18; // r8
  unsigned __int64 v19; // r8
  unsigned int k; // eax
  _QWORD *v21; // r8
  __int64 v22; // r9
  __int64 v23; // rax
  __int64 v24; // rax
  __int64 v25; // rax
  _QWORD *v26; // rdi
  unsigned int v27; // eax
  __int64 v28; // rdx
  _QWORD *m; // rdx

  v3 = *(unsigned int *)(a1 + 24);
  v4 = *(_QWORD *)(a1 + 8);
  v5 = ((((((((((unsigned int)(a2 - 1) | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 2)
            | (unsigned int)(a2 - 1)
            | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 4)
          | (((unsigned int)(a2 - 1) | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 2)
          | (unsigned int)(a2 - 1)
          | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 8)
        | (((((unsigned int)(a2 - 1) | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 2)
          | (unsigned int)(a2 - 1)
          | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 4)
        | (((unsigned int)(a2 - 1) | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 2)
        | (unsigned int)(a2 - 1)
        | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 16)
      | (((((((unsigned int)(a2 - 1) | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 2)
          | (unsigned int)(a2 - 1)
          | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 4)
        | (((unsigned int)(a2 - 1) | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 2)
        | (unsigned int)(a2 - 1)
        | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 8)
      | (((((unsigned int)(a2 - 1) | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 2)
        | (unsigned int)(a2 - 1)
        | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 4)
      | (((unsigned int)(a2 - 1) | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 2)
      | (unsigned int)(a2 - 1)
      | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1))
     + 1;
  if ( (unsigned int)v5 < 0x40 )
    LODWORD(v5) = 64;
  *(_DWORD *)(a1 + 24) = v5;
  v6 = (_QWORD *)sub_22077B0(40LL * (unsigned int)v5);
  *(_QWORD *)(a1 + 8) = v6;
  if ( v4 )
  {
    *(_QWORD *)(a1 + 16) = 0;
    v7 = (__int64 *)(v4 + 40 * v3);
    for ( i = &v6[5 * *(unsigned int *)(a1 + 24)]; i != v6; v6 += 5 )
    {
      if ( v6 )
      {
        *v6 = -8;
        v6[1] = -8;
      }
    }
    if ( v7 != (__int64 *)v4 )
    {
      for ( j = (__int64 *)v4; v7 != j; j += 5 )
      {
        while ( 1 )
        {
          v10 = *j;
          if ( *j != -8 )
            break;
          if ( j[1] == -8 )
          {
LABEL_22:
            j += 5;
            if ( v7 == j )
              goto LABEL_23;
          }
          else
          {
LABEL_12:
            v11 = *(_DWORD *)(a1 + 24);
            if ( !v11 )
            {
              MEMORY[0] = *j;
              BUG();
            }
            v12 = j[1];
            v13 = v11 - 1;
            v15 = 0;
            v16 = 1;
            v17 = ((unsigned int)v12 >> 9) ^ ((unsigned int)v12 >> 4);
            v18 = (((v17 | ((unsigned __int64)(((unsigned int)v10 >> 9) ^ ((unsigned int)v10 >> 4)) << 32))
                  - 1
                  - (v17 << 32)) >> 22)
                ^ ((v17 | ((unsigned __int64)(((unsigned int)v10 >> 9) ^ ((unsigned int)v10 >> 4)) << 32))
                 - 1
                 - (v17 << 32));
            v19 = ((9 * (((v18 - 1 - (v18 << 13)) >> 8) ^ (v18 - 1 - (v18 << 13)))) >> 15)
                ^ (9 * (((v18 - 1 - (v18 << 13)) >> 8) ^ (v18 - 1 - (v18 << 13))));
            for ( k = v13 & (((v19 - 1 - (v19 << 27)) >> 31) ^ (v19 - 1 - ((_DWORD)v19 << 27))); ; k = v13 & v27 )
            {
              v14 = *(_QWORD *)(a1 + 8);
              v21 = (_QWORD *)(v14 + 40LL * k);
              v22 = *v21;
              if ( v10 == *v21 && v21[1] == v12 )
                break;
              if ( v22 == -8 )
              {
                if ( v21[1] == -8 )
                {
                  if ( v15 )
                    v21 = v15;
                  break;
                }
              }
              else if ( v22 == -16 && v21[1] == -16 && !v15 )
              {
                v15 = (_QWORD *)(v14 + 40LL * k);
              }
              v27 = v16 + k;
              ++v16;
            }
            *v21 = v10;
            v23 = j[1];
            v21[2] = 6;
            v21[1] = v23;
            v21[3] = 0;
            v24 = j[4];
            v21[4] = v24;
            if ( v24 != -8 && v24 != 0 && v24 != -16 )
              sub_1649AC0(v21 + 2, j[2] & 0xFFFFFFFFFFFFFFF8LL);
            ++*(_DWORD *)(a1 + 16);
            v25 = j[4];
            if ( v25 == 0 || v25 == -8 || v25 == -16 )
              goto LABEL_22;
            v26 = j + 2;
            j += 5;
            sub_1649B30(v26);
            if ( v7 == j )
              goto LABEL_23;
          }
        }
        if ( v10 != -16 || j[1] != -16 )
          goto LABEL_12;
      }
    }
LABEL_23:
    j___libc_free_0(v4);
  }
  else
  {
    v28 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    for ( m = &v6[5 * v28]; m != v6; v6 += 5 )
    {
      if ( v6 )
      {
        *v6 = -8;
        v6[1] = -8;
      }
    }
  }
}
