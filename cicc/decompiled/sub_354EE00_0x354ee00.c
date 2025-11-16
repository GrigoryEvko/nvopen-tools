// Function: sub_354EE00
// Address: 0x354ee00
//
_QWORD *__fastcall sub_354EE00(__int64 a1, int a2)
{
  __int64 v3; // rbx
  __int64 v4; // r13
  unsigned int v5; // eax
  _QWORD *result; // rax
  __int64 v7; // r14
  _QWORD *i; // rdx
  _DWORD *v9; // rbx
  __int64 v10; // rdx
  _DWORD *v11; // rsi
  int v12; // eax
  int v13; // edi
  __int64 v14; // r8
  int v15; // r11d
  __int64 *v16; // r10
  unsigned int v17; // eax
  __int64 *v18; // rcx
  __int64 v19; // r9
  int *v20; // rdi
  _DWORD *v21; // r8
  _DWORD *v22; // r10
  _DWORD *v23; // r9
  int *v24; // rdx
  int *v25; // rax
  unsigned int v26; // eax
  int v27; // eax
  _DWORD *v28; // rcx
  __int64 v29; // r11
  __int64 v30; // rax
  int v31; // edi
  int *v32; // rax
  int v33; // r8d
  int v34; // ecx
  __int64 v35; // rax
  int v36; // edx
  _QWORD *j; // rdx
  __int64 v38; // [rsp+8h] [rbp-38h]

  v3 = *(unsigned int *)(a1 + 24);
  v4 = *(_QWORD *)(a1 + 8);
  v5 = sub_AF1560((unsigned int)(a2 - 1));
  if ( v5 < 0x40 )
    v5 = 64;
  *(_DWORD *)(a1 + 24) = v5;
  result = (_QWORD *)sub_C7D670(40LL * v5, 8);
  *(_QWORD *)(a1 + 8) = result;
  if ( v4 )
  {
    *(_QWORD *)(a1 + 16) = 0;
    v38 = 40 * v3;
    v7 = v4 + 40 * v3;
    for ( i = &result[5 * *(unsigned int *)(a1 + 24)]; i != result; result += 5 )
    {
      if ( result )
        *result = -4096;
    }
    v9 = (_DWORD *)(v4 + 24);
    if ( v7 != v4 )
    {
      while ( 1 )
      {
        v10 = *((_QWORD *)v9 - 3);
        v11 = v9 - 6;
        if ( v10 != -8192 && v10 != -4096 )
          break;
LABEL_25:
        if ( (_DWORD *)v7 == v9 + 4 )
          return (_QWORD *)sub_C7D6A0(v4, v38, 8);
        v9 += 10;
      }
      v12 = *(_DWORD *)(a1 + 24);
      if ( !v12 )
      {
        MEMORY[0] = *((_QWORD *)v9 - 3);
        BUG();
      }
      v13 = v12 - 1;
      v14 = *(_QWORD *)(a1 + 8);
      v15 = 1;
      v16 = 0;
      v17 = (v12 - 1) & (((unsigned int)v10 >> 9) ^ ((unsigned int)v10 >> 4));
      v18 = (__int64 *)(v14 + 40LL * v17);
      v19 = *v18;
      if ( v10 != *v18 )
      {
        while ( v19 != -4096 )
        {
          if ( !v16 && v19 == -8192 )
            v16 = v18;
          v17 = v13 & (v15 + v17);
          v18 = (__int64 *)(v14 + 40LL * v17);
          v19 = *v18;
          if ( v10 == *v18 )
            goto LABEL_13;
          ++v15;
        }
        if ( v16 )
          v18 = v16;
      }
LABEL_13:
      v18[1] = 0;
      v20 = (int *)(v18 + 3);
      v21 = v18 + 1;
      *v18 = v10;
      v22 = v9 - 4;
      v23 = v18 + 3;
      v24 = (int *)(v18 + 5);
      v18[2] = 1;
      v25 = (int *)(v18 + 3);
      do
      {
        if ( v25 )
          *v25 = -1;
        ++v25;
      }
      while ( v25 != v24 );
      v26 = v11[4] & 0xFFFFFFFE;
      v11[4] = v18[2] & 0xFFFFFFFE | v11[4] & 1;
      *((_DWORD *)v18 + 4) = v26 | v18[2] & 1;
      v27 = *((_DWORD *)v18 + 5);
      *((_DWORD *)v18 + 5) = *(v9 - 1);
      *(v9 - 1) = v27;
      if ( (v18[2] & 1) != 0 )
      {
        v28 = v9;
        if ( (v11[4] & 1) != 0 )
        {
          v32 = v9;
          do
          {
            v33 = *v32;
            v34 = *v20++;
            ++v32;
            *(v20 - 1) = v33;
            *(v32 - 1) = v34;
          }
          while ( v20 != v24 );
          goto LABEL_23;
        }
      }
      else
      {
        if ( (v11[4] & 1) == 0 )
        {
          v35 = v18[3];
          v18[3] = *(_QWORD *)v9;
          v36 = v9[2];
          *(_QWORD *)v9 = v35;
          LODWORD(v35) = *((_DWORD *)v18 + 8);
          *((_DWORD *)v18 + 8) = v36;
          v9[2] = v35;
LABEL_23:
          ++*(_DWORD *)(a1 + 16);
          if ( (v11[4] & 1) == 0 )
            sub_C7D6A0(*(_QWORD *)v9, 4LL * (unsigned int)v9[2], 4);
          goto LABEL_25;
        }
        v28 = v18 + 3;
        v22 = v21;
        v23 = v9;
        v21 = v9 - 4;
      }
      *((_BYTE *)v22 + 8) |= 1u;
      v29 = *((_QWORD *)v22 + 2);
      v30 = 0;
      v31 = v22[6];
      do
      {
        v28[v30] = v23[v30];
        ++v30;
      }
      while ( v30 != 4 );
      *((_BYTE *)v21 + 8) &= ~1u;
      *((_QWORD *)v21 + 2) = v29;
      v21[6] = v31;
      goto LABEL_23;
    }
    return (_QWORD *)sub_C7D6A0(v4, v38, 8);
  }
  else
  {
    *(_QWORD *)(a1 + 16) = 0;
    for ( j = &result[5 * *(unsigned int *)(a1 + 24)]; j != result; result += 5 )
    {
      if ( result )
        *result = -4096;
    }
  }
  return result;
}
