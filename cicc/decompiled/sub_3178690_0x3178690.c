// Function: sub_3178690
// Address: 0x3178690
//
_QWORD *__fastcall sub_3178690(__int64 a1, int a2)
{
  unsigned __int64 v2; // rdx
  __int64 v4; // rbx
  __int64 v5; // r12
  unsigned int v6; // eax
  _QWORD *result; // rax
  __int64 v8; // rdx
  __int64 v9; // r13
  __int64 v10; // r15
  _QWORD *i; // rdx
  __int64 v12; // rbx
  __int64 v13; // rcx
  int v14; // eax
  int v15; // esi
  __int64 v16; // rdi
  int v17; // r10d
  __int64 *v18; // r9
  unsigned int v19; // edx
  __int64 *v20; // rax
  __int64 v21; // r8
  int v22; // edx
  __int64 v23; // rcx
  __int64 v24; // rdx
  __int64 v25; // rdx
  _QWORD *j; // rdx

  v2 = (unsigned int)(a2 - 1);
  v4 = *(unsigned int *)(a1 + 24);
  v5 = *(_QWORD *)(a1 + 8);
  v6 = (((((((((v2 | (v2 >> 1)) >> 2) | v2 | (v2 >> 1)) >> 4) | ((v2 | (v2 >> 1)) >> 2) | v2 | (v2 >> 1)) >> 8)
        | ((((v2 | (v2 >> 1)) >> 2) | v2 | (v2 >> 1)) >> 4)
        | ((v2 | (v2 >> 1)) >> 2)
        | v2
        | (v2 >> 1)) >> 16)
      | ((((((v2 | (v2 >> 1)) >> 2) | v2 | (v2 >> 1)) >> 4) | ((v2 | (v2 >> 1)) >> 2) | v2 | (v2 >> 1)) >> 8)
      | ((((v2 | (v2 >> 1)) >> 2) | v2 | (v2 >> 1)) >> 4)
      | ((v2 | (v2 >> 1)) >> 2)
      | v2
      | (v2 >> 1))
     + 1;
  if ( v6 < 0x40 )
    v6 = 64;
  *(_DWORD *)(a1 + 24) = v6;
  result = (_QWORD *)sub_C7D670(144LL * v6, 8);
  *(_QWORD *)(a1 + 8) = result;
  if ( v5 )
  {
    v8 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    v9 = 144 * v4;
    v10 = v5 + 144 * v4;
    for ( i = &result[18 * v8]; i != result; result += 18 )
    {
      if ( result )
        *result = -4096;
    }
    if ( v10 != v5 )
    {
      v12 = v5;
      do
      {
        v13 = *(_QWORD *)v12;
        if ( *(_QWORD *)v12 != -8192 && v13 != -4096 )
        {
          v14 = *(_DWORD *)(a1 + 24);
          if ( !v14 )
          {
            MEMORY[0] = *(_QWORD *)v12;
            BUG();
          }
          v15 = v14 - 1;
          v16 = *(_QWORD *)(a1 + 8);
          v17 = 1;
          v18 = 0;
          v19 = (v14 - 1) & (((unsigned int)v13 >> 9) ^ ((unsigned int)v13 >> 4));
          v20 = (__int64 *)(v16 + 144LL * v19);
          v21 = *v20;
          if ( v13 != *v20 )
          {
            while ( v21 != -4096 )
            {
              if ( !v18 && v21 == -8192 )
                v18 = v20;
              v19 = v15 & (v17 + v19);
              v20 = (__int64 *)(v16 + 144LL * v19);
              v21 = *v20;
              if ( v13 == *v20 )
                goto LABEL_14;
              ++v17;
            }
            if ( v18 )
              v20 = v18;
          }
LABEL_14:
          *v20 = v13;
          *((_BYTE *)v20 + 8) = *(_BYTE *)(v12 + 8);
          *((_BYTE *)v20 + 9) = *(_BYTE *)(v12 + 9);
          *((_BYTE *)v20 + 10) = *(_BYTE *)(v12 + 10);
          *((_DWORD *)v20 + 3) = *(_DWORD *)(v12 + 12);
          *((_BYTE *)v20 + 16) = *(_BYTE *)(v12 + 16);
          *(__m128i *)(v20 + 3) = _mm_loadu_si128((const __m128i *)(v12 + 24));
          *(__m128i *)(v20 + 5) = _mm_loadu_si128((const __m128i *)(v12 + 40));
          v22 = *(_DWORD *)(v12 + 56);
          v20[10] = 0;
          v20[9] = 0;
          *((_DWORD *)v20 + 14) = v22;
          v20[8] = 1;
          *((_DWORD *)v20 + 22) = 0;
          v23 = *(_QWORD *)(v12 + 72);
          ++*(_QWORD *)(v12 + 64);
          v24 = v20[9];
          v20[9] = v23;
          LODWORD(v23) = *(_DWORD *)(v12 + 80);
          *(_QWORD *)(v12 + 72) = v24;
          LODWORD(v24) = *((_DWORD *)v20 + 20);
          *((_DWORD *)v20 + 20) = v23;
          *(_DWORD *)(v12 + 80) = v24;
          LODWORD(v24) = *((_DWORD *)v20 + 21);
          *((_DWORD *)v20 + 21) = *(_DWORD *)(v12 + 84);
          LODWORD(v23) = *(_DWORD *)(v12 + 88);
          *(_DWORD *)(v12 + 84) = v24;
          LODWORD(v24) = *((_DWORD *)v20 + 22);
          *((_DWORD *)v20 + 22) = v23;
          *(_DWORD *)(v12 + 88) = v24;
          *((_DWORD *)v20 + 24) = *(_DWORD *)(v12 + 96);
          *((_DWORD *)v20 + 25) = *(_DWORD *)(v12 + 100);
          *((_DWORD *)v20 + 26) = *(_DWORD *)(v12 + 104);
          *((_DWORD *)v20 + 27) = *(_DWORD *)(v12 + 108);
          *((_DWORD *)v20 + 28) = *(_DWORD *)(v12 + 112);
          *((_DWORD *)v20 + 29) = *(_DWORD *)(v12 + 116);
          *((_DWORD *)v20 + 30) = *(_DWORD *)(v12 + 120);
          *((_DWORD *)v20 + 31) = *(_DWORD *)(v12 + 124);
          *((_DWORD *)v20 + 32) = *(_DWORD *)(v12 + 128);
          *((_DWORD *)v20 + 33) = *(_DWORD *)(v12 + 132);
          *((_DWORD *)v20 + 34) = *(_DWORD *)(v12 + 136);
          *((_DWORD *)v20 + 35) = *(_DWORD *)(v12 + 140);
          ++*(_DWORD *)(a1 + 16);
          sub_C7D6A0(*(_QWORD *)(v12 + 72), 24LL * *(unsigned int *)(v12 + 88), 8);
        }
        v12 += 144;
      }
      while ( v10 != v12 );
    }
    return (_QWORD *)sub_C7D6A0(v5, v9, 8);
  }
  else
  {
    v25 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    for ( j = &result[18 * v25]; j != result; result += 18 )
    {
      if ( result )
        *result = -4096;
    }
  }
  return result;
}
