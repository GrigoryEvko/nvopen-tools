// Function: sub_D68390
// Address: 0xd68390
//
_QWORD *__fastcall sub_D68390(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v5; // rbx
  bool v6; // zf
  _QWORD *result; // rax
  __int64 v8; // rdx
  _QWORD *i; // rdx
  __int64 v10; // rdi
  __int64 v11; // r8
  __int64 v12; // r9
  unsigned int v13; // edx
  __int64 v14; // rsi
  __int64 v15; // r12
  __int64 v16; // rcx
  __int64 v17; // rdx
  __int64 v18; // rax
  __int64 v19; // rdx
  __int64 v20; // rdx
  __int64 v21; // rax
  __int64 v22; // rdx
  __int64 v23; // rdi
  __int64 v24; // rsi
  __int64 v25; // rdi
  int v26; // edx
  unsigned int v27; // r10d

  v5 = a2;
  v6 = (*(_QWORD *)(a1 + 8) & 1LL) == 0;
  *(_QWORD *)(a1 + 8) &= 1uLL;
  if ( v6 )
  {
    result = *(_QWORD **)(a1 + 16);
    v8 = 17LL * *(unsigned int *)(a1 + 24);
  }
  else
  {
    result = (_QWORD *)(a1 + 16);
    v8 = 68;
  }
  for ( i = &result[v8]; i != result; result += 17 )
  {
    if ( result )
      *result = -4096;
  }
  if ( a2 != a3 )
  {
    do
    {
      result = *(_QWORD **)v5;
      if ( *(_QWORD *)v5 != -4096 && result != (_QWORD *)-8192LL )
      {
        if ( (*(_BYTE *)(a1 + 8) & 1) != 0 )
        {
          v10 = a1 + 16;
          v11 = 3;
        }
        else
        {
          v26 = *(_DWORD *)(a1 + 24);
          v10 = *(_QWORD *)(a1 + 16);
          if ( !v26 )
          {
            MEMORY[0] = *(_QWORD *)v5;
            BUG();
          }
          v11 = (unsigned int)(v26 - 1);
        }
        v12 = 1;
        v13 = v11 & (((unsigned int)result >> 9) ^ ((unsigned int)result >> 4));
        v14 = 0;
        v15 = v10 + 136LL * v13;
        v16 = *(_QWORD *)v15;
        if ( result != *(_QWORD **)v15 )
        {
          while ( v16 != -4096 )
          {
            if ( v16 == -8192 && !v14 )
              v14 = v15;
            v27 = v12 + 1;
            v13 = v11 & (v12 + v13);
            v12 = v13;
            v15 = v10 + 136LL * v13;
            v16 = *(_QWORD *)v15;
            if ( result == *(_QWORD **)v15 )
              goto LABEL_11;
            v12 = v27;
          }
          if ( v14 )
            v15 = v14;
        }
LABEL_11:
        *(_QWORD *)(v15 + 24) = 0;
        *(_QWORD *)(v15 + 16) = 0;
        *(_DWORD *)(v15 + 32) = 0;
        *(_QWORD *)v15 = result;
        *(_QWORD *)(v15 + 8) = 1;
        v17 = *(_QWORD *)(v5 + 16);
        ++*(_QWORD *)(v5 + 8);
        v18 = *(_QWORD *)(v15 + 16);
        *(_QWORD *)(v15 + 16) = v17;
        LODWORD(v17) = *(_DWORD *)(v5 + 24);
        *(_QWORD *)(v5 + 16) = v18;
        LODWORD(v18) = *(_DWORD *)(v15 + 24);
        *(_DWORD *)(v15 + 24) = v17;
        LODWORD(v17) = *(_DWORD *)(v5 + 28);
        *(_DWORD *)(v5 + 24) = v18;
        LODWORD(v18) = *(_DWORD *)(v15 + 28);
        *(_DWORD *)(v15 + 28) = v17;
        LODWORD(v17) = *(_DWORD *)(v5 + 32);
        *(_DWORD *)(v5 + 28) = v18;
        LODWORD(v18) = *(_DWORD *)(v15 + 32);
        *(_DWORD *)(v15 + 32) = v17;
        *(_DWORD *)(v5 + 32) = v18;
        *(_QWORD *)(v15 + 40) = v15 + 56;
        *(_QWORD *)(v15 + 48) = 0x200000000LL;
        v19 = *(unsigned int *)(v5 + 48);
        if ( (_DWORD)v19 )
        {
          v14 = v5 + 40;
          sub_D67C10(v15 + 40, (char **)(v5 + 40), v19, v16, v11, v12);
        }
        *(_QWORD *)(v15 + 88) = 0;
        *(_QWORD *)(v15 + 80) = 0;
        *(_DWORD *)(v15 + 96) = 0;
        *(_QWORD *)(v15 + 72) = 1;
        v20 = *(_QWORD *)(v5 + 80);
        ++*(_QWORD *)(v5 + 72);
        v21 = *(_QWORD *)(v15 + 80);
        *(_QWORD *)(v15 + 80) = v20;
        LODWORD(v20) = *(_DWORD *)(v5 + 88);
        *(_QWORD *)(v5 + 80) = v21;
        LODWORD(v21) = *(_DWORD *)(v15 + 88);
        *(_DWORD *)(v15 + 88) = v20;
        LODWORD(v20) = *(_DWORD *)(v5 + 92);
        *(_DWORD *)(v5 + 88) = v21;
        LODWORD(v21) = *(_DWORD *)(v15 + 92);
        *(_DWORD *)(v15 + 92) = v20;
        v22 = *(unsigned int *)(v5 + 96);
        *(_DWORD *)(v5 + 92) = v21;
        LODWORD(v21) = *(_DWORD *)(v15 + 96);
        *(_DWORD *)(v15 + 96) = v22;
        *(_DWORD *)(v5 + 96) = v21;
        *(_QWORD *)(v15 + 104) = v15 + 120;
        *(_QWORD *)(v15 + 112) = 0x200000000LL;
        if ( *(_DWORD *)(v5 + 112) )
        {
          v14 = v5 + 104;
          sub_D67C10(v15 + 104, (char **)(v5 + 104), v22, v16, v11, v12);
        }
        *(_DWORD *)(a1 + 8) = (2 * (*(_DWORD *)(a1 + 8) >> 1) + 2) | *(_DWORD *)(a1 + 8) & 1;
        v23 = *(_QWORD *)(v5 + 104);
        if ( v23 != v5 + 120 )
          _libc_free(v23, v14);
        v24 = 8LL * *(unsigned int *)(v5 + 96);
        sub_C7D6A0(*(_QWORD *)(v5 + 80), v24, 8);
        v25 = *(_QWORD *)(v5 + 40);
        if ( v25 != v5 + 56 )
          _libc_free(v25, v24);
        result = (_QWORD *)sub_C7D6A0(*(_QWORD *)(v5 + 16), 8LL * *(unsigned int *)(v5 + 32), 8);
      }
      v5 += 136;
    }
    while ( a3 != v5 );
  }
  return result;
}
