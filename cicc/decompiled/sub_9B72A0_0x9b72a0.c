// Function: sub_9B72A0
// Address: 0x9b72a0
//
__int64 __fastcall sub_9B72A0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  _QWORD *v6; // rsi
  _QWORD *v7; // rdx
  _QWORD *v8; // rax
  __int64 v9; // rcx
  __int64 v10; // rdi
  unsigned int v11; // r8d
  unsigned int v12; // ecx
  unsigned int v13; // edx
  unsigned int v14; // esi
  int *v15; // rax
  int v16; // r9d
  __int64 v17; // rdx
  int v18; // esi
  __int64 v19; // r9
  int v20; // esi
  unsigned int v21; // r10d
  __int64 *v22; // rax
  __int64 v23; // r11
  int v25; // eax
  int v26; // r11d
  int v27; // eax
  int v28; // r13d
  __int64 v29; // rax
  _QWORD *v30; // rax

  if ( *(_BYTE *)(a1 + 108) )
  {
    v6 = *(_QWORD **)(a1 + 88);
    v7 = &v6[*(unsigned int *)(a1 + 100)];
    v8 = v6;
    if ( v6 != v7 )
    {
      while ( a2 != *v8 )
      {
        if ( v7 == ++v8 )
          goto LABEL_7;
      }
      v9 = (unsigned int)(*(_DWORD *)(a1 + 100) - 1);
      *(_DWORD *)(a1 + 100) = v9;
      *v8 = v6[v9];
      ++*(_QWORD *)(a1 + 80);
    }
  }
  else
  {
    v30 = (_QWORD *)sub_C8CA60(a1 + 80, a2, a3, a4);
    if ( v30 )
    {
      *v30 = -2;
      ++*(_DWORD *)(a1 + 104);
      ++*(_QWORD *)(a1 + 80);
    }
  }
LABEL_7:
  v10 = *(_QWORD *)(a2 + 16);
  v11 = *(_DWORD *)(a2 + 32);
  if ( *(_DWORD *)a2 )
  {
    v12 = 0;
    do
    {
      v13 = v12 + *(_DWORD *)(a2 + 40);
      if ( v11 )
      {
        v14 = (v11 - 1) & (37 * v13);
        v15 = (int *)(v10 + 16LL * v14);
        v16 = *v15;
        if ( v13 == *v15 )
        {
LABEL_11:
          v17 = *((_QWORD *)v15 + 1);
          if ( v17 )
          {
            v18 = *(_DWORD *)(a1 + 72);
            v19 = *(_QWORD *)(a1 + 56);
            if ( v18 )
            {
              v20 = v18 - 1;
              v21 = v20 & (((unsigned int)v17 >> 9) ^ ((unsigned int)v17 >> 4));
              v22 = (__int64 *)(v19 + 16LL * v21);
              v23 = *v22;
              if ( *v22 == v17 )
              {
LABEL_14:
                *v22 = -8192;
                --*(_DWORD *)(a1 + 64);
                ++*(_DWORD *)(a1 + 68);
                v11 = *(_DWORD *)(a2 + 32);
                v10 = *(_QWORD *)(a2 + 16);
              }
              else
              {
                v27 = 1;
                while ( v23 != -4096 )
                {
                  v28 = v27 + 1;
                  v29 = v20 & (v21 + v27);
                  v21 = v29;
                  v22 = (__int64 *)(v19 + 16 * v29);
                  v23 = *v22;
                  if ( v17 == *v22 )
                    goto LABEL_14;
                  v27 = v28;
                }
              }
            }
          }
        }
        else
        {
          v25 = 1;
          while ( v16 != 0x7FFFFFFF )
          {
            v26 = v25 + 1;
            v14 = (v11 - 1) & (v25 + v14);
            v15 = (int *)(v10 + 16LL * v14);
            v16 = *v15;
            if ( v13 == *v15 )
              goto LABEL_11;
            v25 = v26;
          }
        }
      }
      ++v12;
    }
    while ( *(_DWORD *)a2 > v12 );
  }
  sub_C7D6A0(v10, 16LL * v11, 8);
  return j_j___libc_free_0(a2, 56);
}
