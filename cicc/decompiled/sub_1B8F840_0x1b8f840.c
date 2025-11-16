// Function: sub_1B8F840
// Address: 0x1b8f840
//
__int64 __fastcall sub_1B8F840(__int64 a1, __int64 a2)
{
  __int64 v3; // r9
  unsigned int v5; // ecx
  __int64 v6; // rdx
  int v7; // r11d
  unsigned int v8; // esi
  unsigned int v9; // edi
  int *v10; // rax
  int v11; // r10d
  __int64 v12; // rdx
  int v13; // eax
  int v14; // edi
  __int64 v15; // r10
  unsigned int v16; // esi
  __int64 *v17; // rax
  __int64 v18; // r11
  int v20; // ebx
  int v21; // r13d
  int v22; // eax
  int v23; // r14d
  __int64 v24; // rax
  int v25; // eax
  int v26; // eax
  int v27; // ebx
  int v28; // esi

  v3 = *(_QWORD *)(a2 + 24);
  if ( *(_DWORD *)a2 )
  {
    v5 = 0;
    do
    {
      v6 = *(unsigned int *)(a2 + 40);
      if ( (_DWORD)v6 )
      {
        v7 = v6 - 1;
        v8 = v5 + *(_DWORD *)(a2 + 48);
        v9 = (v6 - 1) & (37 * v8);
        v10 = (int *)(v3 + 16LL * v9);
        v11 = *v10;
        if ( v8 == *v10 )
        {
LABEL_5:
          v12 = *((_QWORD *)v10 + 1);
          if ( v12 )
          {
            v13 = *(_DWORD *)(a1 + 72);
            if ( v13 )
            {
              v14 = v13 - 1;
              v15 = *(_QWORD *)(a1 + 56);
              v16 = (v13 - 1) & (((unsigned int)v12 >> 9) ^ ((unsigned int)v12 >> 4));
              v17 = (__int64 *)(v15 + 16LL * v16);
              v18 = *v17;
              if ( v12 == *v17 )
              {
LABEL_8:
                *v17 = -16;
                --*(_DWORD *)(a1 + 64);
                ++*(_DWORD *)(a1 + 68);
                v3 = *(_QWORD *)(a2 + 24);
              }
              else
              {
                v26 = 1;
                while ( v18 != -8 )
                {
                  v27 = v26 + 1;
                  v16 = v14 & (v26 + v16);
                  v17 = (__int64 *)(v15 + 16LL * v16);
                  v18 = *v17;
                  if ( v12 == *v17 )
                    goto LABEL_8;
                  v26 = v27;
                }
              }
            }
          }
        }
        else
        {
          v20 = *v10;
          v21 = (v6 - 1) & (37 * v8);
          v22 = 1;
          while ( v20 != 0x7FFFFFFF )
          {
            v23 = v22 + 1;
            v24 = v7 & (unsigned int)(v21 + v22);
            v21 = v24;
            v20 = *(_DWORD *)(v3 + 16 * v24);
            if ( v8 == v20 )
            {
              v25 = 1;
              while ( v11 != 0x7FFFFFFF )
              {
                v28 = v25 + 1;
                v9 = v7 & (v25 + v9);
                v10 = (int *)(v3 + 16LL * v9);
                v11 = *v10;
                if ( v20 == *v10 )
                  goto LABEL_5;
                v25 = v28;
              }
              v10 = (int *)(v3 + 16 * v6);
              goto LABEL_5;
            }
            v22 = v23;
          }
        }
      }
      ++v5;
    }
    while ( *(_DWORD *)a2 > v5 );
  }
  j___libc_free_0(v3);
  return j_j___libc_free_0(a2, 64);
}
