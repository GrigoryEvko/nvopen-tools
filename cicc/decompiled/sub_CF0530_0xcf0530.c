// Function: sub_CF0530
// Address: 0xcf0530
//
__int64 __fastcall sub_CF0530(__int64 a1, _QWORD *a2)
{
  __int64 result; // rax
  unsigned int v5; // ebx
  unsigned int v6; // r14d
  __int64 v7; // r9
  unsigned int v8; // r8d
  __int64 v9; // r13
  __int64 v10; // rax
  unsigned int v11; // r10d
  _BYTE *v12; // rdx
  unsigned int v13; // ecx
  _QWORD *v14; // rsi
  _BYTE *v15; // rdi
  int v16; // esi
  unsigned int v17; // r8d
  unsigned int v18; // eax
  __int64 *v19; // rdx
  __int64 v20; // rcx
  __int64 *v21; // rax
  __int64 *v22; // rdx
  __int64 v23; // rcx
  int v24; // edx
  int v25; // esi
  int v26; // [rsp+0h] [rbp-3Ch]
  unsigned int v27; // [rsp+4h] [rbp-38h]
  int v28; // [rsp+8h] [rbp-34h]

  result = (__int64)(a2[1] - *a2) >> 3;
  v28 = result;
  if ( (unsigned int)result > 1 )
  {
    v27 = result - 1;
    result = 0;
    do
    {
      v5 = result;
      v6 = result;
      if ( v28 != (_DWORD)result )
      {
        do
        {
          v7 = *(_QWORD *)(a1 + 8);
          v8 = *(_DWORD *)(a1 + 24);
          v9 = *(_QWORD *)(*a2 + 8LL * v5);
          v10 = *(_QWORD *)(v9 + 16);
          if ( v10 )
          {
            v11 = v8 - 1;
            do
            {
              v12 = *(_BYTE **)(v10 + 24);
              if ( *v12 > 0x1Cu && v8 )
              {
                v13 = v11 & (((unsigned int)v12 >> 9) ^ ((unsigned int)v12 >> 4));
                v14 = (_QWORD *)(v7 + 8LL * v13);
                v15 = (_BYTE *)*v14;
                if ( v12 == (_BYTE *)*v14 )
                {
LABEL_6:
                  if ( (_QWORD *)(v7 + 8LL * v8) != v14 )
                    goto LABEL_21;
                }
                else
                {
                  v16 = 1;
                  while ( v15 != (_BYTE *)-4096LL )
                  {
                    v13 = v11 & (v16 + v13);
                    v26 = v16 + 1;
                    v14 = (_QWORD *)(v7 + 8LL * v13);
                    v15 = (_BYTE *)*v14;
                    if ( v12 == (_BYTE *)*v14 )
                      goto LABEL_6;
                    v16 = v26;
                  }
                }
              }
              v10 = *(_QWORD *)(v10 + 8);
            }
            while ( v10 );
          }
          if ( v8 )
          {
            v17 = v8 - 1;
            v18 = v17 & (((unsigned int)v9 >> 9) ^ ((unsigned int)v9 >> 4));
            v19 = (__int64 *)(v7 + 8LL * v18);
            v20 = *v19;
            if ( v9 == *v19 )
            {
LABEL_17:
              *v19 = -8192;
              --*(_DWORD *)(a1 + 16);
              ++*(_DWORD *)(a1 + 20);
            }
            else
            {
              v24 = 1;
              while ( v20 != -4096 )
              {
                v25 = v24 + 1;
                v18 = v17 & (v24 + v18);
                v19 = (__int64 *)(v7 + 8LL * v18);
                v20 = *v19;
                if ( v9 == *v19 )
                  goto LABEL_17;
                v24 = v25;
              }
            }
          }
          if ( v6 != v5 )
          {
            v21 = (__int64 *)(*a2 + 8LL * v5);
            v22 = (__int64 *)(*a2 + 8LL * v6);
            v23 = *v22;
            *v22 = *v21;
            *v21 = v23;
          }
          ++v6;
LABEL_21:
          ++v5;
        }
        while ( v28 != v5 );
        result = v6;
      }
    }
    while ( (unsigned int)result < v27 );
  }
  return result;
}
