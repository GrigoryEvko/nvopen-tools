// Function: sub_13B3EF0
// Address: 0x13b3ef0
//
char *__fastcall sub_13B3EF0(__int64 a1, char *a2, __int64 a3, __int64 a4)
{
  char *result; // rax
  int v7; // r13d
  char *v8; // r14
  unsigned int i; // r15d
  __int64 v10; // r8
  int v11; // r11d
  char **v12; // r10
  unsigned int v13; // edx
  char **v14; // rdi
  char *v15; // rcx
  unsigned int v16; // esi
  int v17; // esi
  int v18; // esi
  __int64 v19; // r8
  int v20; // edx
  unsigned int v21; // ecx
  int v22; // r9d
  char **v23; // r11
  int v24; // edx
  _BYTE *v25; // rsi
  int v26; // esi
  int v27; // esi
  __int64 v28; // r8
  char **v29; // r11
  int v30; // r9d
  unsigned int v31; // ecx
  char *v32; // rdi
  char *v34; // [rsp+28h] [rbp-38h] BYREF

  result = (char *)sub_157EBA0(a1);
  if ( result )
  {
    v7 = sub_15F4D60(result);
    result = (char *)sub_157EBA0(a1);
    v8 = result;
    if ( v7 )
    {
      for ( i = 0; v7 != i; ++i )
      {
        result = (char *)sub_15F4DF0(v8, i);
        v34 = result;
        if ( a2 != result )
        {
          v16 = *(_DWORD *)(a3 + 24);
          if ( !v16 )
          {
            ++*(_QWORD *)a3;
            goto LABEL_9;
          }
          v10 = *(_QWORD *)(a3 + 8);
          v11 = 1;
          v12 = 0;
          v13 = (v16 - 1) & (((unsigned int)result >> 9) ^ ((unsigned int)result >> 4));
          v14 = (char **)(v10 + 8LL * v13);
          v15 = *v14;
          if ( result != *v14 )
          {
            while ( v15 != (char *)-8LL )
            {
              if ( v15 != (char *)-16LL || v12 )
                v14 = v12;
              v13 = (v16 - 1) & (v11 + v13);
              v15 = *(char **)(v10 + 8LL * v13);
              if ( result == v15 )
                goto LABEL_5;
              ++v11;
              v12 = v14;
              v14 = (char **)(v10 + 8LL * v13);
            }
            v24 = *(_DWORD *)(a3 + 16);
            if ( !v12 )
              v12 = v14;
            ++*(_QWORD *)a3;
            v20 = v24 + 1;
            if ( 4 * v20 >= 3 * v16 )
            {
LABEL_9:
              sub_13B3D40(a3, 2 * v16);
              v17 = *(_DWORD *)(a3 + 24);
              if ( !v17 )
                goto LABEL_50;
              v18 = v17 - 1;
              v19 = *(_QWORD *)(a3 + 8);
              v20 = *(_DWORD *)(a3 + 16) + 1;
              v21 = v18 & (((unsigned int)v34 >> 9) ^ ((unsigned int)v34 >> 4));
              v12 = (char **)(v19 + 8LL * v21);
              result = *v12;
              if ( v34 != *v12 )
              {
                v22 = 1;
                v23 = 0;
                while ( result != (char *)-8LL )
                {
                  if ( result == (char *)-16LL && !v23 )
                    v23 = v12;
                  v21 = v18 & (v22 + v21);
                  v12 = (char **)(v19 + 8LL * v21);
                  result = *v12;
                  if ( v34 == *v12 )
                    goto LABEL_26;
                  ++v22;
                }
                result = v34;
                if ( v23 )
                  v12 = v23;
              }
            }
            else if ( v16 - *(_DWORD *)(a3 + 20) - v20 <= v16 >> 3 )
            {
              sub_13B3D40(a3, v16);
              v26 = *(_DWORD *)(a3 + 24);
              if ( !v26 )
              {
LABEL_50:
                ++*(_DWORD *)(a3 + 16);
                BUG();
              }
              result = v34;
              v27 = v26 - 1;
              v28 = *(_QWORD *)(a3 + 8);
              v29 = 0;
              v30 = 1;
              v31 = v27 & (((unsigned int)v34 >> 9) ^ ((unsigned int)v34 >> 4));
              v12 = (char **)(v28 + 8LL * v31);
              v32 = *v12;
              v20 = *(_DWORD *)(a3 + 16) + 1;
              if ( *v12 != v34 )
              {
                while ( v32 != (char *)-8LL )
                {
                  if ( !v29 && v32 == (char *)-16LL )
                    v29 = v12;
                  v31 = v27 & (v30 + v31);
                  v12 = (char **)(v28 + 8LL * v31);
                  v32 = *v12;
                  if ( v34 == *v12 )
                    goto LABEL_26;
                  ++v30;
                }
                if ( v29 )
                  v12 = v29;
              }
            }
LABEL_26:
            *(_DWORD *)(a3 + 16) = v20;
            if ( *v12 != (char *)-8LL )
              --*(_DWORD *)(a3 + 20);
            *v12 = result;
            v25 = *(_BYTE **)(a4 + 8);
            if ( v25 == *(_BYTE **)(a4 + 16) )
            {
              result = sub_1292090(a4, v25, &v34);
            }
            else
            {
              if ( v25 )
              {
                *(_QWORD *)v25 = v34;
                v25 = *(_BYTE **)(a4 + 8);
              }
              result = (char *)a4;
              *(_QWORD *)(a4 + 8) = v25 + 8;
            }
          }
        }
LABEL_5:
        ;
      }
    }
  }
  return result;
}
