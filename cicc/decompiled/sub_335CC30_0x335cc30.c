// Function: sub_335CC30
// Address: 0x335cc30
//
__int64 __fastcall sub_335CC30(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, unsigned int a6)
{
  __int64 result; // rax
  __int64 *v7; // r15
  __int64 v8; // rsi
  unsigned int v10; // edx
  __int64 v11; // rcx
  __int64 v12; // r8
  __int64 *v13; // rbx
  __int64 v14; // rsi
  unsigned int v15; // r14d
  __int64 v16; // r9
  unsigned __int64 v17; // rsi
  int v18; // r8d
  __int64 v19; // r11
  int v20; // r13d
  unsigned int i; // edx
  __int64 v22; // r10
  unsigned int v23; // edx
  __int64 v24; // r8
  __int64 v25; // r9
  __int64 v26; // r13
  unsigned __int64 v27; // r14
  __int64 v28; // rax
  unsigned __int64 *v29; // rax
  __int64 v30; // rdx
  __int64 v31; // rax
  int v32; // ecx
  int v33; // r10d
  __int64 v34; // [rsp+10h] [rbp-60h]
  unsigned __int64 v36; // [rsp+20h] [rbp-50h]
  __int64 v39; // [rsp+38h] [rbp-38h]

  result = *(unsigned int *)(a2 + 712);
  v7 = *(__int64 **)(a3 + 48);
  v8 = *(_QWORD *)(a2 + 696);
  v34 = *(_QWORD *)(a3 + 40);
  if ( (_DWORD)result )
  {
    v10 = (result - 1) & (((unsigned int)a1 >> 9) ^ ((unsigned int)a1 >> 4));
    v11 = v8 + 40LL * v10;
    v12 = *(_QWORD *)v11;
    if ( a1 == *(_QWORD *)v11 )
    {
LABEL_3:
      result = v8 + 40 * result;
      if ( v11 != result )
      {
        v13 = *(__int64 **)(v11 + 8);
        result = (__int64)&v13[*(unsigned int *)(v11 + 16)];
        v39 = result;
        if ( (__int64 *)result != v13 )
        {
          result = a5 + 16;
          do
          {
            v14 = *v13;
            if ( !*(_BYTE *)(*v13 + 63) )
            {
              result = a6;
              v15 = *(_DWORD *)(v14 + 56);
              if ( !a6 || v15 == a6 )
              {
                if ( !*(_BYTE *)(v14 + 62) )
                {
                  result = *(_QWORD *)(v14 + 8);
                  v16 = result + 24LL * *(_QWORD *)v14;
                  if ( result != v16 )
                  {
                    do
                    {
                      if ( !*(_DWORD *)result )
                      {
                        v17 = *(_QWORD *)(result + 8);
                        v18 = *(_DWORD *)(result + 16);
                        if ( (*(_BYTE *)(a5 + 8) & 1) != 0 )
                        {
                          v19 = a5 + 16;
                          v11 = 15;
                        }
                        else
                        {
                          v11 = *(unsigned int *)(a5 + 24);
                          v19 = *(_QWORD *)(a5 + 16);
                          if ( !(_DWORD)v11 )
                            goto LABEL_25;
                          v11 = (unsigned int)(v11 - 1);
                        }
                        v20 = 1;
                        for ( i = v11 & (v18 + ((v17 >> 9) ^ (v17 >> 4))); ; i = v11 & v23 )
                        {
                          v22 = v19 + 24LL * i;
                          if ( v17 == *(_QWORD *)v22 && v18 == *(_DWORD *)(v22 + 8) )
                            break;
                          if ( !*(_QWORD *)v22 && *(_DWORD *)(v22 + 8) == -1 )
                            goto LABEL_25;
                          v23 = v20 + i;
                          ++v20;
                        }
                      }
                      result += 24;
                    }
                    while ( v16 != result );
                    v14 = *v13;
                  }
                }
                result = sub_37547E0(a3, v14, a5, v11);
                v26 = result;
                if ( result )
                {
                  v27 = v15 | v36 & 0xFFFFFFFF00000000LL;
                  v28 = *(unsigned int *)(a4 + 8);
                  v36 = v27;
                  if ( v28 + 1 > (unsigned __int64)*(unsigned int *)(a4 + 12) )
                  {
                    sub_C8D5F0(a4, (const void *)(a4 + 16), v28 + 1, 0x10u, v24, v25);
                    v28 = *(unsigned int *)(a4 + 8);
                  }
                  v29 = (unsigned __int64 *)(*(_QWORD *)a4 + 16 * v28);
                  v29[1] = v26;
                  *v29 = v27;
                  ++*(_DWORD *)(a4 + 8);
                  sub_2E31040((__int64 *)(v34 + 40), v26);
                  v30 = *v7;
                  v31 = *(_QWORD *)v26;
                  *(_QWORD *)(v26 + 8) = v7;
                  v30 &= 0xFFFFFFFFFFFFFFF8LL;
                  *(_QWORD *)v26 = v30 | v31 & 7;
                  *(_QWORD *)(v30 + 8) = v26;
                  result = *v7 & 7;
                  *v7 = result | v26;
                }
              }
            }
LABEL_25:
            ++v13;
          }
          while ( (__int64 *)v39 != v13 );
        }
      }
    }
    else
    {
      v32 = 1;
      while ( v12 != -4096 )
      {
        v33 = v32 + 1;
        v10 = (result - 1) & (v32 + v10);
        v11 = v8 + 40LL * v10;
        v12 = *(_QWORD *)v11;
        if ( a1 == *(_QWORD *)v11 )
          goto LABEL_3;
        v32 = v33;
      }
    }
  }
  return result;
}
