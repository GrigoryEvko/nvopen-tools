// Function: sub_264FCD0
// Address: 0x264fcd0
//
__int64 __fastcall sub_264FCD0(__int64 a1, __int64 a2, __int64 a3)
{
  int *v5; // rbx
  int *v7; // r12
  int v8; // eax
  __int64 v9; // rsi
  int v10; // edx
  unsigned int v11; // eax
  int v12; // edi
  int v13; // r8d
  _BYTE v14[96]; // [rsp+0h] [rbp-60h] BYREF

  *(_QWORD *)a1 = 0;
  *(_QWORD *)(a1 + 8) = 0;
  *(_QWORD *)(a1 + 16) = 0;
  *(_DWORD *)(a1 + 24) = 0;
  if ( *(_DWORD *)(a2 + 16) )
  {
    v5 = *(int **)(a2 + 8);
    v7 = &v5[*(unsigned int *)(a2 + 24)];
    if ( v5 != v7 )
    {
      while ( (unsigned int)*v5 > 0xFFFFFFFD )
      {
        if ( v7 == ++v5 )
          return a1;
      }
      while ( v5 != v7 )
      {
        v8 = *(_DWORD *)(a3 + 24);
        v9 = *(_QWORD *)(a3 + 8);
        if ( v8 )
        {
          v10 = v8 - 1;
          v11 = (v8 - 1) & (37 * *v5);
          v12 = *(_DWORD *)(v9 + 4LL * (v10 & (unsigned int)(37 * *v5)));
          if ( v12 == *v5 )
          {
LABEL_10:
            sub_22B6470((__int64)v14, a1, v5);
          }
          else
          {
            v13 = 1;
            while ( v12 != -1 )
            {
              v11 = v10 & (v13 + v11);
              v12 = *(_DWORD *)(v9 + 4LL * v11);
              if ( *v5 == v12 )
                goto LABEL_10;
              ++v13;
            }
          }
        }
        if ( ++v5 == v7 )
          break;
        while ( (unsigned int)*v5 > 0xFFFFFFFD )
        {
          if ( v7 == ++v5 )
            return a1;
        }
      }
    }
  }
  return a1;
}
