// Function: sub_1DF39E0
// Address: 0x1df39e0
//
__int64 __fastcall sub_1DF39E0(__int64 a1, unsigned int **a2, __int64 a3)
{
  unsigned int *v3; // r11
  __int64 result; // rax
  __int64 v5; // r12
  __int64 v7; // r8
  unsigned __int16 *v8; // rcx
  unsigned __int16 *v9; // rdx
  int v10; // ecx
  int v11; // ecx
  __int64 v12; // r10
  unsigned int v13; // r9d
  int *v14; // rax
  int v15; // r14d
  int v16; // eax
  int v17; // r13d

  v3 = *a2;
  result = *((unsigned int *)a2 + 2);
  v5 = (__int64)&(*a2)[result];
  if ( (unsigned int *)v5 != *a2 )
  {
    do
    {
      v7 = *v3;
      result = *(_QWORD *)(a3 + 56);
      v8 = (unsigned __int16 *)(result + 2LL * *(unsigned int *)(*(_QWORD *)(a3 + 8) + 24 * v7 + 4));
LABEL_5:
      v9 = v8;
      while ( v9 )
      {
        v10 = *(_DWORD *)(a1 + 24);
        if ( v10 )
        {
          v11 = v10 - 1;
          v12 = *(_QWORD *)(a1 + 8);
          v13 = v11 & (37 * (unsigned __int16)v7);
          v14 = (int *)(v12 + 16LL * v13);
          v15 = *v14;
          if ( (unsigned __int16)v7 == *v14 )
          {
LABEL_3:
            *v14 = -2;
            --*(_DWORD *)(a1 + 16);
            ++*(_DWORD *)(a1 + 20);
          }
          else
          {
            v16 = 1;
            while ( v15 != -1 )
            {
              v17 = v16 + 1;
              v13 = v11 & (v16 + v13);
              v14 = (int *)(v12 + 16LL * v13);
              v15 = *v14;
              if ( (unsigned __int16)v7 == *v14 )
                goto LABEL_3;
              v16 = v17;
            }
          }
        }
        result = *v9;
        v8 = 0;
        ++v9;
        LOWORD(v7) = result + v7;
        if ( !(_WORD)result )
          goto LABEL_5;
      }
      ++v3;
    }
    while ( v3 != (unsigned int *)v5 );
  }
  return result;
}
