// Function: sub_2AD96C0
// Address: 0x2ad96c0
//
__int64 __fastcall sub_2AD96C0(__int64 a1)
{
  __int64 v1; // rbx
  __int64 result; // rax
  __int64 i; // r14
  unsigned int v5; // esi
  __int64 v6; // r10
  int v7; // r15d
  char v8; // r9
  int *v9; // rdi
  int *v10; // r8
  int v11; // ecx
  int v12; // eax
  int v13; // edx
  int v14; // eax
  int v15; // eax
  _QWORD v16[7]; // [rsp+8h] [rbp-38h] BYREF

  v1 = *(_QWORD *)(a1 + 32);
  result = *(unsigned int *)(a1 + 40);
  for ( i = v1 + 8 * result; i != v1; *((_BYTE *)v9 + 4) = result )
  {
LABEL_2:
    v5 = *(_DWORD *)(a1 + 24);
    if ( v5 )
    {
      v6 = *(_QWORD *)(a1 + 8);
      v7 = 1;
      v8 = *(_BYTE *)(v1 + 4);
      v9 = 0;
      for ( result = (v5 - 1) & ((v8 == 0) + 37 * *(_DWORD *)v1 - 1); ; result = (v5 - 1) & v12 )
      {
        v10 = (int *)(v6 + 8LL * (unsigned int)result);
        v11 = *v10;
        if ( *(_DWORD *)v1 == *v10 && v8 == *((_BYTE *)v10 + 4) )
        {
          v1 += 8;
          if ( i != v1 )
            goto LABEL_2;
          return result;
        }
        if ( v11 == -1 )
        {
          if ( *((_BYTE *)v10 + 4) )
          {
            v15 = *(_DWORD *)(a1 + 16);
            if ( !v9 )
              v9 = v10;
            ++*(_QWORD *)a1;
            v13 = v15 + 1;
            v16[0] = v9;
            if ( 4 * (v15 + 1) >= 3 * v5 )
              goto LABEL_12;
            if ( v5 - *(_DWORD *)(a1 + 20) - v13 > v5 >> 3 )
              goto LABEL_14;
            goto LABEL_13;
          }
        }
        else if ( v11 == -2 && *((_BYTE *)v10 + 4) != 1 && !v9 )
        {
          v9 = (int *)(v6 + 8LL * (unsigned int)result);
        }
        v12 = v7 + result;
        ++v7;
      }
    }
    ++*(_QWORD *)a1;
    v16[0] = 0;
LABEL_12:
    v5 *= 2;
LABEL_13:
    sub_2AD9490(a1, v5);
    sub_2AC3BB0(a1, (int *)v1, v16);
    v9 = (int *)v16[0];
    v13 = *(_DWORD *)(a1 + 16) + 1;
LABEL_14:
    *(_DWORD *)(a1 + 16) = v13;
    if ( *v9 != -1 || !*((_BYTE *)v9 + 4) )
      --*(_DWORD *)(a1 + 20);
    v14 = *(_DWORD *)v1;
    v1 += 8;
    *v9 = v14;
    result = *(unsigned __int8 *)(v1 - 4);
  }
  return result;
}
