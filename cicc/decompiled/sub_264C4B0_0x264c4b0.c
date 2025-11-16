// Function: sub_264C4B0
// Address: 0x264c4b0
//
__int64 __fastcall sub_264C4B0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  int *v5; // rbx
  __int64 result; // rax
  int *v7; // r12
  int v9; // edx
  __int64 v10; // r8
  int v11; // ecx
  unsigned int v12; // edx
  int *v13; // rdi
  int v14; // r9d
  int v15; // edi
  int v16; // r10d
  _BYTE v18[96]; // [rsp+10h] [rbp-60h] BYREF

  v5 = *(int **)(a2 + 8);
  result = *(unsigned int *)(a2 + 16);
  v7 = &v5[*(unsigned int *)(a2 + 24)];
  if ( (_DWORD)result && v7 != v5 )
  {
    while ( (unsigned int)*v5 > 0xFFFFFFFD )
    {
      if ( v7 == ++v5 )
        return result;
    }
    if ( v5 != v7 )
    {
      v9 = *(_DWORD *)(a1 + 24);
      v10 = *(_QWORD *)(a1 + 8);
      if ( !v9 )
        goto LABEL_17;
LABEL_9:
      v11 = v9 - 1;
      v12 = (v9 - 1) & (37 * *v5);
      v13 = (int *)(v10 + 4LL * (v11 & (unsigned int)(37 * *v5)));
      v14 = *v13;
      if ( *v5 == *v13 )
      {
LABEL_10:
        *v13 = -2;
        --*(_DWORD *)(a1 + 16);
        ++*(_DWORD *)(a1 + 20);
        result = sub_22B6470((__int64)v18, a3, v5);
        goto LABEL_11;
      }
      v15 = 1;
      while ( v14 != -1 )
      {
        v16 = v15 + 1;
        v12 = v11 & (v15 + v12);
        v13 = (int *)(v10 + 4LL * v12);
        v14 = *v13;
        if ( *v5 == *v13 )
          goto LABEL_10;
        v15 = v16;
      }
LABEL_17:
      while ( 1 )
      {
        result = sub_22B6470((__int64)v18, a4, v5);
LABEL_11:
        if ( ++v5 == v7 )
          break;
        while ( (unsigned int)*v5 > 0xFFFFFFFD )
        {
          if ( v7 == ++v5 )
            return result;
        }
        if ( v5 == v7 )
          break;
        v9 = *(_DWORD *)(a1 + 24);
        v10 = *(_QWORD *)(a1 + 8);
        if ( v9 )
          goto LABEL_9;
      }
    }
  }
  return result;
}
