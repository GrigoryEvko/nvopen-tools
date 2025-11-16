// Function: sub_2649AE0
// Address: 0x2649ae0
//
_DWORD *__fastcall sub_2649AE0(__int64 a1, __int64 a2)
{
  unsigned int v3; // ecx
  _DWORD *result; // rax
  _DWORD *v5; // rdx
  int v6; // ecx
  __int64 v7; // r9
  int v8; // esi
  unsigned int v9; // ecx
  int *v10; // r8
  int v11; // r10d
  int v12; // r8d
  int v13; // r11d
  _DWORD *v14; // r9
  _DWORD *v15; // rdx
  _DWORD *v16; // r11
  __int64 v17; // rcx
  __int64 v18; // r10
  unsigned int v19; // edi
  int *v20; // r14
  int v21; // r8d
  int v22; // r14d
  int v23; // r15d
  __int64 v24; // [rsp+0h] [rbp-50h] BYREF
  _DWORD *v25; // [rsp+10h] [rbp-40h]
  _DWORD *v26; // [rsp+18h] [rbp-38h]

  v3 = *(_DWORD *)(a2 + 16);
  if ( *(_DWORD *)(a1 + 16) < v3 )
  {
    sub_22B0690(&v24, (__int64 *)a1);
    result = *(_DWORD **)(a1 + 8);
    v14 = v25;
    v15 = v26;
    v16 = &result[*(unsigned int *)(a1 + 24)];
    if ( v25 != v16 )
    {
      while ( 1 )
      {
        for ( result = v14 + 1; v15 != result; ++result )
        {
          if ( *result <= 0xFFFFFFFD )
            break;
        }
        v17 = *(unsigned int *)(a2 + 24);
        v18 = *(_QWORD *)(a2 + 8);
        if ( (_DWORD)v17 )
        {
          v19 = (v17 - 1) & (37 * *v14);
          v20 = (int *)(v18 + 4LL * v19);
          v21 = *v20;
          if ( *v20 == *v14 )
          {
LABEL_25:
            if ( v20 != (int *)(v18 + 4 * v17) )
            {
              *v14 = -2;
              --*(_DWORD *)(a1 + 16);
              ++*(_DWORD *)(a1 + 20);
            }
          }
          else
          {
            v22 = 1;
            while ( v21 != -1 )
            {
              v23 = v22 + 1;
              v19 = (v17 - 1) & (v22 + v19);
              v20 = (int *)(v18 + 4LL * v19);
              v21 = *v20;
              if ( *v14 == *v20 )
                goto LABEL_25;
              v22 = v23;
            }
          }
        }
        if ( v16 == result )
          break;
        v14 = result;
      }
    }
  }
  else
  {
    result = *(_DWORD **)(a2 + 8);
    v5 = &result[*(unsigned int *)(a2 + 24)];
    if ( v3 && v5 != result )
    {
      while ( *result > 0xFFFFFFFD )
      {
        if ( v5 == ++result )
          return result;
      }
LABEL_8:
      if ( v5 != result )
      {
        v6 = *(_DWORD *)(a1 + 24);
        v7 = *(_QWORD *)(a1 + 8);
        if ( !v6 )
          goto LABEL_13;
        v8 = v6 - 1;
        v9 = (v6 - 1) & (37 * *result);
        v10 = (int *)(v7 + 4LL * (v8 & (unsigned int)(37 * *result)));
        v11 = *v10;
        if ( *result == *v10 )
        {
LABEL_11:
          *v10 = -2;
          --*(_DWORD *)(a1 + 16);
          ++*(_DWORD *)(a1 + 20);
          goto LABEL_13;
        }
        v12 = 1;
        while ( v11 != -1 )
        {
          v13 = v12 + 1;
          v9 = v8 & (v12 + v9);
          v10 = (int *)(v7 + 4LL * v9);
          v11 = *v10;
          if ( *result == *v10 )
            goto LABEL_11;
          v12 = v13;
        }
LABEL_13:
        while ( v5 != ++result )
        {
          if ( *result <= 0xFFFFFFFD )
            goto LABEL_8;
        }
      }
    }
  }
  return result;
}
