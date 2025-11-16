// Function: sub_1A26430
// Address: 0x1a26430
//
__int64 __fastcall sub_1A26430(__int64 *a1, __int64 *a2)
{
  _BOOL4 v3; // r8d
  __int64 result; // rax
  __int64 v5; // rdx
  int v6; // ecx
  __int64 v7; // rsi
  int v8; // ecx
  __int64 v9; // r9
  unsigned int v10; // eax
  __int64 *v11; // rdi
  __int64 v12; // r8
  int v13; // edi
  int v14; // r10d

  v3 = sub_1A26350(*a1, *a2);
  result = 0;
  if ( v3 )
  {
    v5 = a1[1];
    result = 1;
    v6 = *(_DWORD *)(v5 + 24);
    if ( v6 )
    {
      v7 = *a2;
      v8 = v6 - 1;
      v9 = *(_QWORD *)(v5 + 8);
      v10 = v8 & (((unsigned int)*a2 >> 9) ^ ((unsigned int)*a2 >> 4));
      v11 = (__int64 *)(v9 + 8LL * v10);
      v12 = *v11;
      if ( *a2 == *v11 )
      {
LABEL_5:
        *v11 = -16;
        --*(_DWORD *)(v5 + 16);
        ++*(_DWORD *)(v5 + 20);
        return 1;
      }
      else
      {
        v13 = 1;
        while ( v12 != -8 )
        {
          v14 = v13 + 1;
          v10 = v8 & (v13 + v10);
          v11 = (__int64 *)(v9 + 8LL * v10);
          v12 = *v11;
          if ( v7 == *v11 )
            goto LABEL_5;
          v13 = v14;
        }
        return 1;
      }
    }
  }
  return result;
}
