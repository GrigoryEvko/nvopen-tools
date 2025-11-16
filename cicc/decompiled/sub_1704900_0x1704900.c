// Function: sub_1704900
// Address: 0x1704900
//
__int64 __fastcall sub_1704900(_QWORD *a1)
{
  __int64 v2; // rdx
  __int64 result; // rax
  int v4; // ecx
  int v5; // esi
  __int64 v6; // r9
  unsigned int v7; // ecx
  __int64 *v8; // rdi
  __int64 v9; // r8
  int v10; // edi
  int v11; // r10d

  v2 = a1[4];
  result = a1[3];
  v4 = *(_DWORD *)(v2 + 24);
  if ( v4 )
  {
    v5 = v4 - 1;
    v6 = *(_QWORD *)(v2 + 8);
    v7 = (v4 - 1) & (((unsigned int)result >> 9) ^ ((unsigned int)result >> 4));
    v8 = (__int64 *)(v6 + 8LL * v7);
    v9 = *v8;
    if ( result == *v8 )
    {
LABEL_3:
      *v8 = -16;
      --*(_DWORD *)(v2 + 16);
      ++*(_DWORD *)(v2 + 20);
      result = a1[3];
    }
    else
    {
      v10 = 1;
      while ( v9 != -8 )
      {
        v11 = v10 + 1;
        v7 = v5 & (v10 + v7);
        v8 = (__int64 *)(v6 + 8LL * v7);
        v9 = *v8;
        if ( result == *v8 )
          goto LABEL_3;
        v10 = v11;
      }
    }
  }
  if ( result )
  {
    if ( result != -16 && result != -8 )
      result = sub_1649B30(a1 + 1);
    a1[3] = 0;
  }
  return result;
}
