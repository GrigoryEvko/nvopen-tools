// Function: sub_1A9B950
// Address: 0x1a9b950
//
__int64 __fastcall sub_1A9B950(__int64 *a1, __int64 a2)
{
  __int64 v2; // rax
  __int64 v3; // rbx
  __int64 v4; // r12
  bool v5; // r8
  __int64 result; // rax
  _QWORD *v7; // rsi
  __int64 v8; // rax
  __int64 v9; // rdi
  unsigned int v10; // ecx
  __int64 *v11; // rdx
  __int64 v12; // r9
  __int64 v13; // rdx
  int v14; // edx
  int v15; // r10d

  v2 = sub_1A9B680(a2, *a1);
  v3 = a1[1];
  v4 = v2;
  v5 = sub_1A94B30(v2);
  result = 1;
  if ( !v5 )
  {
    v7 = *(_QWORD **)v3;
    v8 = *(unsigned int *)(*(_QWORD *)v3 + 24LL);
    if ( (_DWORD)v8 )
    {
      v9 = v7[1];
      v10 = (v8 - 1) & (((unsigned int)v4 >> 9) ^ ((unsigned int)v4 >> 4));
      v11 = (__int64 *)(v9 + 16LL * v10);
      v12 = *v11;
      if ( v4 == *v11 )
      {
LABEL_4:
        if ( v11 != (__int64 *)(v9 + 16 * v8) )
        {
          v13 = v7[4] + 24LL * *((unsigned int *)v11 + 2);
          return *(unsigned int *)(v13 + 8);
        }
      }
      else
      {
        v14 = 1;
        while ( v12 != -8 )
        {
          v15 = v14 + 1;
          v10 = (v8 - 1) & (v14 + v10);
          v11 = (__int64 *)(v9 + 16LL * v10);
          v12 = *v11;
          if ( v4 == *v11 )
            goto LABEL_4;
          v14 = v15;
        }
      }
    }
    v13 = v7[5];
    return *(unsigned int *)(v13 + 8);
  }
  return result;
}
