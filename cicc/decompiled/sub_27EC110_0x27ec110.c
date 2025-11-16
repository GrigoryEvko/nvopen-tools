// Function: sub_27EC110
// Address: 0x27ec110
//
__int64 __fastcall sub_27EC110(__int64 a1, __int64 a2, __int64 (__fastcall *a3)(__int64, _QWORD), __int64 a4)
{
  __int64 result; // rax
  __int64 *v5; // rbx
  __int64 v8; // rdi
  __int64 v9; // rsi
  unsigned int v10; // ecx
  __int64 *v11; // rdx
  __int64 v12; // r9
  __int64 v13; // r14
  __int64 j; // r15
  int v15; // edx
  int v16; // r10d
  __int64 *i; // [rsp+8h] [rbp-38h]

  result = *(_QWORD *)(a2 + 40);
  v5 = *(__int64 **)(a2 + 32);
  for ( i = (__int64 *)result; i != v5; ++v5 )
  {
    v8 = *v5;
    v9 = *(_QWORD *)(a1 + 72);
    result = *(unsigned int *)(a1 + 88);
    if ( (_DWORD)result )
    {
      v10 = (result - 1) & (((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4));
      v11 = (__int64 *)(v9 + 16LL * v10);
      v12 = *v11;
      if ( v8 == *v11 )
      {
LABEL_4:
        result = v9 + 16 * result;
        if ( v11 != (__int64 *)result )
        {
          v13 = v11[1];
          if ( v13 )
          {
            for ( j = *(_QWORD *)(v13 + 8); v13 != j; j = *(_QWORD *)(j + 8) )
            {
              if ( !j )
                BUG();
              if ( (unsigned int)*(unsigned __int8 *)(j - 32) - 26 <= 1 )
                result = a3(a4, *(_QWORD *)(j + 40));
            }
          }
        }
      }
      else
      {
        v15 = 1;
        while ( v12 != -4096 )
        {
          v16 = v15 + 1;
          v10 = (result - 1) & (v15 + v10);
          v11 = (__int64 *)(v9 + 16LL * v10);
          v12 = *v11;
          if ( v8 == *v11 )
            goto LABEL_4;
          v15 = v16;
        }
      }
    }
  }
  return result;
}
