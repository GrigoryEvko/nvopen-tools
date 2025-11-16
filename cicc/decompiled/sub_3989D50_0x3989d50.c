// Function: sub_3989D50
// Address: 0x3989d50
//
__int64 __fastcall sub_3989D50(__int64 a1)
{
  __int64 *v1; // rbx
  __int64 result; // rax
  __int64 *i; // r13
  __int64 v5; // rax
  int v6; // edx
  __int64 v7; // rdi
  int v8; // esi
  __int64 v9; // rdi
  unsigned int v10; // ecx
  __int64 *v11; // rdx
  __int64 v12; // r8
  __int64 v13; // rsi
  int v14; // edx
  int v15; // r9d

  v1 = *(__int64 **)(a1 + 664);
  result = *(unsigned int *)(a1 + 672);
  for ( i = &v1[result]; i != v1; result = sub_39C8490(v7, v13) )
  {
    v5 = sub_3981E80(*(_QWORD *)(*v1 + 16));
    v6 = *(_DWORD *)(a1 + 600);
    v7 = 0;
    if ( v6 )
    {
      v8 = v6 - 1;
      v9 = *(_QWORD *)(a1 + 584);
      v10 = (v6 - 1) & (((unsigned int)v5 >> 9) ^ ((unsigned int)v5 >> 4));
      v11 = (__int64 *)(v9 + 16LL * v10);
      v12 = *v11;
      if ( v5 == *v11 )
      {
LABEL_4:
        v7 = v11[1];
      }
      else
      {
        v14 = 1;
        while ( v12 != -8 )
        {
          v15 = v14 + 1;
          v10 = v8 & (v14 + v10);
          v11 = (__int64 *)(v9 + 16LL * v10);
          v12 = *v11;
          if ( v5 == *v11 )
            goto LABEL_4;
          v14 = v15;
        }
        v7 = 0;
      }
    }
    v13 = *v1++;
  }
  return result;
}
