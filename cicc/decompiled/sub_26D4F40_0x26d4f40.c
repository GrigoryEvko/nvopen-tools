// Function: sub_26D4F40
// Address: 0x26d4f40
//
__int64 __fastcall sub_26D4F40(__int64 a1)
{
  __int64 v1; // rbx
  __int64 v2; // r13
  __int64 result; // rax
  unsigned int v4; // ecx
  __int64 *v5; // rdx
  __int64 v6; // r9
  unsigned int v7; // eax
  __int64 v8; // rax
  __int64 v9; // r8
  __int64 v10; // rsi
  int v11; // edx
  int v12; // r10d

  v1 = *(_QWORD *)(a1 + 96);
  v2 = *(_QWORD *)(v1 - 16);
  for ( result = *(_QWORD *)(v1 - 24) + 24LL; v2 != result; result = *(_QWORD *)(v1 - 24) + 24LL )
  {
    *(_QWORD *)(v1 - 16) = sub_220EF30(v2);
    v8 = *(unsigned int *)(a1 + 32);
    v9 = *(_QWORD *)(v2 + 40);
    v10 = *(_QWORD *)(a1 + 16);
    if ( (_DWORD)v8 )
    {
      v4 = (v8 - 1) & (((unsigned int)v9 >> 9) ^ ((unsigned int)v9 >> 4));
      v5 = (__int64 *)(v10 + 16LL * v4);
      v6 = *v5;
      if ( v9 == *v5 )
      {
LABEL_4:
        if ( v5 != (__int64 *)(v10 + 16 * v8) )
        {
          v1 = *(_QWORD *)(a1 + 96);
          v7 = *((_DWORD *)v5 + 2);
          if ( *(_DWORD *)(v1 - 8) > v7 )
          {
            *(_DWORD *)(v1 - 8) = v7;
            v1 = *(_QWORD *)(a1 + 96);
          }
          goto LABEL_7;
        }
      }
      else
      {
        v11 = 1;
        while ( v6 != -4096 )
        {
          v12 = v11 + 1;
          v4 = (v8 - 1) & (v11 + v4);
          v5 = (__int64 *)(v10 + 16LL * v4);
          v6 = *v5;
          if ( v9 == *v5 )
            goto LABEL_4;
          v11 = v12;
        }
      }
    }
    sub_26D4D40((int *)a1, *(_QWORD *)(v2 + 40));
    v1 = *(_QWORD *)(a1 + 96);
LABEL_7:
    v2 = *(_QWORD *)(v1 - 16);
  }
  return result;
}
