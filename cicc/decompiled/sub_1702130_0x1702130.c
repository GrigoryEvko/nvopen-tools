// Function: sub_1702130
// Address: 0x1702130
//
__int64 __fastcall sub_1702130(__int64 a1, __int64 a2, __int64 *a3)
{
  __int64 **v5; // rdi
  __int64 v6; // r12
  __int64 result; // rax
  __int64 v8; // rax
  __int64 v9; // rsi
  unsigned int v10; // ecx
  __int64 *v11; // rdx
  __int64 v12; // r8
  int v13; // edx
  int v14; // r9d

  v5 = (__int64 **)a3;
  if ( *(_BYTE *)(*(_QWORD *)a2 + 8LL) == 16 )
    v5 = (__int64 **)sub_16463B0(a3, *(_QWORD *)(*(_QWORD *)a2 + 32LL));
  if ( *(_BYTE *)(a2 + 16) > 0x10u )
  {
    v8 = *(unsigned int *)(a1 + 104);
    if ( (_DWORD)v8 )
    {
      v9 = *(_QWORD *)(a1 + 88);
      v10 = (v8 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v11 = (__int64 *)(v9 + 16LL * v10);
      v12 = *v11;
      if ( a2 == *v11 )
      {
LABEL_9:
        if ( v11 != (__int64 *)(v9 + 16 * v8) )
          return *(_QWORD *)(*(_QWORD *)(a1 + 112) + 24LL * *((unsigned int *)v11 + 2) + 16);
      }
      else
      {
        v13 = 1;
        while ( v12 != -8 )
        {
          v14 = v13 + 1;
          v10 = (v8 - 1) & (v13 + v10);
          v11 = (__int64 *)(v9 + 16LL * v10);
          v12 = *v11;
          if ( a2 == *v11 )
            goto LABEL_9;
          v13 = v14;
        }
      }
    }
    return 0;
  }
  else
  {
    v6 = sub_15A4750((__int64 ***)a2, v5, 0);
    result = sub_14DBA30(v6, *(_QWORD *)(a1 + 8), *(_QWORD *)a1);
    if ( !result )
      return v6;
  }
  return result;
}
