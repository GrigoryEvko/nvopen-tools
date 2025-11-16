// Function: sub_D495C0
// Address: 0xd495c0
//
__int64 __fastcall sub_D495C0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 result; // rax
  int v5; // edx
  __int64 v6; // rcx
  __int64 v7; // rsi
  int v8; // edi
  __int64 v9; // rdx
  __int64 *v10; // rax
  __int64 v11; // r9
  __int64 v12; // rdi
  int v13; // eax
  int v14; // r10d

  result = 0;
  if ( *(_BYTE *)(*(_QWORD *)(a2 + 8) + 8LL) != 11 && *(_BYTE *)a2 > 0x1Cu )
  {
    v5 = *(_DWORD *)(a1 + 24);
    v6 = *(_QWORD *)(a2 + 40);
    v7 = *(_QWORD *)(a1 + 8);
    if ( v5 )
    {
      v8 = v5 - 1;
      v9 = (v5 - 1) & (((unsigned int)v6 >> 9) ^ ((unsigned int)v6 >> 4));
      v10 = (__int64 *)(v7 + 16 * v9);
      v11 = *v10;
      if ( *v10 == v6 )
      {
LABEL_5:
        v12 = v10[1];
        if ( v12 )
          return (unsigned int)sub_B19060(v12 + 56, a3, v9, v6) ^ 1;
      }
      else
      {
        v13 = 1;
        while ( v11 != -4096 )
        {
          v14 = v13 + 1;
          v9 = v8 & (unsigned int)(v13 + v9);
          v10 = (__int64 *)(v7 + 16LL * (unsigned int)v9);
          v11 = *v10;
          if ( v6 == *v10 )
            goto LABEL_5;
          v13 = v14;
        }
      }
      return 0;
    }
  }
  return result;
}
