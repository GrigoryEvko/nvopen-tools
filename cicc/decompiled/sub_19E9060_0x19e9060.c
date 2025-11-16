// Function: sub_19E9060
// Address: 0x19e9060
//
__int64 *__fastcall sub_19E9060(__int64 a1)
{
  __int64 *result; // rax
  __int64 v3; // rdx
  __int64 v4; // rcx
  __int64 v5; // rdi
  __int64 v6; // rsi
  int v7; // r8d
  int v8; // r10d
  __int64 v9; // r9
  unsigned int v10; // r8d
  __int64 *v11; // rsi
  __int64 v12; // r11
  __int64 v13; // rdx
  __int64 v14; // rdx
  __int64 v15; // rax
  int v16; // esi
  int v17; // r13d
  __int64 *v18; // [rsp-40h] [rbp-40h] BYREF
  __int64 v19[7]; // [rsp-38h] [rbp-38h] BYREF

  result = *(__int64 **)a1;
  if ( *(_QWORD *)a1 != *(_QWORD *)(a1 + 8) )
  {
    do
    {
      v3 = *result;
      v4 = **(_QWORD **)(a1 + 16);
      if ( *result != v4 )
      {
        v5 = *(_QWORD *)(a1 + 24);
        v6 = 0;
        v7 = *(_DWORD *)(v5 + 1984);
        if ( v7 )
        {
          v8 = v7 - 1;
          v9 = *(_QWORD *)(v5 + 1968);
          v10 = (v7 - 1) & (((unsigned int)v3 >> 9) ^ ((unsigned int)v3 >> 4));
          v11 = (__int64 *)(v9 + 16LL * v10);
          v12 = *v11;
          if ( v3 == *v11 )
          {
LABEL_5:
            v6 = v11[1];
          }
          else
          {
            v16 = 1;
            while ( v12 != -8 )
            {
              v17 = v16 + 1;
              v10 = v8 & (v16 + v10);
              v11 = (__int64 *)(v9 + 16LL * v10);
              v12 = *v11;
              if ( v3 == *v11 )
                goto LABEL_5;
              v16 = v17;
            }
            v6 = 0;
          }
        }
        if ( *(_QWORD *)(v5 + 1432) != v6 )
        {
          v13 = (*(_BYTE *)(v4 + 23) & 0x40) != 0 ? *(_QWORD *)(v4 - 8) : v4 - 24LL * (*(_DWORD *)(v4 + 20) & 0xFFFFFFF);
          v14 = *(_QWORD *)(v13
                          + 0xFFFFFFFD55555558LL * (unsigned int)(((__int64)result - v13) >> 3)
                          + 24LL * *(unsigned int *)(v4 + 76)
                          + 8);
          v15 = **(_QWORD **)(a1 + 32);
          v19[0] = v14;
          v19[1] = v15;
          result = (__int64 *)sub_19E8F30(v5 + 2200, v19, &v18);
          if ( (_BYTE)result )
            break;
        }
      }
      result = (__int64 *)(*(_QWORD *)a1 + 24LL);
      *(_QWORD *)a1 = result;
    }
    while ( *(__int64 **)(a1 + 8) != result );
  }
  return result;
}
