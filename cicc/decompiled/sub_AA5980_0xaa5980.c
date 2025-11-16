// Function: sub_AA5980
// Address: 0xaa5980
//
__int64 __fastcall sub_AA5980(__int64 a1, __int64 a2, unsigned __int8 a3)
{
  __int64 result; // rax
  __int64 v6; // rdx
  __int64 v7; // r15
  __int64 v8; // rax
  __int64 v9; // rbx
  __int64 v10; // rax
  unsigned int v11; // r8d
  __int64 v12; // rsi
  __int64 v13; // rdi
  __int64 v14; // rdx
  __int64 v15; // rcx
  int v16; // [rsp+4h] [rbp-3Ch]
  __int64 v17; // [rsp+8h] [rbp-38h]

  result = *(_QWORD *)(a1 + 48) & 0xFFFFFFFFFFFFFFF8LL;
  if ( result != a1 + 48 )
  {
    result = *(_QWORD *)(a1 + 56);
    if ( !result )
      BUG();
    if ( *(_BYTE *)(result - 24) == 84 )
    {
      v16 = *(_DWORD *)(result - 20) & 0x7FFFFFF;
      result = sub_AA5930(a1);
      v17 = v6;
      v7 = result;
      while ( v7 != v17 )
      {
        if ( !v7 )
          BUG();
        v8 = *(_QWORD *)(v7 + 32);
        if ( !v8 )
          BUG();
        v9 = 0;
        if ( *(_BYTE *)(v8 - 24) == 84 )
          v9 = v8 - 24;
        if ( (*(_DWORD *)(v7 + 4) & 0x7FFFFFF) != 0 )
        {
          v10 = 0;
          while ( 1 )
          {
            v11 = v10;
            if ( a2 == *(_QWORD *)(*(_QWORD *)(v7 - 8) + 32LL * *(unsigned int *)(v7 + 72) + 8 * v10) )
              break;
            if ( (*(_DWORD *)(v7 + 4) & 0x7FFFFFF) == (_DWORD)++v10 )
              goto LABEL_20;
          }
        }
        else
        {
LABEL_20:
          v11 = -1;
        }
        result = sub_B48BF0(v7, v11, a3 ^ 1u);
        if ( v16 == 1 || a3 || (result = sub_B48DC0(v7), (v12 = result) == 0) )
        {
          v7 = v9;
        }
        else
        {
          sub_BD84D0(v7, result);
          v13 = v7;
          v7 = v9;
          result = sub_B43D60(v13, v12, v14, v15);
        }
      }
    }
  }
  return result;
}
