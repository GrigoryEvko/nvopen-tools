// Function: sub_30EC360
// Address: 0x30ec360
//
__int64 __fastcall sub_30EC360(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 result; // rax
  __int64 v6; // rsi
  int v7; // ecx
  unsigned int v8; // edx
  __int64 v9; // rdi
  unsigned int v10; // r8d

  result = (**(__int64 (__fastcall ***)(__int64))a1)(a1);
  if ( (_BYTE)result )
  {
    result = *(unsigned int *)(a1 + 32);
    v6 = *(_QWORD *)(a1 + 16);
    if ( (_DWORD)result )
    {
      v7 = result - 1;
      v8 = (result - 1) & (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4));
      result = v6 + 16LL * v8;
      v9 = *(_QWORD *)result;
      if ( *(_QWORD *)result == a3 )
      {
LABEL_5:
        *(_QWORD *)result = -8192;
        --*(_DWORD *)(a1 + 24);
        ++*(_DWORD *)(a1 + 28);
      }
      else
      {
        result = 1;
        while ( v9 != -4096 )
        {
          v10 = result + 1;
          v8 = v7 & (result + v8);
          result = v6 + 16LL * v8;
          v9 = *(_QWORD *)result;
          if ( *(_QWORD *)result == a3 )
            goto LABEL_5;
          result = v10;
        }
      }
    }
  }
  return result;
}
