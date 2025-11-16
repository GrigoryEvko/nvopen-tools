// Function: sub_1CD0600
// Address: 0x1cd0600
//
__int64 __fastcall sub_1CD0600(__int64 a1, __int64 *a2)
{
  __int64 result; // rax
  __int64 v3; // rsi
  int v4; // ecx
  __int64 v5; // r8
  unsigned int v6; // edx
  __int64 v7; // r9
  unsigned int v8; // r10d

  result = *(unsigned int *)(a1 + 24);
  if ( (_DWORD)result )
  {
    v3 = *a2;
    v4 = result - 1;
    v5 = *(_QWORD *)(a1 + 8);
    v6 = (result - 1) & (((unsigned int)v3 >> 9) ^ ((unsigned int)v3 >> 4));
    result = v5 + 16LL * v6;
    v7 = *(_QWORD *)result;
    if ( v3 == *(_QWORD *)result )
    {
LABEL_3:
      *(_QWORD *)result = -16;
      --*(_DWORD *)(a1 + 16);
      ++*(_DWORD *)(a1 + 20);
    }
    else
    {
      result = 1;
      while ( v7 != -8 )
      {
        v8 = result + 1;
        v6 = v4 & (result + v6);
        result = v5 + 16LL * v6;
        v7 = *(_QWORD *)result;
        if ( v3 == *(_QWORD *)result )
          goto LABEL_3;
        result = v8;
      }
    }
  }
  return result;
}
