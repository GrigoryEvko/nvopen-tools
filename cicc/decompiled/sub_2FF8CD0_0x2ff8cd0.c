// Function: sub_2FF8CD0
// Address: 0x2ff8cd0
//
__int64 __fastcall sub_2FF8CD0(__int64 a1, __int64 *a2)
{
  __int64 result; // rax
  __int64 v3; // r8
  __int64 v4; // rsi
  int v5; // ecx
  unsigned int v6; // edx
  __int64 v7; // r9
  unsigned int v8; // r10d

  result = *(unsigned int *)(a1 + 24);
  v3 = *(_QWORD *)(a1 + 8);
  if ( (_DWORD)result )
  {
    v4 = *a2;
    v5 = result - 1;
    v6 = (result - 1) & (((unsigned int)v4 >> 9) ^ ((unsigned int)v4 >> 4));
    result = v3 + 16LL * v6;
    v7 = *(_QWORD *)result;
    if ( v4 == *(_QWORD *)result )
    {
LABEL_3:
      *(_QWORD *)result = -8192;
      --*(_DWORD *)(a1 + 16);
      ++*(_DWORD *)(a1 + 20);
    }
    else
    {
      result = 1;
      while ( v7 != -4096 )
      {
        v8 = result + 1;
        v6 = v5 & (result + v6);
        result = v3 + 16LL * v6;
        v7 = *(_QWORD *)result;
        if ( v4 == *(_QWORD *)result )
          goto LABEL_3;
        result = v8;
      }
    }
  }
  return result;
}
