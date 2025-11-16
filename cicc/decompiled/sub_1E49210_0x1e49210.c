// Function: sub_1E49210
// Address: 0x1e49210
//
__int64 __fastcall sub_1E49210(__int64 **a1, __int64 *a2)
{
  __int64 v2; // rbp
  __int64 v3; // rcx
  __int64 result; // rax
  int v5; // edx
  __int64 v6; // r9
  int v7; // edx
  int v8; // r10d
  unsigned int v9; // eax
  __int64 v10; // rcx
  __int64 v11; // rbx
  _QWORD v12[4]; // [rsp-20h] [rbp-20h] BYREF

  v3 = **a1;
  result = 0;
  v5 = *(_DWORD *)(v3 + 24);
  if ( v5 )
  {
    v6 = *(_QWORD *)(v3 + 8);
    v7 = v5 - 1;
    v8 = 1;
    v9 = v7 & (((unsigned int)*a2 >> 9) ^ ((unsigned int)*a2 >> 4));
    v10 = *(_QWORD *)(v6 + 8LL * v9);
    if ( *a2 == v10 )
    {
LABEL_3:
      v12[3] = v2;
      v11 = (__int64)a1[1];
      result = sub_1E49160(v11, a2, v12);
      if ( (_BYTE)result )
      {
        *(_QWORD *)v12[0] = -16;
        --*(_DWORD *)(v11 + 16);
        ++*(_DWORD *)(v11 + 20);
      }
      else
      {
        return 1;
      }
    }
    else
    {
      while ( v10 != -8 )
      {
        v9 = v7 & (v8 + v9);
        v10 = *(_QWORD *)(v6 + 8LL * v9);
        if ( *a2 == v10 )
          goto LABEL_3;
        ++v8;
      }
      return 0;
    }
  }
  return result;
}
