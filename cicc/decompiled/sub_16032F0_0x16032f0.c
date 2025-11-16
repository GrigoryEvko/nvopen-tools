// Function: sub_16032F0
// Address: 0x16032f0
//
__int64 __fastcall sub_16032F0(__int64 *a1, __int64 a2)
{
  __int64 v2; // r12
  __int64 result; // rax
  int v4; // edx
  int v5; // r8d
  __int64 v6; // rdi
  __int64 *v7; // rbx
  __int64 v8; // rcx
  __int64 *v9; // rdi

  v2 = *a1;
  result = *(unsigned int *)(*a1 + 2952);
  if ( (_DWORD)result )
  {
    v4 = result - 1;
    v5 = 1;
    v6 = *(_QWORD *)(v2 + 2936);
    result = ((_DWORD)result - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v7 = (__int64 *)(v6 + 40 * result);
    v8 = *v7;
    if ( a2 == *v7 )
    {
LABEL_3:
      v9 = (__int64 *)v7[1];
      result = (__int64)(v7 + 3);
      if ( v9 != v7 + 3 )
        result = j_j___libc_free_0(v9, v7[3] + 1);
      *v7 = -16;
      --*(_DWORD *)(v2 + 2944);
      ++*(_DWORD *)(v2 + 2948);
    }
    else
    {
      while ( v8 != -8 )
      {
        result = v4 & (unsigned int)(v5 + result);
        v7 = (__int64 *)(v6 + 40LL * (unsigned int)result);
        v8 = *v7;
        if ( a2 == *v7 )
          goto LABEL_3;
        ++v5;
      }
    }
  }
  return result;
}
