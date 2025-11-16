// Function: sub_B6F830
// Address: 0xb6f830
//
__int64 __fastcall sub_B6F830(__int64 *a1, __int64 a2)
{
  __int64 v2; // rbx
  __int64 result; // rax
  __int64 v4; // rdi
  int v5; // edx
  __int64 *v6; // r12
  __int64 v7; // rcx
  __int64 *v8; // rdi
  int v9; // r8d

  v2 = *a1;
  result = *(unsigned int *)(*a1 + 3488);
  v4 = *(_QWORD *)(*a1 + 3472);
  if ( (_DWORD)result )
  {
    v5 = result - 1;
    result = ((_DWORD)result - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v6 = (__int64 *)(v4 + 40 * result);
    v7 = *v6;
    if ( a2 == *v6 )
    {
LABEL_3:
      v8 = (__int64 *)v6[1];
      result = (__int64)(v6 + 3);
      if ( v8 != v6 + 3 )
        result = j_j___libc_free_0(v8, v6[3] + 1);
      *v6 = -8192;
      --*(_DWORD *)(v2 + 3480);
      ++*(_DWORD *)(v2 + 3484);
    }
    else
    {
      v9 = 1;
      while ( v7 != -4096 )
      {
        result = v5 & (unsigned int)(v9 + result);
        v6 = (__int64 *)(v4 + 40LL * (unsigned int)result);
        v7 = *v6;
        if ( a2 == *v6 )
          goto LABEL_3;
        ++v9;
      }
    }
  }
  return result;
}
