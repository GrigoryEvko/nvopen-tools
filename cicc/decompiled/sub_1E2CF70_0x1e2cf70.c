// Function: sub_1E2CF70
// Address: 0x1e2cf70
//
__int64 __fastcall sub_1E2CF70(__int64 a1, __int64 a2)
{
  __int64 result; // rax
  __int64 v4; // rcx
  int v5; // edx
  int v6; // r8d
  __int64 *v7; // r12
  __int64 v8; // rdi
  __int64 v9; // r13

  result = *(unsigned int *)(a1 + 1776);
  if ( (_DWORD)result )
  {
    v4 = *(_QWORD *)(a1 + 1760);
    v5 = result - 1;
    v6 = 1;
    result = ((_DWORD)result - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v7 = (__int64 *)(v4 + 16 * result);
    v8 = *v7;
    if ( a2 == *v7 )
    {
LABEL_3:
      v9 = v7[1];
      if ( v9 )
      {
        sub_1E11810(v7[1], a2);
        result = j_j___libc_free_0(v9, 752);
      }
      *v7 = -16;
      --*(_DWORD *)(a1 + 1768);
      ++*(_DWORD *)(a1 + 1772);
    }
    else
    {
      while ( v8 != -8 )
      {
        result = v5 & (unsigned int)(v6 + result);
        v7 = (__int64 *)(v4 + 16LL * (unsigned int)result);
        v8 = *v7;
        if ( a2 == *v7 )
          goto LABEL_3;
        ++v6;
      }
    }
  }
  *(_QWORD *)(a1 + 1792) = 0;
  *(_QWORD *)(a1 + 1800) = 0;
  return result;
}
