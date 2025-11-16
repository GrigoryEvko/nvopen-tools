// Function: sub_138F420
// Address: 0x138f420
//
__int64 __fastcall sub_138F420(__int64 a1, __int64 a2)
{
  __int64 result; // rax
  int v3; // edx
  int v4; // r8d
  __int64 v6; // rdi
  __int64 *v7; // r12
  __int64 v8; // rcx
  unsigned __int64 v9; // rdi
  unsigned __int64 v10; // rdi
  __int64 v11; // rdi

  result = *(unsigned int *)(a1 + 40);
  if ( (_DWORD)result )
  {
    v3 = result - 1;
    v4 = 1;
    result = ((_DWORD)result - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v6 = *(_QWORD *)(a1 + 24);
    v7 = (__int64 *)(v6 + 424 * result);
    v8 = *v7;
    if ( *v7 == a2 )
    {
LABEL_3:
      if ( *((_BYTE *)v7 + 416) )
      {
        v9 = v7[34];
        if ( (__int64 *)v9 != v7 + 36 )
          _libc_free(v9);
        v10 = v7[8];
        if ( (__int64 *)v10 != v7 + 10 )
          _libc_free(v10);
        v11 = v7[5];
        if ( v11 )
          j_j___libc_free_0(v11, v7[7] - v11);
        result = j___libc_free_0(v7[2]);
      }
      *v7 = -16;
      --*(_DWORD *)(a1 + 32);
      ++*(_DWORD *)(a1 + 36);
    }
    else
    {
      while ( v8 != -8 )
      {
        result = v3 & (unsigned int)(v4 + result);
        v7 = (__int64 *)(v6 + 424LL * (unsigned int)result);
        v8 = *v7;
        if ( *v7 == a2 )
          goto LABEL_3;
        ++v4;
      }
    }
  }
  return result;
}
