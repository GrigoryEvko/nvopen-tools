// Function: sub_13836B0
// Address: 0x13836b0
//
__int64 __fastcall sub_13836B0(__int64 a1, __int64 a2)
{
  __int64 result; // rax
  int v3; // edx
  int v4; // r8d
  __int64 v6; // rdi
  __int64 *v7; // r12
  __int64 v8; // rcx
  unsigned __int64 v9; // rdi
  unsigned __int64 v10; // rdi
  __int64 v11; // r14
  _QWORD *v12; // r13
  _QWORD *v13; // r14
  __int64 v14; // rdi

  result = *(unsigned int *)(a1 + 40);
  if ( (_DWORD)result )
  {
    v3 = result - 1;
    v4 = 1;
    result = ((_DWORD)result - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v6 = *(_QWORD *)(a1 + 24);
    v7 = (__int64 *)(v6 + 432 * result);
    v8 = *v7;
    if ( *v7 == a2 )
    {
LABEL_3:
      if ( *((_BYTE *)v7 + 424) )
      {
        v9 = v7[35];
        if ( (__int64 *)v9 != v7 + 37 )
          _libc_free(v9);
        v10 = v7[9];
        if ( (__int64 *)v10 != v7 + 11 )
          _libc_free(v10);
        j___libc_free_0(v7[6]);
        v11 = *((unsigned int *)v7 + 8);
        if ( (_DWORD)v11 )
        {
          v12 = (_QWORD *)v7[2];
          v13 = &v12[4 * v11];
          do
          {
            if ( *v12 != -16 && *v12 != -8 )
            {
              v14 = v12[1];
              if ( v14 )
                j_j___libc_free_0(v14, v12[3] - v14);
            }
            v12 += 4;
          }
          while ( v13 != v12 );
        }
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
        v7 = (__int64 *)(v6 + 432LL * (unsigned int)result);
        v8 = *v7;
        if ( *v7 == a2 )
          goto LABEL_3;
        ++v4;
      }
    }
  }
  return result;
}
