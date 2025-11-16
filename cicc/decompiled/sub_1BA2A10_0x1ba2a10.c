// Function: sub_1BA2A10
// Address: 0x1ba2a10
//
__int64 __fastcall sub_1BA2A10(__int64 a1, int a2)
{
  __int64 v2; // rbx
  __int64 v3; // r13
  unsigned int v4; // eax
  __int64 result; // rax
  __int64 v6; // r14
  __int64 i; // rdx
  __int64 v8; // rbx
  _QWORD *v9; // rax
  __int64 j; // rdx
  _QWORD *v11; // [rsp+8h] [rbp-38h] BYREF

  v2 = *(unsigned int *)(a1 + 24);
  v3 = *(_QWORD *)(a1 + 8);
  v4 = sub_1454B60((unsigned int)(a2 - 1));
  if ( v4 < 0x40 )
    v4 = 64;
  *(_DWORD *)(a1 + 24) = v4;
  result = sub_22077B0(24LL * v4);
  *(_QWORD *)(a1 + 8) = result;
  if ( v3 )
  {
    *(_QWORD *)(a1 + 16) = 0;
    v6 = v3 + 24 * v2;
    for ( i = result + 24LL * *(unsigned int *)(a1 + 24); i != result; result += 24 )
    {
      if ( result )
      {
        *(_QWORD *)result = -8;
        *(_DWORD *)(result + 8) = -1;
      }
    }
    if ( v6 != v3 )
    {
      v8 = v3;
      while ( *(_QWORD *)v8 == -8 )
      {
        if ( *(_DWORD *)(v8 + 8) == -1 )
        {
          v8 += 24;
          if ( v6 == v8 )
            return j___libc_free_0(v3);
        }
        else
        {
LABEL_11:
          sub_1B99450(a1, (__int64 *)v8, &v11);
          v9 = v11;
          *v11 = *(_QWORD *)v8;
          *((_DWORD *)v9 + 2) = *(_DWORD *)(v8 + 8);
          v9[2] = *(_QWORD *)(v8 + 16);
          ++*(_DWORD *)(a1 + 16);
LABEL_12:
          v8 += 24;
          if ( v6 == v8 )
            return j___libc_free_0(v3);
        }
      }
      if ( *(_QWORD *)v8 == -16 && *(_DWORD *)(v8 + 8) == -2 )
        goto LABEL_12;
      goto LABEL_11;
    }
    return j___libc_free_0(v3);
  }
  else
  {
    *(_QWORD *)(a1 + 16) = 0;
    for ( j = result + 24LL * *(unsigned int *)(a1 + 24); j != result; result += 24 )
    {
      if ( result )
      {
        *(_QWORD *)result = -8;
        *(_DWORD *)(result + 8) = -1;
      }
    }
  }
  return result;
}
