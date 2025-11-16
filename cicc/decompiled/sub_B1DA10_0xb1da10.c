// Function: sub_B1DA10
// Address: 0xb1da10
//
__int64 __fastcall sub_B1DA10(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v4; // rbx
  bool v5; // zf
  __int64 result; // rax
  __int64 v7; // rdx
  __int64 i; // rdx
  __int64 *v9; // rax
  __int64 *v10; // [rsp+8h] [rbp-28h] BYREF

  v4 = a2;
  v5 = (*(_QWORD *)(a1 + 8) & 1LL) == 0;
  *(_QWORD *)(a1 + 8) &= 1uLL;
  if ( v5 )
  {
    result = *(_QWORD *)(a1 + 16);
    v7 = 24LL * *(unsigned int *)(a1 + 24);
  }
  else
  {
    result = a1 + 16;
    v7 = 96;
  }
  for ( i = result + v7; i != result; result += 24 )
  {
    if ( result )
    {
      *(_QWORD *)result = -4096;
      *(_QWORD *)(result + 8) = -4096;
    }
  }
  if ( a2 != a3 )
  {
    while ( 1 )
    {
      result = *(_QWORD *)v4;
      if ( *(_QWORD *)v4 != -4096 )
        break;
      if ( *(_QWORD *)(v4 + 8) == -4096 )
      {
        v4 += 24;
        if ( a3 == v4 )
          return result;
      }
      else
      {
LABEL_10:
        sub_B1C410(a1, (__int64 *)v4, &v10);
        v9 = v10;
        *v10 = *(_QWORD *)v4;
        v9[1] = *(_QWORD *)(v4 + 8);
        *((_DWORD *)v10 + 4) = *(_DWORD *)(v4 + 16);
        result = (2 * (*(_DWORD *)(a1 + 8) >> 1) + 2) | *(_DWORD *)(a1 + 8) & 1u;
        *(_DWORD *)(a1 + 8) = result;
LABEL_11:
        v4 += 24;
        if ( a3 == v4 )
          return result;
      }
    }
    if ( result == -8192 && *(_QWORD *)(v4 + 8) == -8192 )
      goto LABEL_11;
    goto LABEL_10;
  }
  return result;
}
