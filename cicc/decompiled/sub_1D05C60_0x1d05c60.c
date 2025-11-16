// Function: sub_1D05C60
// Address: 0x1d05c60
//
__int64 __fastcall sub_1D05C60(__int64 a1, unsigned __int64 a2, int *a3)
{
  _DWORD *v5; // rdi
  __int64 v6; // r8
  __int64 result; // rax
  unsigned __int64 v8; // rdx
  int v9; // ecx
  __int64 v10; // rcx
  unsigned __int64 v11; // rsi
  int j; // edx
  _DWORD *v13; // rdx
  int v14; // ecx
  __int64 v15; // r13
  int v16; // ecx
  __int64 v17; // rdx
  __int64 i; // rsi

  v5 = *(_DWORD **)a1;
  v6 = *(_QWORD *)(a1 + 16) - (_QWORD)v5;
  if ( v6 >> 2 < a2 )
  {
    result = 0x1FFFFFFFFFFFFFFFLL;
    if ( a2 > 0x1FFFFFFFFFFFFFFFLL )
      sub_4262D8((__int64)"cannot create std::vector larger than max_size()");
    v15 = 4 * a2;
    if ( a2 )
    {
      result = sub_22077B0(4 * a2);
      v16 = *a3;
      v17 = result + v15;
      for ( i = result; result != v17; result += 4 )
        *(_DWORD *)result = v16;
      v5 = *(_DWORD **)a1;
      v6 = *(_QWORD *)(a1 + 16) - *(_QWORD *)a1;
    }
    else
    {
      i = 0;
      v17 = 0;
    }
    *(_QWORD *)a1 = i;
    *(_QWORD *)(a1 + 8) = v17;
    *(_QWORD *)(a1 + 16) = v17;
    if ( v5 )
      return j_j___libc_free_0(v5, v6);
  }
  else
  {
    result = *(_QWORD *)(a1 + 8);
    v8 = (result - (__int64)v5) >> 2;
    if ( a2 <= v8 )
    {
      v13 = v5;
      if ( a2 )
      {
        v13 = &v5[a2];
        v14 = *a3;
        if ( v5 != v13 )
        {
          do
            *v5++ = v14;
          while ( v13 != v5 );
          result = *(_QWORD *)(a1 + 8);
        }
      }
      if ( (_DWORD *)result != v13 )
        *(_QWORD *)(a1 + 8) = v13;
    }
    else
    {
      v9 = *a3;
      if ( v5 != (_DWORD *)result )
      {
        do
          *v5++ = v9;
        while ( (_DWORD *)result != v5 );
        result = *(_QWORD *)(a1 + 8);
        v8 = (result - *(_QWORD *)a1) >> 2;
      }
      v10 = result;
      v11 = a2 - v8;
      if ( v11 )
      {
        v10 = result + 4 * v11;
        for ( j = *a3; v10 != result; result += 4 )
          *(_DWORD *)result = j;
      }
      *(_QWORD *)(a1 + 8) = v10;
    }
  }
  return result;
}
