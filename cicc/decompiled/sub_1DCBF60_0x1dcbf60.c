// Function: sub_1DCBF60
// Address: 0x1dcbf60
//
__int64 __fastcall sub_1DCBF60(__int64 a1, unsigned __int64 a2, __int64 *a3)
{
  _QWORD *v5; // rdi
  __int64 v6; // r8
  __int64 result; // rax
  unsigned __int64 v8; // rdx
  __int64 v9; // rcx
  __int64 v10; // rcx
  unsigned __int64 v11; // rsi
  __int64 j; // rdx
  _QWORD *v13; // rdx
  __int64 v14; // rcx
  __int64 v15; // r13
  __int64 v16; // rcx
  __int64 v17; // rdx
  __int64 i; // rsi

  v5 = *(_QWORD **)a1;
  v6 = *(_QWORD *)(a1 + 16) - (_QWORD)v5;
  if ( v6 >> 3 < a2 )
  {
    result = 0xFFFFFFFFFFFFFFFLL;
    if ( a2 > 0xFFFFFFFFFFFFFFFLL )
      sub_4262D8((__int64)"cannot create std::vector larger than max_size()");
    v15 = 8 * a2;
    if ( a2 )
    {
      result = sub_22077B0(8 * a2);
      v16 = *a3;
      v17 = result + v15;
      for ( i = result; result != v17; result += 8 )
        *(_QWORD *)result = v16;
      v5 = *(_QWORD **)a1;
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
    v8 = (result - (__int64)v5) >> 3;
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
      if ( (_QWORD *)result != v13 )
        *(_QWORD *)(a1 + 8) = v13;
    }
    else
    {
      v9 = *a3;
      if ( v5 != (_QWORD *)result )
      {
        do
          *v5++ = v9;
        while ( (_QWORD *)result != v5 );
        result = *(_QWORD *)(a1 + 8);
        v8 = (result - *(_QWORD *)a1) >> 3;
      }
      v10 = result;
      v11 = a2 - v8;
      if ( v11 )
      {
        v10 = result + 8 * v11;
        for ( j = *a3; v10 != result; result += 8 )
          *(_QWORD *)result = j;
      }
      *(_QWORD *)(a1 + 8) = v10;
    }
  }
  return result;
}
