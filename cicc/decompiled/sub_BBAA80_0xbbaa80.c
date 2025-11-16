// Function: sub_BBAA80
// Address: 0xbbaa80
//
__int64 __fastcall sub_BBAA80(__int64 a1)
{
  __int64 v2; // rax
  _QWORD *v3; // r12
  _QWORD *v4; // r15
  _QWORD *v5; // rbx
  __int64 result; // rax
  __int64 v7; // r12
  __int64 v8; // rbx
  __int64 *v9; // rdi

  v2 = *(_QWORD *)(a1 + 192);
  if ( v2 != *(_QWORD *)(a1 + 200) )
    *(_QWORD *)(a1 + 200) = v2;
  v3 = *(_QWORD **)(a1 + 136);
  v4 = *(_QWORD **)(a1 + 144);
  if ( v3 != v4 )
  {
    v5 = *(_QWORD **)(a1 + 136);
    do
    {
      if ( (_QWORD *)*v5 != v5 + 2 )
        j_j___libc_free_0(*v5, v5[2] + 1LL);
      v5 += 4;
    }
    while ( v4 != v5 );
    *(_QWORD *)(a1 + 144) = v3;
  }
  result = *(_QWORD *)(a1 + 160);
  v7 = *(_QWORD *)(a1 + 168);
  v8 = result + 8;
  if ( result != v7 )
  {
    while ( 1 )
    {
      v9 = *(__int64 **)(a1 + 144);
      if ( v9 == *(__int64 **)(a1 + 152) )
      {
        sub_8FD760((__m128i **)(a1 + 136), *(const __m128i **)(a1 + 144), v8);
        result = v8 + 48;
        if ( v7 == v8 + 40 )
          return result;
      }
      else
      {
        if ( v9 )
        {
          *v9 = (__int64)(v9 + 2);
          sub_BB8750(v9, *(_BYTE **)v8, *(_QWORD *)v8 + *(_QWORD *)(v8 + 8));
          v9 = *(__int64 **)(a1 + 144);
        }
        result = v8 + 48;
        *(_QWORD *)(a1 + 144) = v9 + 4;
        if ( v7 == v8 + 40 )
          return result;
      }
      v8 = result;
    }
  }
  return result;
}
