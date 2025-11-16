// Function: sub_101AF30
// Address: 0x101af30
//
__int64 __fastcall sub_101AF30(__int64 a1, unsigned __int8 *a2, __m128i *a3, unsigned int a4)
{
  __int64 v8; // rsi
  __int64 v10; // rax
  __int64 v11; // rdi

  if ( *a2 == 69 )
  {
    v10 = *((_QWORD *)a2 - 4);
    if ( v10 )
    {
      v11 = *(_QWORD *)(v10 + 8);
      if ( (unsigned int)*(unsigned __int8 *)(v11 + 8) - 17 <= 1 )
        v11 = **(_QWORD **)(v11 + 16);
      v8 = 1;
      if ( sub_BCAC40(v11, 1) )
        return sub_AD6530(*(_QWORD *)(a1 + 8), v8);
    }
  }
  v8 = (__int64)a2;
  if ( sub_98F660((unsigned __int8 *)a1, a2, 0, 1) )
    return sub_AD6530(*(_QWORD *)(a1 + 8), v8);
  else
    return sub_101AA20(0x17u, (unsigned __int8 *)a1, a2, a3, a4);
}
