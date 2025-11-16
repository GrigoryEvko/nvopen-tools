// Function: sub_13899F0
// Address: 0x13899f0
//
__m128i *__fastcall sub_13899F0(__int64 a1, __int64 a2)
{
  __int64 v2; // r13
  __int64 v3; // r14
  __m128i *result; // rax
  unsigned __int8 v5; // al
  __int64 v6; // rbx
  __int64 v7; // rax
  __int64 v8; // rbx
  __int64 v9; // rax

  v2 = *(_QWORD *)(a2 - 48);
  v3 = *(_QWORD *)(a2 - 24);
  if ( *(_BYTE *)(*(_QWORD *)v2 + 8LL) == 15 )
  {
    result = *(__m128i **)a2;
    if ( *(_BYTE *)(*(_QWORD *)a2 + 8LL) != 15 )
      return result;
    v5 = *(_BYTE *)(v2 + 16);
    if ( v5 > 3u )
    {
      if ( v5 == 5 )
      {
        if ( (unsigned int)*(unsigned __int16 *)(v2 + 18) - 51 > 1
          && (unsigned __int8)sub_13848E0(*(_QWORD *)(a1 + 24), *(_QWORD *)(a2 - 48), 0, 0) )
        {
          sub_1389140(a1, v2);
        }
      }
      else
      {
        sub_13848E0(*(_QWORD *)(a1 + 24), *(_QWORD *)(a2 - 48), 0, 0);
      }
    }
    else
    {
      v6 = *(_QWORD *)(a1 + 24);
      v7 = sub_14C81A0(*(_QWORD *)(a2 - 48));
      if ( (unsigned __int8)sub_13848E0(v6, v2, 0, v7) )
      {
        v8 = *(_QWORD *)(a1 + 24);
        v9 = sub_14C8160();
        sub_13848E0(v8, v2, 1u, v9);
      }
    }
    if ( a2 != v2 )
      sub_1389510(a1, v2, a2, 0);
  }
  result = *(__m128i **)v3;
  if ( *(_BYTE *)(*(_QWORD *)v3 + 8LL) == 15 )
  {
    result = *(__m128i **)a2;
    if ( *(_BYTE *)(*(_QWORD *)a2 + 8LL) == 15 )
    {
      result = (__m128i *)sub_1389430(a1, v3, 0);
      if ( a2 != v3 )
        return sub_1389510(a1, v3, a2, 0);
    }
  }
  return result;
}
