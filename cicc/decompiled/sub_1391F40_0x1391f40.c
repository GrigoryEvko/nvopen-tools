// Function: sub_1391F40
// Address: 0x1391f40
//
__int64 __fastcall sub_1391F40(__int64 a1, __int64 a2, __int64 a3, __m128i *a4)
{
  __int64 result; // rax
  unsigned __int8 v6; // al
  __int64 v9; // rbx
  __int64 v10; // rax
  __int64 v11; // rdi
  __int64 v12; // rdx
  __int64 v13; // rdx
  __int64 v14; // rbx
  __int64 v15; // rax

  result = *(_QWORD *)a2;
  if ( *(_BYTE *)(*(_QWORD *)a2 + 8LL) == 15 )
  {
    result = *(_QWORD *)a3;
    if ( *(_BYTE *)(*(_QWORD *)a3 + 8LL) == 15 )
    {
      v6 = *(_BYTE *)(a2 + 16);
      if ( v6 > 3u )
      {
        if ( v6 == 5 )
        {
          result = (unsigned int)*(unsigned __int16 *)(a2 + 18) - 51;
          if ( (unsigned int)result > 1 )
          {
            result = sub_13848E0(*(_QWORD *)(a1 + 24), a2, 0, 0);
            if ( (_BYTE)result )
              result = sub_1391610(a1, a2, v13);
          }
        }
        else
        {
          result = sub_13848E0(*(_QWORD *)(a1 + 24), a2, 0, 0);
        }
      }
      else
      {
        v9 = *(_QWORD *)(a1 + 24);
        v10 = sub_14C81A0(a2);
        v11 = v9;
        result = sub_13848E0(v9, a2, 0, v10);
        if ( (_BYTE)result )
        {
          v14 = *(_QWORD *)(a1 + 24);
          v15 = sub_14C8160(v11, a2, v12);
          result = sub_13848E0(v14, a2, 1u, v15);
        }
      }
      if ( a2 != a3 )
        return (__int64)sub_1391C50(a1, a2, a3, a4);
    }
  }
  return result;
}
