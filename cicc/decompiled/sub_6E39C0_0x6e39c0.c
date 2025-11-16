// Function: sub_6E39C0
// Address: 0x6e39c0
//
__int64 __fastcall sub_6E39C0(const __m128i *a1, __int64 a2)
{
  __m128i *v2; // rax
  __int64 v3; // rdi
  __m128i *v4; // r12
  __int64 result; // rax
  __int64 v6; // rdi
  __int64 v7; // rdi

  v2 = sub_6E3700(a1, *(__m128i **)(a2 + 80));
  v3 = a2;
  *(_QWORD *)(a2 + 80) = v2;
  v4 = v2;
  result = *(unsigned __int8 *)(a2 + 24);
  if ( (_BYTE)result == 2 )
  {
    v7 = *(_QWORD *)(a2 + 56);
    if ( *(_BYTE *)(v7 + 173) != 12 || *(_BYTE *)(v7 + 176) != 1 )
      return result;
    v3 = sub_72E9A0(v7);
    LODWORD(result) = *(unsigned __int8 *)(v3 + 24);
  }
  if ( (_BYTE)result == 1 )
  {
    result = sub_730740(v3);
    if ( !(_DWORD)result || v4[23].m128i_i64[1] )
      return result;
  }
  else
  {
    result = (unsigned int)result & 0xFFFFFFFD;
    if ( (_BYTE)result != 5 || v4[23].m128i_i64[1] )
      return result;
  }
  if ( *(_BYTE *)(a2 + 24) == 7 )
  {
    result = *(_QWORD *)(*(_QWORD *)(a2 + 56) + 8LL);
    v4[23].m128i_i64[1] = result;
  }
  else
  {
    v6 = *(_QWORD *)(a2 + 8);
    if ( v6 )
    {
      result = *(unsigned __int8 *)(a2 + 58);
      if ( (result & 0x10) != 0 )
      {
        if ( (result & 0x20) != 0 )
          result = sub_72D6A0(v6);
        else
          result = sub_72D600(v6);
        v4[23].m128i_i64[1] = result;
      }
      else
      {
        v4[23].m128i_i64[1] = v6;
      }
    }
    else
    {
      result = *(_QWORD *)a2;
      v4[23].m128i_i64[1] = *(_QWORD *)a2;
    }
  }
  return result;
}
