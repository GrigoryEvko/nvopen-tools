// Function: sub_68A310
// Address: 0x68a310
//
__int64 __fastcall sub_68A310(__m128i *a1)
{
  __int64 v2; // rdi
  __int64 v3; // rbx
  const __m128i *v4; // rax
  __int64 v5; // rax
  __int64 v6; // rax
  _DWORD *v7; // rbx
  __int64 result; // rax
  char v9; // al
  __int64 v10; // r12
  __int64 v11; // rax
  __int64 v12; // rbx
  char i; // dl

  while ( 1 )
  {
    while ( a1[1].m128i_i8[8] == 2 )
    {
      v2 = a1[3].m128i_i64[1];
      if ( *(_BYTE *)(v2 + 173) != 12 || *(_BYTE *)(v2 + 176) != 1 )
        break;
      v3 = a1[1].m128i_i64[0];
      v4 = (const __m128i *)sub_72E9A0();
      *a1 = _mm_loadu_si128(v4);
      a1[1] = _mm_loadu_si128(v4 + 1);
      a1[2] = _mm_loadu_si128(v4 + 2);
      a1[3] = _mm_loadu_si128(v4 + 3);
      a1[4] = _mm_loadu_si128(v4 + 4);
      v5 = v4[5].m128i_i64[0];
      a1[1].m128i_i64[0] = v3;
      a1[5].m128i_i64[0] = v5;
    }
    v6 = sub_6E4240(a1, 0);
    v7 = (_DWORD *)v6;
    if ( *(_BYTE *)(v6 + 24) != 1 )
      break;
    v9 = *(_BYTE *)(v6 + 56);
    if ( (unsigned __int8)(v9 - 87) <= 1u )
    {
      v12 = *((_QWORD *)v7 + 9);
      sub_68A310(v12);
      a1 = *(__m128i **)(v12 + 16);
    }
    else
    {
      if ( (unsigned __int8)(v9 - 105) > 4u )
        break;
      if ( (*((_BYTE *)v7 + 59) & 8) == 0 )
        break;
      v10 = *((_QWORD *)v7 + 9);
      v11 = sub_72B0F0(v10, 0);
      if ( *(_BYTE *)(v11 + 174) != 5 || (unsigned __int8)(*(_BYTE *)(v11 + 176) - 35) > 1u )
        break;
      sub_68A310(*(_QWORD *)(v10 + 16));
      a1 = *(__m128i **)(*(_QWORD *)(v10 + 16) + 16LL);
    }
  }
  result = sub_8D29A0(*(_QWORD *)v7);
  if ( !(_DWORD)result )
  {
    result = sub_8D3D40(*(_QWORD *)v7);
    if ( !(_DWORD)result )
    {
      result = *(_QWORD *)v7;
      for ( i = *(_BYTE *)(*(_QWORD *)v7 + 140LL); i == 12; i = *(_BYTE *)(result + 140) )
        result = *(_QWORD *)(result + 160);
      if ( i )
      {
        sub_6851C0(0xBE0u, v7 + 7);
        sub_7264E0(v7, 0);
        result = sub_72C930(v7);
        *(_QWORD *)v7 = result;
      }
    }
  }
  return result;
}
