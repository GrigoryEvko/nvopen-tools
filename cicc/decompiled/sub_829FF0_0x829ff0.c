// Function: sub_829FF0
// Address: 0x829ff0
//
__int64 __fastcall sub_829FF0(__m128i *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rbx
  char v7; // al
  char v8; // al
  __int64 result; // rax
  __int8 v10; // dl
  __int64 v11; // rdi
  char v12; // dl

  if ( unk_4D0480C && (unsigned int)sub_6F4D20(a1, 0, 1, a4, a5, a6) )
  {
    if ( (unsigned int)sub_8D32E0(a1->m128i_i64[0]) )
      a1->m128i_i64[0] = sub_8D46C0(a1->m128i_i64[0]);
    v10 = a1[19].m128i_i8[13];
    result = 0;
    if ( v10 != 12 )
    {
      result = 1;
      if ( v10 == 6 )
        return (unsigned int)sub_828980((__int64)a1[9].m128i_i64) ^ 1;
    }
  }
  else
  {
    v6 = a1[9].m128i_i64[0];
    v7 = *(_BYTE *)(v6 + 24);
    if ( v7 == 1 )
    {
      while ( *(_BYTE *)(v6 + 56) == 4 )
      {
        v6 = *(_QWORD *)(v6 + 72);
        v7 = *(_BYTE *)(v6 + 24);
        if ( v7 != 1 )
          goto LABEL_12;
      }
    }
    else
    {
LABEL_12:
      if ( v7 == 3
        && !(unsigned int)sub_8D32E0(*(_QWORD *)(*(_QWORD *)(v6 + 56) + 120LL))
        && *(_BYTE *)(*(_QWORD *)(v6 + 56) + 177LL) != 5 )
      {
        return 0;
      }
    }
    v8 = *(_BYTE *)(v6 + 24);
    if ( v8 == 2 )
    {
      v11 = *(_QWORD *)(v6 + 56);
      result = 0;
      v12 = *(_BYTE *)(v11 + 173);
      if ( v12 != 12 )
      {
        result = 1;
        if ( v12 == 6 )
          return (unsigned int)sub_828980(v11) ^ 1;
      }
    }
    else
    {
      return (v8 != 20 || (*(_BYTE *)(v6 + 25) & 1) == 0) && v8 != 0;
    }
  }
  return result;
}
