// Function: sub_6FE880
// Address: 0x6fe880
//
__int64 __fastcall sub_6FE880(__m128i *a1, int a2)
{
  __int64 v4; // rcx
  __int64 v5; // rdi
  __int64 result; // rax
  __int64 i; // rsi
  __int64 v8; // rax
  __int64 v9; // rax
  void *v10; // rdx
  __int64 v11; // rax
  __int32 *v12; // r12
  __int64 v13; // rdi
  __int64 v14; // rax

  sub_6F69D0(a1, 0);
  sub_6FB450((__int64)a1, 0);
  v5 = a1->m128i_i64[0];
  result = *(unsigned __int8 *)(a1->m128i_i64[0] + 140);
  for ( i = a1->m128i_i64[0]; (_BYTE)result == 12; result = *(unsigned __int8 *)(i + 140) )
    i = *(_QWORD *)(i + 160);
  if ( (_BYTE)result == 2 )
  {
    if ( (*(_BYTE *)(i + 161) & 0x10) == 0 )
      return sub_6FC420(a1);
    if ( !a2 )
      return result;
    v8 = sub_72BA30(*(unsigned __int8 *)(i + 160));
    v9 = sub_8D6540(v8);
    return sub_6FC3F0(v9, a1, 1u);
  }
  if ( (*(_BYTE *)(i + 141) & 0x20) != 0 )
  {
    sub_6E5F60(&a1[4].m128i_i32[1], (FILE *)i, 8);
    return sub_6E6840((__int64)a1);
  }
  if ( (unsigned __int8)(result - 9) > 2u )
  {
    v9 = sub_8D6740(v5);
    return sub_6FC3F0(v9, a1, 1u);
  }
  result = (__int64)&dword_4F077C4;
  if ( dword_4F077C4 == 2 )
  {
    result = qword_4F04C50;
    if ( qword_4F04C50 )
    {
      result = *(_QWORD *)(qword_4F04C50 + 32LL);
      if ( result )
      {
        if ( (*(_BYTE *)(result + 198) & 0x10) != 0 && a2 )
        {
          v10 = &unk_4F07778;
          v11 = *(_QWORD *)i;
          if ( unk_4F07778 > 201102 || (v10 = (void *)dword_4F07774, dword_4F07774) )
          {
            v13 = *(_QWORD *)(v11 + 96);
            if ( !*(_QWORD *)(v13 + 8) || !(unsigned int)sub_879360(v13, i, v10, v4) )
            {
              if ( *(_BYTE *)(i + 140) == 12 )
              {
                v14 = i;
                do
                  v14 = *(_QWORD *)(v14 + 160);
                while ( *(_BYTE *)(v14 + 140) == 12 );
                result = *(_QWORD *)(*(_QWORD *)v14 + 96LL);
                if ( !*(_QWORD *)(result + 24) )
                  return result;
                do
                  i = *(_QWORD *)(i + 160);
                while ( *(_BYTE *)(i + 140) == 12 );
                result = *(_QWORD *)(*(_QWORD *)i + 96LL);
              }
              else
              {
                result = *(_QWORD *)(*(_QWORD *)i + 96LL);
                if ( !*(_QWORD *)(result + 24) )
                  return result;
              }
              if ( (*(_BYTE *)(result + 177) & 2) != 0 )
                return result;
            }
          }
          else
          {
            result = *(_QWORD *)(v11 + 96);
            if ( *(char *)(result + 178) < 0 )
              return result;
          }
          result = qword_4D03C50;
          if ( (*(_BYTE *)(qword_4D03C50 + 17LL) & 1) != 0 )
          {
            v12 = &a1[4].m128i_i32[1];
            result = sub_6E53E0(5, 0x50Au, v12);
            if ( (_DWORD)result )
              return sub_684B30(0x50Au, v12);
          }
        }
      }
    }
  }
  return result;
}
