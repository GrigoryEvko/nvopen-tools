// Function: sub_70AFD0
// Address: 0x70afd0
//
__int64 __fastcall sub_70AFD0(unsigned __int8 a1, const char *a2, __m128i *a3, _DWORD *a4)
{
  size_t v6; // rdx
  __int64 result; // rax
  bool v8; // cc
  size_t v9; // rax
  __m128i v10[3]; // [rsp+0h] [rbp-30h] BYREF

  v10[0] = 0;
  if ( a1 <= 1u || a1 == 10 )
  {
    v9 = strlen(a2);
    result = sub_8F59E0(v10, a2, v9);
    goto LABEL_14;
  }
  v6 = strlen(a2);
  if ( (a1 & 0xFD) == 9 || a1 == 2 )
  {
    result = sub_8F6190(v10, a2, v6);
    goto LABEL_14;
  }
  if ( (unsigned __int8)(a1 - 3) <= 1u )
    goto LABEL_21;
  if ( (unsigned __int8)(a1 - 5) <= 1u )
  {
    if ( !dword_4F07890 )
    {
LABEL_22:
      result = sub_8F8060(v10, a2, v6);
      goto LABEL_14;
    }
    goto LABEL_21;
  }
  if ( a1 != 14 && a1 > 8u )
  {
    if ( qword_4D040A0[a1] != 8 )
    {
LABEL_10:
      if ( a1 == 8 || a1 == 13 )
      {
        result = sub_8F8860(v10, a2, v6);
        goto LABEL_14;
      }
      goto LABEL_22;
    }
LABEL_21:
    result = sub_8F6940(v10, a2, v6);
    goto LABEL_14;
  }
  if ( a1 != 7 )
    goto LABEL_10;
  result = sub_8F80B0(v10, a2, v6);
LABEL_14:
  if ( HIDWORD(qword_4F077B4) )
  {
    *a4 = 0;
    goto LABEL_18;
  }
  v8 = (int)result <= 0;
  result = (int)result > 0;
  *a4 = result;
  if ( v8 )
LABEL_18:
    *a3 = _mm_loadu_si128(v10);
  return result;
}
