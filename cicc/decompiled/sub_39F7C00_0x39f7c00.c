// Function: sub_39F7C00
// Address: 0x39f7c00
//
__int64 __fastcall sub_39F7C00(_QWORD *a1, __m128i *a2, __int64 *a3)
{
  __int64 i; // r13
  int v4; // eax
  int v5; // r15d
  __int64 result; // rax
  _QWORD *v7; // rdx
  char v9[336]; // [rsp+10h] [rbp-1B8h] BYREF
  __int64 (__fastcall *v10)(__int64, _QWORD, _QWORD, _QWORD *, __m128i *); // [rsp+160h] [rbp-68h]
  __int64 v11; // [rsp+178h] [rbp-50h]

  for ( i = 1; ; ++i )
  {
    v4 = sub_39F7420(a2, v9);
    v5 = 4 * (a1[3] == a2[9].m128i_i64[0] - ((unsigned __int64)a2[12].m128i_i64[0] >> 63));
    if ( v4 )
      return 2;
    if ( v10 )
      break;
LABEL_8:
    if ( v5 )
      goto LABEL_18;
    sub_39F6770(a2, (__int64)v9);
    if ( *(_DWORD *)&v9[16 * v11 + 8] == 6 )
    {
      a2[9].m128i_i64[1] = 0;
    }
    else
    {
      if ( (int)v11 > 17 )
        goto LABEL_18;
      v7 = (_QWORD *)a2->m128i_i64[(int)v11];
      if ( (a2[12].m128i_i8[7] & 0x40) == 0 || !a2[13].m128i_i8[(int)v11 + 8] )
      {
        if ( byte_5057700[(int)v11] != 8 )
LABEL_18:
          abort();
        v7 = (_QWORD *)*v7;
      }
      a2[9].m128i_i64[1] = (__int64)v7;
    }
  }
  result = v10(1, v5 | 2u, *a1, a1, a2);
  if ( (_DWORD)result != 7 )
  {
    if ( (_DWORD)result != 8 )
      return 2;
    goto LABEL_8;
  }
  *a3 = i;
  return result;
}
