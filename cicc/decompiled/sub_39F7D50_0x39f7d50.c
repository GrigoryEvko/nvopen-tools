// Function: sub_39F7D50
// Address: 0x39f7d50
//
__int64 __fastcall sub_39F7D50(_QWORD *a1, __m128i *a2, __int64 *a3)
{
  __int64 v3; // rbp
  __int64 v4; // r14
  unsigned int v5; // eax
  unsigned int v6; // r15d
  __int64 v7; // rdx
  unsigned int v8; // eax
  _QWORD *v9; // rdx
  unsigned int (__fastcall *v11)(__int64, __int64, __int64, _QWORD *, __m128i *, __int64); // [rsp+0h] [rbp-1C8h]
  char v13[336]; // [rsp+10h] [rbp-1B8h] BYREF
  __int64 (__fastcall *v14)(__int64, __int64, _QWORD, _QWORD *, __m128i *); // [rsp+160h] [rbp-68h]
  __int64 v15; // [rsp+178h] [rbp-50h]

  v3 = 1;
  v4 = a1[3];
  v11 = (unsigned int (__fastcall *)(__int64, __int64, __int64, _QWORD *, __m128i *, __int64))a1[2];
  while ( 1 )
  {
    v5 = sub_39F7420(a2, v13);
    v6 = v5;
    if ( v5 && v5 != 5 )
      return 2;
    v7 = *a1;
    if ( v5 == 5 )
      break;
    if ( v11(1, 10, v7, a1, a2, v4) )
      return 2;
    if ( v14 )
    {
      v8 = v14(1, 10, *a1, a1, a2);
      v6 = v8;
      if ( v8 == 7 )
        goto LABEL_21;
      if ( v8 != 8 )
        return 2;
    }
    sub_39F6770(a2, (__int64)v13);
    if ( *(_DWORD *)&v13[16 * v15 + 8] == 6 )
    {
      a2[9].m128i_i64[1] = 0;
    }
    else
    {
      if ( (int)v15 > 17 )
        goto LABEL_22;
      v9 = (_QWORD *)a2->m128i_i64[(int)v15];
      if ( (a2[12].m128i_i8[7] & 0x40) == 0 || !a2[13].m128i_i8[(int)v15 + 8] )
      {
        if ( byte_5057700[(int)v15] != 8 )
LABEL_22:
          abort();
        v9 = (_QWORD *)*v9;
      }
      a2[9].m128i_i64[1] = (__int64)v9;
    }
    ++v3;
  }
  if ( v11(1, 26, v7, a1, a2, v4) )
    return 2;
LABEL_21:
  *a3 = v3;
  return v6;
}
