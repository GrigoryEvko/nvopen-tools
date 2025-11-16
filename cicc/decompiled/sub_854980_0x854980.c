// Function: sub_854980
// Address: 0x854980
//
_QWORD *__fastcall sub_854980(__int64 a1, __int64 a2)
{
  _QWORD *result; // rax
  __m128i *v3; // r14
  __m128i *v5; // rbx
  bool v6; // r13
  void (__fastcall *v7)(__m128i *, __int64, __int64); // rcx
  unsigned __int8 *v8; // rdx
  unsigned __int8 v9; // al
  unsigned __int8 v10; // di
  void (__fastcall *v11)(__m128i *, __int64, __int64); // [rsp-40h] [rbp-40h]

  result = (_QWORD *)(qword_4F04C68[0] + 776LL * dword_4F04C64);
  if ( (*((_BYTE *)result + 12) & 4) == 0 )
  {
    v3 = (__m128i *)result[55];
    result[55] = 0;
    if ( v3 )
    {
      v5 = v3;
      v6 = a1 != 0;
      while ( 1 )
      {
        v8 = (unsigned __int8 *)v5->m128i_i64[1];
        v9 = v8[17];
        if ( (v9 & 1) != 0 && v6 )
          break;
        if ( (v9 & 2) != 0 && a2 )
          break;
        v10 = v8[19];
        if ( v10 == 3 )
        {
LABEL_9:
          v5 = (__m128i *)v5->m128i_i64[0];
          if ( !v5 )
            return sub_854000(v3);
        }
        else
        {
          sub_684AA0(v10, 607 - (v8[17] & 1), (__m128i *)v5[3].m128i_i32);
          v5 = (__m128i *)v5->m128i_i64[0];
          if ( !v5 )
            return sub_854000(v3);
        }
      }
      v7 = (void (__fastcall *)(__m128i *, __int64, __int64))off_4A51EC0[v8[16]];
      if ( (v9 & 8) != 0 )
      {
        v11 = (void (__fastcall *)(__m128i *, __int64, __int64))off_4A51EC0[v8[16]];
        sub_8543B0(v5, a1, a2);
        v7 = v11;
      }
      if ( v7 )
        v7(v5, a1, a2);
      goto LABEL_9;
    }
  }
  return result;
}
