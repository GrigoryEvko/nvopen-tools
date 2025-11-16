// Function: sub_397C150
// Address: 0x397c150
//
__int64 __fastcall sub_397C150(__int64 a1, unsigned int a2)
{
  __int64 v2; // r8
  char *v3; // rax
  _BYTE *v4; // rcx
  char v5; // dl
  void (*v6)(); // r9
  __m128i *v7; // rcx
  char v8; // dl
  void (*v10)(); // rcx
  __m128i v11; // [rsp+0h] [rbp-50h] BYREF
  __int64 v12; // [rsp+10h] [rbp-40h]
  __m128i v13; // [rsp+20h] [rbp-30h] BYREF
  __int64 v14; // [rsp+30h] [rbp-20h]

  v2 = *(_QWORD *)(a1 + 256);
  if ( *(_BYTE *)(a1 + 416) )
  {
    v3 = sub_397BEF0(a2);
    v5 = *v3;
    if ( v4 )
    {
      v6 = *(void (**)())(*(_QWORD *)v2 + 104LL);
      if ( v5 )
      {
        if ( *v4 )
        {
          v11.m128i_i64[0] = (__int64)v4;
          v11.m128i_i64[1] = (__int64)" Encoding = ";
          v7 = &v11;
          LOWORD(v12) = 771;
          v8 = 2;
        }
        else
        {
          v7 = (__m128i *)" Encoding = ";
          v8 = 3;
          v11.m128i_i64[0] = (__int64)" Encoding = ";
          LOWORD(v12) = 259;
        }
        v13.m128i_i64[0] = (__int64)v7;
        v13.m128i_i64[1] = (__int64)v3;
        LOBYTE(v14) = v8;
        BYTE1(v14) = 3;
      }
      else
      {
        if ( *v4 )
        {
          v11.m128i_i64[0] = (__int64)v4;
          v11.m128i_i64[1] = (__int64)" Encoding = ";
          LOWORD(v12) = 771;
        }
        else
        {
          v11.m128i_i64[0] = (__int64)" Encoding = ";
          LOWORD(v12) = 259;
        }
        v13 = _mm_loadu_si128(&v11);
        v14 = v12;
      }
      if ( v6 != nullsub_580 )
      {
        ((void (__fastcall *)(__int64, __m128i *, __int64))v6)(v2, &v13, 1);
        v2 = *(_QWORD *)(a1 + 256);
      }
    }
    else
    {
      v10 = *(void (**)())(*(_QWORD *)v2 + 104LL);
      if ( v5 )
      {
        v13.m128i_i64[1] = (__int64)v3;
        v13.m128i_i64[0] = (__int64)"Encoding = ";
        LOWORD(v14) = 771;
      }
      else
      {
        v13.m128i_i64[0] = (__int64)"Encoding = ";
        LOWORD(v14) = 259;
      }
      if ( v10 != nullsub_580 )
      {
        ((void (__fastcall *)(__int64, __m128i *, __int64))v10)(v2, &v13, 1);
        v2 = *(_QWORD *)(a1 + 256);
      }
    }
  }
  return (*(__int64 (__fastcall **)(__int64, _QWORD, __int64))(*(_QWORD *)v2 + 424LL))(v2, a2, 1);
}
