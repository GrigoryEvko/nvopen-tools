// Function: sub_31F0A90
// Address: 0x31f0a90
//
__int64 __fastcall sub_31F0A90(__int64 a1, unsigned int a2)
{
  __int64 v2; // r8
  char *v3; // rax
  _BYTE *v4; // rcx
  __int64 v5; // r9
  char v6; // dl
  void (*v7)(); // r10
  __m128i *v8; // rcx
  char v9; // dl
  __m128i v11; // xmm1
  void (*v12)(); // rcx
  __m128i v13; // [rsp+0h] [rbp-70h] BYREF
  __m128i v14; // [rsp+10h] [rbp-60h] BYREF
  __int64 v15; // [rsp+20h] [rbp-50h]
  __m128i v16; // [rsp+30h] [rbp-40h] BYREF
  __m128i v17; // [rsp+40h] [rbp-30h]
  __int64 v18; // [rsp+50h] [rbp-20h]

  v2 = *(_QWORD *)(a1 + 224);
  if ( *(_BYTE *)(a1 + 488) )
  {
    v3 = sub_31F08F0(a2);
    v6 = *v3;
    if ( v4 )
    {
      v7 = *(void (**)())(*(_QWORD *)v2 + 120LL);
      if ( v6 )
      {
        if ( *v4 )
        {
          v13.m128i_i64[0] = (__int64)v4;
          v14.m128i_i64[0] = (__int64)" Encoding = ";
          v8 = &v13;
          LOWORD(v15) = 771;
          v9 = 2;
        }
        else
        {
          v8 = (__m128i *)" Encoding = ";
          v5 = v13.m128i_i64[1];
          v9 = 3;
          v13.m128i_i64[0] = (__int64)" Encoding = ";
          LOWORD(v15) = 259;
        }
        v16.m128i_i64[0] = (__int64)v8;
        v16.m128i_i64[1] = v5;
        v17.m128i_i64[0] = (__int64)v3;
        LOBYTE(v18) = v9;
        BYTE1(v18) = 3;
      }
      else
      {
        if ( *v4 )
        {
          v13.m128i_i64[0] = (__int64)v4;
          v14.m128i_i64[0] = (__int64)" Encoding = ";
          LOWORD(v15) = 771;
        }
        else
        {
          v13.m128i_i64[0] = (__int64)" Encoding = ";
          LOWORD(v15) = 259;
        }
        v11 = _mm_loadu_si128(&v14);
        v16 = _mm_loadu_si128(&v13);
        v18 = v15;
        v17 = v11;
      }
      if ( v7 != nullsub_98 )
      {
        ((void (__fastcall *)(__int64, __m128i *, __int64))v7)(v2, &v16, 1);
        v2 = *(_QWORD *)(a1 + 224);
      }
    }
    else
    {
      v12 = *(void (**)())(*(_QWORD *)v2 + 120LL);
      if ( v6 )
      {
        v17.m128i_i64[0] = (__int64)v3;
        v16.m128i_i64[0] = (__int64)"Encoding = ";
        LOWORD(v18) = 771;
      }
      else
      {
        v16.m128i_i64[0] = (__int64)"Encoding = ";
        LOWORD(v18) = 259;
      }
      if ( v12 != nullsub_98 )
      {
        ((void (__fastcall *)(__int64, __m128i *, __int64))v12)(v2, &v16, 1);
        v2 = *(_QWORD *)(a1 + 224);
      }
    }
  }
  return (*(__int64 (__fastcall **)(__int64, _QWORD, __int64))(*(_QWORD *)v2 + 536LL))(v2, a2, 1);
}
