// Function: sub_38B8180
// Address: 0x38b8180
//
__int64 __fastcall sub_38B8180(
        __int64 a1,
        __m128 a2,
        __m128i a3,
        double a4,
        double a5,
        double a6,
        double a7,
        double a8,
        __m128 a9)
{
  unsigned int v9; // eax
  bool v10; // cc
  unsigned __int64 v12; // rsi
  int i; // eax
  const char *v14; // [rsp+0h] [rbp-30h] BYREF
  char v15; // [rsp+10h] [rbp-20h]
  char v16; // [rsp+11h] [rbp-1Fh]

  if ( *(_QWORD *)(a1 + 176) )
  {
    while ( 1 )
    {
LABEL_2:
      while ( 1 )
      {
        v9 = *(_DWORD *)(a1 + 64);
        v10 = v9 <= 0x12D;
        if ( v9 != 301 )
          break;
LABEL_17:
        if ( (unsigned __int8)sub_38B7E70(a1, 0, *(double *)a2.m128_u64, *(double *)a3.m128i_i64, a4) )
          return 1;
      }
      while ( 1 )
      {
        if ( !v10 )
        {
          if ( v9 != 302 )
          {
            switch ( v9 )
            {
              case 0x170u:
                if ( (unsigned __int8)sub_38AAF20(a1, a2, *(double *)a3.m128i_i64, a4, a5, a6, a7, a8, a9) )
                  return 1;
                goto LABEL_2;
              case 0x171u:
                if ( (unsigned __int8)sub_38936D0(a1) )
                  return 1;
                goto LABEL_2;
              case 0x173u:
                if ( (unsigned __int8)sub_38B6E00(a1) )
                  return 1;
                goto LABEL_2;
              case 0x175u:
                if ( (unsigned __int8)sub_38AB100(a1, a2, *(double *)a3.m128i_i64, a4, a5, a6, a7, a8, a9) )
                  return 1;
                goto LABEL_2;
              case 0x176u:
                if ( (unsigned __int8)sub_3893D60(a1) )
                  return 1;
                goto LABEL_2;
              case 0x177u:
                if ( (unsigned __int8)sub_3892870(a1) )
                  return 1;
                goto LABEL_2;
              case 0x178u:
                if ( (unsigned __int8)sub_3897F00(a1, a2, *(double *)a3.m128i_i64, a4, a5, a6, a7, a8, a9) )
                  return 1;
                goto LABEL_2;
              default:
                goto LABEL_43;
            }
          }
          if ( !(unsigned __int8)sub_38B7840(a1, *(double *)a2.m128_u64, *(double *)a3.m128i_i64, a4) )
            goto LABEL_2;
          return 1;
        }
        if ( v9 == 60 )
        {
          if ( (unsigned __int8)sub_388B2E0(a1) )
            return 1;
          goto LABEL_2;
        }
        if ( v9 <= 0x3C )
          break;
        if ( v9 == 93 )
        {
          if ( (unsigned __int8)sub_388B7A0(a1) )
            return 1;
          goto LABEL_2;
        }
        if ( v9 > 0x5D )
        {
          if ( v9 == 142 )
          {
            if ( !(unsigned __int8)sub_38975B0(a1) )
              goto LABEL_2;
            return 1;
          }
          goto LABEL_43;
        }
        if ( v9 == 62 )
        {
          if ( (unsigned __int8)sub_388B5F0(a1) )
            return 1;
          goto LABEL_2;
        }
        if ( v9 != 64 )
          goto LABEL_43;
        if ( (unsigned __int8)sub_388B120(a1) )
          return 1;
        v9 = *(_DWORD *)(a1 + 64);
        v10 = v9 <= 0x12D;
        if ( v9 == 301 )
          goto LABEL_17;
      }
      if ( v9 != 20 )
        break;
      if ( (unsigned __int8)sub_38AA300(a1, a2, *(double *)a3.m128i_i64, a4, a5, a6, a7, a8, a9) )
        return 1;
    }
    if ( v9 > 0x14 )
    {
      if ( v9 == 21 )
      {
        if ( !(unsigned __int8)sub_38B8110(a1, a2, a3, a4, a5, a6, a7, a8, a9) )
          goto LABEL_2;
        return 1;
      }
LABEL_43:
      v12 = *(_QWORD *)(a1 + 56);
      v16 = 1;
      v15 = 3;
      v14 = "expected top-level entity";
      return sub_38814C0(a1 + 8, v12, (__int64)&v14);
    }
    if ( !v9 )
      return 0;
    if ( v9 != 14 )
      goto LABEL_43;
    if ( !(unsigned __int8)sub_38A9F20(a1, a2, *(double *)a3.m128i_i64, a4, a5, a6, a7, a8, a9) )
      goto LABEL_2;
    return 1;
  }
  else
  {
    for ( i = *(_DWORD *)(a1 + 64); i == 62; i = *(_DWORD *)(a1 + 64) )
    {
LABEL_49:
      if ( (unsigned __int8)sub_388B5F0(a1) )
        return 1;
LABEL_50:
      ;
    }
    while ( 1 )
    {
      if ( i == 371 )
      {
        if ( (unsigned __int8)sub_38B6E00(a1) )
          return 1;
        goto LABEL_50;
      }
      if ( !i )
        return 0;
      i = sub_3887100(a1 + 8);
      *(_DWORD *)(a1 + 64) = i;
      if ( i == 62 )
        goto LABEL_49;
    }
  }
}
