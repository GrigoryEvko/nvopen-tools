// Function: sub_27B9170
// Address: 0x27b9170
//
__int64 __fastcall sub_27B9170(__int64 a1, __int64 a2, __int64 **a3)
{
  __int64 v5; // r15
  __int64 v6; // r14
  __int64 v7; // rdx
  __int64 v8; // rax
  __int64 v9; // rax
  __int64 v10; // [rsp+8h] [rbp-58h]
  __m128i v11; // [rsp+10h] [rbp-50h] BYREF
  __int64 v12; // [rsp+20h] [rbp-40h]

  if ( *(_BYTE *)a2 <= 0x1Cu )
  {
    v9 = sub_AA5BA0(**a3);
    *(_BYTE *)(a1 + 16) = 1;
    if ( v9 )
      v9 -= 24;
    *(_QWORD *)a1 = v9 + 24;
    *(_WORD *)(a1 + 8) = 0;
  }
  else
  {
    sub_B445D0((__int64)&v11, (char *)a2);
    if ( (_BYTE)v12 )
    {
      v5 = v11.m128i_i64[0];
      if ( v11.m128i_i64[0] )
        v5 = v11.m128i_i64[0] - 24;
      if ( (unsigned __int8)sub_B19DB0((__int64)a3, a2, v5) )
      {
        v6 = *(_QWORD *)(a2 + 16);
        if ( !v6 )
        {
LABEL_15:
          v8 = v12;
          *(__m128i *)a1 = _mm_loadu_si128(&v11);
          *(_QWORD *)(a1 + 16) = v8;
          return a1;
        }
        while ( 1 )
        {
          v7 = *(_QWORD *)(v6 + 24);
          if ( v5 != v7 )
          {
            v10 = *(_QWORD *)(v6 + 24);
            if ( (unsigned __int8)sub_B19DB0((__int64)a3, a2, v7) )
            {
              if ( !(unsigned __int8)sub_B19DB0((__int64)a3, v5, v10) )
                break;
            }
          }
          v6 = *(_QWORD *)(v6 + 8);
          if ( !v6 )
            goto LABEL_15;
        }
      }
    }
    *(_BYTE *)(a1 + 16) = 0;
  }
  return a1;
}
