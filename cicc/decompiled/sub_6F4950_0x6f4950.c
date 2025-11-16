// Function: sub_6F4950
// Address: 0x6f4950
//
__m128i *__fastcall sub_6F4950(__m128i *a1, __int64 a2)
{
  __int8 v3; // al
  __m128i *result; // rax
  __int64 v5; // r13
  __int64 v6; // r15
  __int64 v7; // rax
  __int64 v8; // rdi
  __int64 v9; // rax
  _DWORD *v10; // r14
  __m128i v11[4]; // [rsp+0h] [rbp-40h] BYREF

  v3 = a1[1].m128i_i8[0];
  switch ( v3 )
  {
    case 1:
      if ( !word_4D04898 )
        goto LABEL_4;
      v8 = a1[9].m128i_i64[0];
      v11[0] = 0u;
      if ( (unsigned int)sub_7A30C0(v8, 1, 0, a2) )
      {
        if ( !*(_BYTE *)(qword_4D03C50 + 16LL) )
          *(_QWORD *)(a2 + 144) = 0;
      }
      else if ( (dword_4F04C44 != -1
              || (v9 = qword_4F04C68[0] + 776LL * dword_4F04C64, (*(_BYTE *)(v9 + 6) & 0x12) != 0)
              || (*(_BYTE *)(v9 + 6) & 4) != 0 && (*(_BYTE *)(v9 + 12) & 0x10) == 0)
             && (unsigned int)sub_696840((__int64)a1) )
      {
        sub_6F4910(a1, a2, 0);
      }
      else
      {
        if ( (unsigned int)sub_6E5430() )
        {
          v10 = sub_67D9D0(0x1Cu, &a1[4].m128i_i32[1]);
          sub_67E370((__int64)v10, v11);
          sub_685910((__int64)v10, (FILE *)v11);
        }
        sub_72C970(a2);
      }
      sub_67E3D0(v11);
      break;
    case 2:
      sub_72A510(&a1[9], a2);
      if ( (a1[1].m128i_i8[4] & 4) == 0 )
        goto LABEL_6;
      result = (__m128i *)qword_4D03C50;
      if ( *(_BYTE *)(a2 + 173) == 12 || !*(_BYTE *)(qword_4D03C50 + 16LL) )
        goto LABEL_7;
      v5 = sub_73A460(&a1[9]);
      v6 = sub_73A720(v5);
      v7 = sub_7CADA0(v5, &a1[1].m128i_u64[1]);
      *(_QWORD *)(v6 + 64) = v7;
      if ( v7 )
      {
        *(_BYTE *)(a2 + 170) &= ~0x10u;
        result = (__m128i *)qword_4D03C50;
        *(_QWORD *)(a2 + 144) = v6;
        goto LABEL_7;
      }
      break;
    case 0:
LABEL_5:
      sub_72C970(a2);
LABEL_6:
      result = (__m128i *)qword_4D03C50;
      goto LABEL_7;
    default:
LABEL_4:
      sub_6E68E0(0x1Cu, (__int64)a1);
      goto LABEL_5;
  }
  result = (__m128i *)qword_4D03C50;
LABEL_7:
  if ( (result[1].m128i_i8[3] & 2) != 0 )
  {
    result = sub_6E3700(a1, 0);
    *(_QWORD *)(a2 + 152) = result;
  }
  return result;
}
