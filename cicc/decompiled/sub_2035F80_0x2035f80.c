// Function: sub_2035F80
// Address: 0x2035f80
//
__int64 __fastcall sub_2035F80(
        __int64 a1,
        unsigned __int64 a2,
        __int64 a3,
        __int64 a4,
        int a5,
        int a6,
        __m128 a7,
        double a8,
        __m128i a9)
{
  __int64 v9; // rcx
  const __m128i *v10; // r9
  unsigned int v11; // edx
  unsigned __int64 v12; // r8
  __int64 result; // rax
  unsigned int v14; // edx
  unsigned int v15; // edx
  unsigned int v16; // edx
  unsigned int v17; // edx
  unsigned int v18; // edx
  unsigned int v19; // edx
  unsigned int v20; // edx
  unsigned int v21; // edx

  switch ( *(_WORD *)(a2 + 24) )
  {
    case 0x6A:
      v9 = sub_2035500(a1, a2, *(double *)a7.m128_u64, a8, *(double *)a9.m128i_i64);
      v12 = v14;
      if ( !v9 )
        return 0;
      goto LABEL_4;
    case 0x6B:
      v9 = (__int64)sub_2035340(a1, a2, *(double *)a7.m128_u64, a8, a9, a3, a4, a5, a6);
      v12 = v18;
      goto LABEL_3;
    case 0x86:
      v9 = sub_2035010(a1, a2, a3, *(double *)a7.m128_u64, a8, *(double *)a9.m128i_i64);
      v12 = v19;
      goto LABEL_3;
    case 0x87:
      v9 = (__int64)sub_20356B0(a1, a2, a7, a8, a9);
      v12 = v20;
      goto LABEL_3;
    case 0x89:
      v9 = sub_2035790(a1, a2, a7, a8, a9);
      v12 = v17;
      goto LABEL_3;
    case 0x8E:
    case 0x8F:
    case 0x90:
    case 0x91:
    case 0x92:
    case 0x93:
    case 0x98:
    case 0x99:
      v9 = sub_20351B0(a1, a2, *(double *)a7.m128_u64, a8, *(double *)a9.m128i_i64);
      v12 = v11;
      goto LABEL_3;
    case 0x9A:
      v9 = sub_2035D60(a1, a2, *(double *)a7.m128_u64, a8, a9, a3, a4);
      v12 = v21;
      goto LABEL_3;
    case 0x9E:
      v9 = sub_20350F0(a1, a2, *(double *)a7.m128_u64, a8, *(double *)a9.m128i_i64);
      v12 = v16;
      goto LABEL_3;
    case 0xBA:
      v9 = sub_2035AA0(a1, a2);
      v12 = v15;
LABEL_3:
      if ( !v9 )
        return 0;
LABEL_4:
      result = 1;
      if ( a2 != v9 )
      {
        sub_2013400(a1, a2, 0, v9, (__m128i *)v12, v10);
        return 0;
      }
      return result;
    default:
      sub_16BD130("Do not know how to scalarize this operator's operand!\n", 1u);
  }
}
