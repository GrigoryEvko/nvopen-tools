// Function: sub_2124800
// Address: 0x2124800
//
__int64 __fastcall sub_2124800(__int64 a1, unsigned __int64 a2, unsigned int a3, __m128i a4, __m128i a5, __m128i a6)
{
  __int16 v6; // ax
  __int64 v7; // r14
  __int64 v9; // rcx
  const __m128i *v10; // r9
  unsigned int v11; // edx
  unsigned __int64 v12; // r8
  unsigned int v13; // edx
  unsigned int v14; // edx
  unsigned int v15; // edx
  unsigned int v16; // edx
  unsigned int v17; // edx
  unsigned int v18; // edx
  unsigned int v19; // edx
  unsigned int v20; // edx
  __int64 v21; // rax
  unsigned int v22; // edx
  unsigned int v23; // edx
  unsigned __int8 *v24; // rax
  __int64 v25; // r12
  unsigned int v26; // edx
  _BYTE v27[16]; // [rsp+D0h] [rbp-40h] BYREF

  v6 = *(_WORD *)(a2 + 24);
  if ( v6 != 46 )
  {
    v7 = a3;
    switch ( v6 )
    {
      case 101:
        v9 = (__int64)sub_2123FF0(a1, a2);
        v12 = v14;
        goto LABEL_5;
      case 134:
        v9 = (__int64)sub_21242A0(a1, (__int64 *)a2);
        v12 = v15;
        goto LABEL_5;
      case 136:
        v9 = (__int64)sub_2124340((__m128i **)a1, a2, *(double *)a4.m128i_i64, *(double *)a5.m128i_i64, a6);
        v12 = v16;
        goto LABEL_5;
      case 137:
        v9 = (__int64)sub_2124550((__m128i **)a1, a2, *(double *)a4.m128i_i64, *(double *)a5.m128i_i64, a6);
        v12 = v17;
        goto LABEL_5;
      case 152:
      case 153:
        v9 = sub_2124080(a1, a2, a4, a5, a6);
        v12 = v11;
        goto LABEL_5;
      case 154:
      case 161:
        v9 = sub_2123C50((__m128i **)a1, a2, a4, a5, a6);
        v12 = v13;
        goto LABEL_5;
      case 157:
        v9 = sub_2123AD0((__m128i **)a1, a2, a4, a5, a6);
        v12 = v18;
        goto LABEL_5;
      case 158:
        v9 = sub_2123930(a1, a2, *(double *)a4.m128i_i64, *(double *)a5.m128i_i64, *(double *)a6.m128i_i64);
        v12 = v19;
        goto LABEL_5;
      case 162:
      case 163:
        v9 = (__int64)sub_2123F90(a1, a2);
        v12 = v20;
        goto LABEL_5;
      case 186:
        v21 = sub_21246C0(a1, a2, a4, *(double *)a5.m128i_i64, a6);
        v9 = v21;
        v12 = v22;
        if ( a2 != v21 )
        {
          if ( v21 )
            goto LABEL_7;
          return 0;
        }
        v24 = (unsigned __int8 *)(*(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a2 + 32) + 40 * v7) + 40LL)
                                + 16LL * *(unsigned int *)(*(_QWORD *)(a2 + 32) + 40 * v7 + 8));
        v25 = *v24;
        sub_1F40D10(
          (__int64)v27,
          *(_QWORD *)a1,
          *(_QWORD *)(*(_QWORD *)(a1 + 8) + 48LL),
          (unsigned __int8)v25,
          *((_QWORD *)v24 + 1));
        if ( (_BYTE)v25 == v27[8] && (_BYTE)v25 && *(_QWORD *)(*(_QWORD *)a1 + 8 * v25 + 120) )
          return 0;
        break;
      case 192:
        v9 = (__int64)sub_2123D70((__m128i **)a1, a2, *(double *)a4.m128i_i64, *(double *)a5.m128i_i64, a6);
        v12 = v23;
        goto LABEL_5;
      default:
        sub_211BAF0((__int64 *)a1, a2, a3);
        return 0;
    }
    return 1;
  }
  v9 = (__int64)sub_21239F0(a1, a2);
  v12 = v26;
LABEL_5:
  if ( !v9 )
    return 0;
  if ( a2 != v9 )
  {
LABEL_7:
    sub_2013400(a1, a2, 0, v9, (__m128i *)v12, v10);
    return 0;
  }
  return 1;
}
