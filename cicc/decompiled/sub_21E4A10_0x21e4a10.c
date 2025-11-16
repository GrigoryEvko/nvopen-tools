// Function: sub_21E4A10
// Address: 0x21e4a10
//
__int64 __fastcall sub_21E4A10(
        __int64 a1,
        __int64 a2,
        __m128i a3,
        __m128i a4,
        __m128i a5,
        __int64 a6,
        __int64 a7,
        __int64 a8,
        __int64 a9)
{
  __int64 v9; // rax
  _QWORD *v10; // rdx
  __int64 result; // rax
  __int16 v12; // r9
  bool v13; // cf
  __int16 v14; // r9
  bool v15; // cf
  __int16 v16; // r9
  bool v17; // cf
  __int16 v18; // r9
  bool v19; // cf

  v9 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a2 + 32) + 40LL) + 88LL);
  v10 = *(_QWORD **)(v9 + 24);
  if ( *(_DWORD *)(v9 + 32) > 0x40u )
    v10 = (_QWORD *)*v10;
  if ( (unsigned int)v10 > 0x1135 )
  {
    if ( (_DWORD)v10 == 5301 )
      return sub_21E0290(a1, a2);
    if ( (unsigned int)v10 <= 0x14B5 )
    {
      if ( (_DWORD)v10 == 5293 )
        return sub_21E0110(a1, a2);
      if ( (_DWORD)v10 == 5300 )
        return sub_21E01D0(a1, a2);
    }
    else if ( (unsigned int)((_DWORD)v10 - 5304) <= 0x8F )
    {
      return sub_21E3310(a1, a2, (int)v10, a3, *(double *)a4.m128i_i64, a5);
    }
    return 0;
  }
  if ( (unsigned int)v10 <= 0xF73 )
  {
    switch ( (int)v10 )
    {
      case 3749:
        result = sub_21E1280(
                   a1,
                   0,
                   4u,
                   174 - ((*(_BYTE *)(*(_QWORD *)(a1 + 16) + 936LL) == 0) - 1),
                   a2,
                   *(double *)a3.m128i_i64,
                   *(double *)a4.m128i_i64,
                   a5);
        break;
      case 3750:
        result = sub_21E1280(
                   a1,
                   1u,
                   4u,
                   174 - ((*(_BYTE *)(*(_QWORD *)(a1 + 16) + 936LL) == 0) - 1),
                   a2,
                   *(double *)a3.m128i_i64,
                   *(double *)a4.m128i_i64,
                   a5);
        break;
      case 3751:
        result = sub_21E15D0(
                   a1,
                   176 - ((unsigned int)(*(_BYTE *)(*(_QWORD *)(a1 + 16) + 936LL) == 0) - 1),
                   a2,
                   *(double *)a3.m128i_i64,
                   *(double *)a4.m128i_i64,
                   a5);
        break;
      case 3752:
        result = sub_21E2280(a1, 4u, 178, a2, a3, *(double *)a4.m128i_i64, a5);
        break;
      case 3785:
        result = sub_21E10C0(a1, a2, a3, *(double *)a4.m128i_i64, a5);
        break;
      default:
        return 0;
    }
  }
  else
  {
    switch ( (int)v10 )
    {
      case 3956:
        return sub_21E0D40(a1, a2);
      case 3962:
        return sub_21E0360(
                 a1,
                 0,
                 534 - ((*(_BYTE *)(*(_QWORD *)(a1 + 16) + 936LL) == 0) - 1),
                 a2,
                 *(double *)a3.m128i_i64,
                 *(double *)a4.m128i_i64,
                 a5);
      case 3963:
        return sub_21E0360(
                 a1,
                 1u,
                 534 - ((*(_BYTE *)(*(_QWORD *)(a1 + 16) + 936LL) == 0) - 1),
                 a2,
                 *(double *)a3.m128i_i64,
                 *(double *)a4.m128i_i64,
                 a5);
      case 3964:
        return sub_21E0630(
                 a1,
                 536 - ((*(_BYTE *)(*(_QWORD *)(a1 + 16) + 936LL) == 0) - 1),
                 a2,
                 *(double *)a3.m128i_i64,
                 *(double *)a4.m128i_i64,
                 a5);
      case 3965:
        return sub_21E0630(
                 a1,
                 538 - ((*(_BYTE *)(*(_QWORD *)(a1 + 16) + 936LL) == 0) - 1),
                 a2,
                 *(double *)a3.m128i_i64,
                 *(double *)a4.m128i_i64,
                 a5);
      case 3966:
        return sub_21E0870(a1, 0, 540, a2, a3, *(double *)a4.m128i_i64, a5);
      case 3967:
        return sub_21E0870(a1, 1, 541, a2, a3, *(double *)a4.m128i_i64, a5);
      case 3968:
        return sub_21E0870(a1, 0, 542, a2, a3, *(double *)a4.m128i_i64, a5);
      case 3969:
        return sub_21E0870(a1, 1, 543, a2, a3, *(double *)a4.m128i_i64, a5);
      case 3976:
        return sub_21E0360(
                 a1,
                 0,
                 548 - ((*(_BYTE *)(*(_QWORD *)(a1 + 16) + 936LL) == 0) - 1),
                 a2,
                 *(double *)a3.m128i_i64,
                 *(double *)a4.m128i_i64,
                 a5);
      case 3977:
        return sub_21E0360(
                 a1,
                 1u,
                 548 - ((*(_BYTE *)(*(_QWORD *)(a1 + 16) + 936LL) == 0) - 1),
                 a2,
                 *(double *)a3.m128i_i64,
                 *(double *)a4.m128i_i64,
                 a5);
      case 3978:
        return sub_21E0630(
                 a1,
                 550 - ((*(_BYTE *)(*(_QWORD *)(a1 + 16) + 936LL) == 0) - 1),
                 a2,
                 *(double *)a3.m128i_i64,
                 *(double *)a4.m128i_i64,
                 a5);
      case 3979:
        return sub_21E0630(
                 a1,
                 552 - ((*(_BYTE *)(*(_QWORD *)(a1 + 16) + 936LL) == 0) - 1),
                 a2,
                 *(double *)a3.m128i_i64,
                 *(double *)a4.m128i_i64,
                 a5);
      case 3980:
        return sub_21E0870(a1, 0, 554, a2, a3, *(double *)a4.m128i_i64, a5);
      case 3981:
        return sub_21E0870(a1, 1, 555, a2, a3, *(double *)a4.m128i_i64, a5);
      case 3982:
        return sub_21E0870(a1, 0, 556, a2, a3, *(double *)a4.m128i_i64, a5);
      case 3983:
        return sub_21E0870(a1, 1, 557, a2, a3, *(double *)a4.m128i_i64, a5);
      case 3986:
        return sub_21E0360(
                 a1,
                 0,
                 562 - ((*(_BYTE *)(*(_QWORD *)(a1 + 16) + 936LL) == 0) - 1),
                 a2,
                 *(double *)a3.m128i_i64,
                 *(double *)a4.m128i_i64,
                 a5);
      case 3987:
        return sub_21E0360(
                 a1,
                 1u,
                 562 - ((*(_BYTE *)(*(_QWORD *)(a1 + 16) + 936LL) == 0) - 1),
                 a2,
                 *(double *)a3.m128i_i64,
                 *(double *)a4.m128i_i64,
                 a5);
      case 3988:
        return sub_21E0630(
                 a1,
                 564 - ((*(_BYTE *)(*(_QWORD *)(a1 + 16) + 936LL) == 0) - 1),
                 a2,
                 *(double *)a3.m128i_i64,
                 *(double *)a4.m128i_i64,
                 a5);
      case 3989:
        return sub_21E0630(
                 a1,
                 566 - ((*(_BYTE *)(*(_QWORD *)(a1 + 16) + 936LL) == 0) - 1),
                 a2,
                 *(double *)a3.m128i_i64,
                 *(double *)a4.m128i_i64,
                 a5);
      case 3990:
        return sub_21E0870(a1, 0, 568, a2, a3, *(double *)a4.m128i_i64, a5);
      case 3991:
        return sub_21E0870(a1, 1, 569, a2, a3, *(double *)a4.m128i_i64, a5);
      case 3992:
        return sub_21E0870(a1, 0, 570, a2, a3, *(double *)a4.m128i_i64, a5);
      case 3993:
        return sub_21E0870(a1, 1, 571, a2, a3, *(double *)a4.m128i_i64, a5);
      case 4006:
        v14 = 580;
        v15 = *(_BYTE *)(*(_QWORD *)(a1 + 16) + 936LL) == 0;
        goto LABEL_26;
      case 4007:
        v12 = 580;
        v13 = *(_BYTE *)(*(_QWORD *)(a1 + 16) + 936LL) == 0;
        goto LABEL_24;
      case 4008:
        v18 = 580;
        v19 = *(_BYTE *)(*(_QWORD *)(a1 + 16) + 936LL) == 0;
        goto LABEL_43;
      case 4009:
        v16 = 580;
        v17 = *(_BYTE *)(*(_QWORD *)(a1 + 16) + 936LL) == 0;
        goto LABEL_41;
      case 4010:
        return sub_21E15D0(
                 a1,
                 582 - ((unsigned int)(*(_BYTE *)(*(_QWORD *)(a1 + 16) + 936LL) == 0) - 1),
                 a2,
                 *(double *)a3.m128i_i64,
                 *(double *)a4.m128i_i64,
                 a5);
      case 4011:
        return sub_21E1D20(a1, 1u, 584, a2, a3, *(double *)a4.m128i_i64, a5);
      case 4012:
        return sub_21E1D20(a1, 0, 584, a2, a3, *(double *)a4.m128i_i64, a5);
      case 4014:
        v14 = 587;
        v15 = *(_BYTE *)(*(_QWORD *)(a1 + 16) + 936LL) == 0;
        goto LABEL_26;
      case 4015:
        v12 = 587;
        v13 = *(_BYTE *)(*(_QWORD *)(a1 + 16) + 936LL) == 0;
        goto LABEL_24;
      case 4016:
        v18 = 589;
        v19 = *(_BYTE *)(*(_QWORD *)(a1 + 16) + 936LL) == 0;
        goto LABEL_43;
      case 4017:
        v16 = 589;
        v17 = *(_BYTE *)(*(_QWORD *)(a1 + 16) + 936LL) == 0;
        goto LABEL_41;
      case 4018:
        return sub_21E15D0(
                 a1,
                 591 - ((unsigned int)(*(_BYTE *)(*(_QWORD *)(a1 + 16) + 936LL) == 0) - 1),
                 a2,
                 *(double *)a3.m128i_i64,
                 *(double *)a4.m128i_i64,
                 a5);
      case 4019:
        return sub_21E1D20(a1, 1u, 593, a2, a3, *(double *)a4.m128i_i64, a5);
      case 4020:
        return sub_21E1D20(a1, 0, 593, a2, a3, *(double *)a4.m128i_i64, a5);
      case 4022:
        v14 = 596;
        v15 = *(_BYTE *)(*(_QWORD *)(a1 + 16) + 936LL) == 0;
LABEL_26:
        result = sub_21E1280(a1, 0, 1u, v14 - (v15 - 1), a2, *(double *)a3.m128i_i64, *(double *)a4.m128i_i64, a5);
        break;
      case 4023:
        v12 = 596;
        v13 = *(_BYTE *)(*(_QWORD *)(a1 + 16) + 936LL) == 0;
LABEL_24:
        result = sub_21E1280(a1, 0, 0, v12 - (v13 - 1), a2, *(double *)a3.m128i_i64, *(double *)a4.m128i_i64, a5);
        break;
      case 4024:
        v18 = 598;
        v19 = *(_BYTE *)(*(_QWORD *)(a1 + 16) + 936LL) == 0;
LABEL_43:
        result = sub_21E1280(a1, 1u, 1u, v18 - (v19 - 1), a2, *(double *)a3.m128i_i64, *(double *)a4.m128i_i64, a5);
        break;
      case 4025:
        v16 = 598;
        v17 = *(_BYTE *)(*(_QWORD *)(a1 + 16) + 936LL) == 0;
LABEL_41:
        result = sub_21E1280(a1, 1u, 0, v16 - (v17 - 1), a2, *(double *)a3.m128i_i64, *(double *)a4.m128i_i64, a5);
        break;
      case 4026:
        result = sub_21E15D0(
                   a1,
                   600 - ((unsigned int)(*(_BYTE *)(*(_QWORD *)(a1 + 16) + 936LL) == 0) - 1),
                   a2,
                   *(double *)a3.m128i_i64,
                   *(double *)a4.m128i_i64,
                   a5);
        break;
      case 4027:
        result = sub_21E1D20(a1, 1u, 602, a2, a3, *(double *)a4.m128i_i64, a5);
        break;
      case 4028:
        result = sub_21E1D20(a1, 0, 602, a2, a3, *(double *)a4.m128i_i64, a5);
        break;
      case 4030:
        result = sub_21E1280(
                   a1,
                   0,
                   3u,
                   605 - ((*(_BYTE *)(*(_QWORD *)(a1 + 16) + 936LL) == 0) - 1),
                   a2,
                   *(double *)a3.m128i_i64,
                   *(double *)a4.m128i_i64,
                   a5);
        break;
      case 4031:
        result = sub_21E1280(
                   a1,
                   0,
                   2u,
                   605 - ((*(_BYTE *)(*(_QWORD *)(a1 + 16) + 936LL) == 0) - 1),
                   a2,
                   *(double *)a3.m128i_i64,
                   *(double *)a4.m128i_i64,
                   a5);
        break;
      case 4032:
        result = sub_21E1280(
                   a1,
                   1u,
                   3u,
                   605 - ((*(_BYTE *)(*(_QWORD *)(a1 + 16) + 936LL) == 0) - 1),
                   a2,
                   *(double *)a3.m128i_i64,
                   *(double *)a4.m128i_i64,
                   a5);
        break;
      case 4033:
        result = sub_21E1280(
                   a1,
                   1u,
                   2u,
                   605 - ((*(_BYTE *)(*(_QWORD *)(a1 + 16) + 936LL) == 0) - 1),
                   a2,
                   *(double *)a3.m128i_i64,
                   *(double *)a4.m128i_i64,
                   a5);
        break;
      case 4034:
        result = sub_21E15D0(
                   a1,
                   607 - ((unsigned int)(*(_BYTE *)(*(_QWORD *)(a1 + 16) + 936LL) == 0) - 1),
                   a2,
                   *(double *)a3.m128i_i64,
                   *(double *)a4.m128i_i64,
                   a5);
        break;
      case 4035:
        result = sub_21E1D20(a1, 3u, 609, a2, a3, *(double *)a4.m128i_i64, a5);
        break;
      case 4036:
        result = sub_21E1D20(a1, 2u, 609, a2, a3, *(double *)a4.m128i_i64, a5);
        break;
      case 4095:
        result = sub_21E2840(a1, a2, 1, *(double *)a3.m128i_i64, *(double *)a4.m128i_i64, a5, a2, a8, a9);
        break;
      case 4096:
        result = sub_21E2840(a1, a2, 2, *(double *)a3.m128i_i64, *(double *)a4.m128i_i64, a5, a2, a8, a9);
        break;
      case 4097:
        result = sub_21E2840(a1, a2, 4, *(double *)a3.m128i_i64, *(double *)a4.m128i_i64, a5, a2, a8, a9);
        break;
      case 4111:
        result = sub_21DF500(a1, a2, (__int64)v10, a2, a8, a9);
        break;
      case 4163:
      case 4166:
        result = sub_21E2B60(a1, a2, a3, a4, a5);
        break;
      case 4175:
        result = sub_21E47B0(a1, a2, a3, *(double *)a4.m128i_i64, a5);
        break;
      case 4350:
        result = sub_21E0BD0(a1, a2, a3, *(double *)a4.m128i_i64, a5);
        break;
      case 4405:
        result = sub_21E0020(a1, a2);
        break;
      default:
        return 0;
    }
  }
  return result;
}
