// Function: sub_21C91B0
// Address: 0x21c91b0
//
void __fastcall sub_21C91B0(
        __int64 *a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __m128i a7,
        __m128i a8,
        __m128i a9)
{
  __int16 v9; // ax
  __int64 v10; // rax
  __int64 v11; // r8
  __int64 v12; // r9
  __int64 v13; // r14
  __int64 v14; // rdx
  __int64 v15; // rcx
  __int64 v16; // r8
  __int64 v17; // r9
  __int64 v18; // rax
  __int64 v19; // rdx

  v9 = *(_WORD *)(a2 + 24);
  if ( v9 < 0 )
  {
    *(_DWORD *)(a2 + 28) = -1;
    return;
  }
  if ( v9 == 220 )
    goto LABEL_61;
  if ( v9 <= 220 )
  {
    if ( v9 <= 47 )
    {
      if ( v9 > 10 )
      {
        switch ( v9 )
        {
          case 11:
            if ( !(unsigned __int8)sub_21BE640((__int64)a1, a2, *(double *)a7.m128i_i64, *(double *)a8.m128i_i64, a9) )
              goto LABEL_15;
            break;
          case 43:
            if ( !(unsigned __int8)sub_21BEE20((__int64)a1, a2) )
              goto LABEL_15;
            break;
          case 44:
            if ( !(unsigned __int8)sub_21C72B0(a1, a2, a7, a8, a9, a3, a4, a5) )
              goto LABEL_15;
            break;
          case 46:
            v18 = *(_QWORD *)(a2 + 32);
            v19 = *(_QWORD *)(v18 + 40);
            if ( *(_BYTE *)(*(_QWORD *)(v19 + 40) + 16LL * *(unsigned int *)(v18 + 48)) != 7 )
              goto LABEL_15;
            sub_21BF200((__int64)a1, a2, *(double *)a7.m128i_i64, *(double *)a8.m128i_i64, a9, v19, a4, a5, a6);
            break;
          case 47:
            if ( *(_BYTE *)(*(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a2 + 32) + 40LL) + 40LL)
                          + 16LL * *(unsigned int *)(*(_QWORD *)(a2 + 32) + 48LL)) != 7 )
              goto LABEL_15;
            sub_21BF570((__int64)a1, a2);
            break;
          default:
            goto LABEL_15;
        }
        return;
      }
LABEL_15:
      v10 = sub_21E56B0(a1 + 56, a2);
      v13 = v10;
      if ( v10 )
      {
        sub_1D444E0(a1[34], a2, v10);
        sub_1D49010(v13);
        sub_1D2DC70((const __m128i *)a1[34], a2, v14, v15, v16, v17);
      }
      else
      {
        sub_1D4BB00((__int64)a1, a2, (__int64)&unk_433DA40, 99358, v11, v12, (__m128)a7, a8, a9);
      }
      return;
    }
    if ( v9 == 159 )
    {
      sub_21BEE70(a1, a2, a7, *(double *)a8.m128i_i64, a9);
      return;
    }
    if ( v9 <= 159 )
    {
      if ( v9 != 118 )
      {
        if ( v9 <= 118 )
        {
          if ( v9 == 106 && (unsigned __int8)sub_21BE960((__int64)a1, a2, a3, a4, a5, a6) )
            return;
          goto LABEL_15;
        }
        if ( (unsigned __int16)(v9 - 123) > 1u )
          goto LABEL_15;
      }
      if ( (unsigned __int8)sub_21C2320((__int64)a1, a2, a7, *(double *)a8.m128i_i64, a9) )
        return;
      goto LABEL_15;
    }
    if ( v9 != 186 )
    {
      if ( (v9 == 219 || v9 == 185) && (unsigned __int8)sub_21C72F0((__int64)a1, a2, a7, a8, a9) )
        return;
      goto LABEL_15;
    }
LABEL_61:
    if ( (unsigned __int8)sub_21C2FA0((__int64)a1, a2, a7, *(double *)a8.m128i_i64, a9) )
      return;
    goto LABEL_15;
  }
  if ( v9 > 677 )
  {
    if ( v9 <= 857 )
    {
      if ( v9 > 683 && (unsigned __int8)sub_21C15C0((__int64)a1, a2) )
        return;
    }
    else if ( (unsigned __int16)(v9 - 858) <= 0xC5u && (unsigned __int8)sub_21C1860((__int64)a1, a2) )
    {
      return;
    }
    goto LABEL_15;
  }
  if ( v9 <= 658 )
  {
    if ( v9 == 298 )
    {
      sub_21BE780((__int64)a1, a2, a7, *(double *)a8.m128i_i64, a9);
      return;
    }
    goto LABEL_15;
  }
  switch ( v9 )
  {
    case 661:
    case 662:
    case 663:
    case 664:
      if ( !(unsigned __int8)sub_21C5A60(a1, a2, a7, a8, a9, a3, a4, a5) )
        goto LABEL_15;
      break;
    case 665:
    case 666:
      if ( !(unsigned __int8)sub_21C3D80((__int64)a1, a2, *(double *)a7.m128i_i64, a8, a9) )
        goto LABEL_15;
      break;
    case 667:
    case 668:
    case 669:
      if ( !(unsigned __int8)sub_21BF6F0((__int64)a1, a2, *(double *)a7.m128i_i64, *(double *)a8.m128i_i64, a9) )
        goto LABEL_15;
      break;
    case 670:
    case 671:
    case 672:
    case 673:
    case 674:
      if ( !(unsigned __int8)sub_21C09D0((__int64)a1, a2, a7) )
        goto LABEL_15;
      break;
    case 675:
    case 676:
    case 677:
      if ( !(unsigned __int8)sub_21BFCB0((__int64)a1, a2, a7, *(double *)a8.m128i_i64, a9) )
        goto LABEL_15;
      break;
    default:
      if ( !(unsigned __int8)sub_21C80B0((__int64)a1, a2, a7, a8, a9) )
        goto LABEL_15;
      break;
  }
}
